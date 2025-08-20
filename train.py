from pprint import pprint
from typing import Tuple, Dict

from dataclasses import asdict
import gin
import json
import datetime
import torch
import numpy as np
from sklearn.model_selection import train_test_split

import offline_moo.off_moo_bench as ob
from offline_moo.utils import get_quantile_solutions
from offline_moo.off_moo_bench.task_set import ALLTASKSDICT
from offline_moo.off_moo_bench.evaluation.metrics import hv

from models import elucidated_diffusion, diffusion_utils
from models.model_helpers import (
    TaskConfig,
    parse_args,
    set_seed,
    reweight_multi_objective,
    sample_uniform_toward_ideal,
    get_slurm_job_id,
    get_slurm_task_id
)


def create_task(config: TaskConfig) -> Tuple[
        object,
        np.ndarray,
        np.ndarray,
        np.ndarray,
]:
    """
    Create and prepare a task dataset based on the given configuration.

    Returns:
        task (object): Instantiated task object.
        X (np.ndarray): Input features (normalized and reshaped if needed).
        y (np.ndarray): Target objective values.
        d_best (np.ndarray): Non-dominated (Pareto-optimal) objectives,
                             normalized if config.normalize_ys is True.
    """
    task_name = config.task_name.lower()
    if task_name in ALLTASKSDICT:
        task_name = ALLTASKSDICT[task_name]
    else:
        raise ValueError(
            f"Task '{config.task_name}' not found in ALLTASKSDICT. "
            f"Available tasks: {list(ALLTASKSDICT.keys())}"
        )

    task = ob.make(task_name)

    X = task.x.copy()
    y = task.y.copy()

    if config.data_pruning:
        X, y = task.get_N_non_dominated_solutions(
            N=int(X.shape[0] * config.data_preserved_ratio),
            return_x=True,
            return_y=True,
        )

    if task.is_discrete:
        task.map_to_logits()
        X = task.to_logits(X)
        data_size, n_dim, n_classes = tuple(X.shape)
        X = X.reshape(-1, n_dim * n_classes)

    if task.is_sequence:
        X = task.to_logits(X)

    data_size, n_dim = X.shape
    n_obj = y.shape[1]

    print(
        f"Task: {task_name}\n"
        f"Data size: {data_size}\n"
        f"Number of objectives: {n_obj}\n"
        f"Number of dimensions: {n_dim}\n"
    )
    print()

    _, d_best = task.get_N_non_dominated_solutions(
        N=config.num_pareto_solutions, return_x=False, return_y=True
    )

    if config.normalize_xs:
        task.map_normalize_x()
        X = task.normalize_x(X, config.normalize_method_xs)

    if config.normalize_ys:
        task.map_normalize_y()
        y = task.normalize_y(y, config.normalize_method_ys)
        d_best = task.normalize_y(d_best, config.normalize_method_ys)

    X = X.astype(np.float32)
    y = y.astype(np.float32)
    d_best = d_best.astype(np.float32)

    return task, X, y, d_best


def train_diffusion(
    config,
    X: np.ndarray,
    y: np.ndarray,
):
    """
    Train a diffusion model based on the given configuration and dataset.

    Args:
        config (TaskConfig): Configuration parameters for training.
        X (np.ndarray): Input feature data.
        y (np.ndarray): Target values.

    Returns:
        elucidated_diffusion.Trainer: Trainer object containing the trained
        model, training state and the EMA-smoothed model for evaluation.
        You can access `trainer.ema.ema_model` for evaluation or
        `trainer.val_loss` for metrics.
    """
    if config.use_val_split:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=config.val_ratio, random_state=config.seed
        )
    else:
        X_train, y_train = X, y
        X_val, y_val = None, None

    print(f"Train data shape: {X_train.shape}, dtype: {X_train.dtype}")
    if X_val is not None:
        print(f"Validation data shape: {X_val.shape}, dtype: {X_val.dtype}")

    gin.parse_config_files_and_bindings(
        config.gin_config_files,
        config.gin_params
    )
    
    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).float()
    if config.reweight_loss:
        print("Using reweighted loss")
        weights = reweight_multi_objective(y_train, num_bins=20)
        weights_tensor = torch.from_numpy(weights).float()
    else:        
        print("Using standard loss")
        weights_tensor = torch.ones(X_train_tensor.shape[0]).float()

    assert weights_tensor.shape[0] == X_train_tensor.shape[0], (
            f"Mismatch of shapes for X and weights: {X_train_tensor.shape=} "
            f"!= {weights_tensor.shape=}"
    )

    train_dataset = torch.utils.data.TensorDataset(
        X_train_tensor,
        y_train_tensor,
        weights_tensor
    )
    if X_val is not None and y_val is not None:
        X_val_tensor = torch.from_numpy(X_val).float()
        y_val_tensor = torch.from_numpy(y_val).float()
        val_weights = torch.ones(y_val.shape[0]).float() 
        val_dataset = torch.utils.data.TensorDataset(
            X_val_tensor,
            y_val_tensor,
            val_weights
        )
    else:
        val_dataset = None

    diffusion = diffusion_utils.construct_diffusion_model(
        inputs=X_train_tensor,
        cond_dim=y_train.shape[1],
    )

    trainer = elucidated_diffusion.Trainer(
        diffusion,
        train_dataset,
        val_dataset,
        use_wandb=config.use_wandb
    )

    trainer.train()

    return trainer


def sampling(
    task,
    config,
    diffusion,
    d_best: np.ndarray,
    guidance_scale: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate samples from the
    diffusion modelconditioned on extrapolated points.

    Args:
        config (TaskConfig): Sampling parameters and settings.
        d_best (np.ndarray): Best objective values for conditioning.
        diffusion: Diffusion model with a sample method.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            Generated samples and predicted targets.

    Raises:
        ValueError:
            If num_pareto_solutions is not divisible by conditioning points.
    """
    # Sample extrapolated conditioning points
    cond_points = sample_uniform_toward_ideal(
        d_best=d_best,
        k=32,
    )

    cond_points_tensor = torch.from_numpy(cond_points).float()

    if config.num_pareto_solutions % cond_points_tensor.shape[0] != 0:
        raise ValueError(
            f"num_pareto_solutions ({config.num_pareto_solutions}) must be "
            f"divisible by conditioning points ({cond_points_tensor.shape[0]})"
        )

    if cond_points_tensor.shape[0] != config.num_pareto_solutions:
        batch_interleave = (
            config.num_pareto_solutions // cond_points_tensor.shape[0]
        )
        cond_points_tensor = cond_points_tensor.repeat_interleave(
            batch_interleave, dim=0
        )

    res_x = diffusion.sample(
        batch_size=cond_points_tensor.shape[0],
        cond=cond_points_tensor,
        guidance_scale=guidance_scale,
        clamp=False,
    )

    res_x = res_x.cpu().numpy()

    if config.normalize_xs:
        task.map_denormalize_x()
        res_x = task.denormalize_x(res_x)

    if task.is_sequence:
        res_x = task.to_integers(res_x)

    res_y = task.predict(res_x)

    return res_x, res_y


def evaluation(
    task,
    config,
    res_y: np.ndarray,
) -> Dict:
    """
    Evaluate generated samples using various multi-objective metrics.

    Args:
        task: Task object with normalization and prediction methods.
        config (TaskConfig): Configuration object.
        res_y (np.ndarray): Corresponding predicted objectives.

    Returns:
        Dict[str, float]: Dictionary containing HV metrics:
            - hv_d_best: HV of d_best
            - hv_100th: HV of all predicted samples
            - hv_75th: HV of top 75% solutions
            - hv_50th: HV of top 50% solutions
    """
    res_y_75_percent = get_quantile_solutions(res_y, 0.75)
    res_y_50_percent = get_quantile_solutions(res_y, 0.50)

    # For calculating hypervolume, we use the min-max normalization
    res_y = task.normalize_y(res_y, normalization_method="min-max")
    res_y_50_percent = task.normalize_y(
        res_y_50_percent, normalization_method="min-max"
    )
    res_y_75_percent = task.normalize_y(
        res_y_75_percent, normalization_method="min-max"
    )

    task_name = config.task_name.lower()

    nadir_point = task.nadir_point
    nadir_point = task.normalize_y(nadir_point, normalization_method="min-max")

    _, d_best = task.get_N_non_dominated_solutions(
         N=256,
         return_x=False,
         return_y=True
    )
    d_best = task.normalize_y(d_best, normalization_method="min-max")

    # Hypervolume (Normalized)
    d_best_hv = hv(nadir_point, d_best, task_name)
    hv_value = hv(nadir_point, res_y, task_name)
    hv_value_50_percentile = hv(nadir_point, res_y_50_percent, task_name)
    hv_value_75_percentile = hv(nadir_point, res_y_75_percent, task_name)

    # Save the results
    results = {
        "hv_d_best": d_best_hv,
        "hv_100th": hv_value,
        "hv_75th": hv_value_75_percentile,
        "hv_50th": hv_value_50_percentile,
    }

    return results


def setup_wandb(config):
    now = datetime.datetime.now()
    ts = now.strftime("%Y-%m-%dT%H-%M")

    cfg = asdict(config)


    cfg.update({
        "slurm_job_id": get_slurm_job_id(), 
        "slurm_array_task_id": get_slurm_task_id()
    })
    run_name = f"{config.task_name}-{config.seed}"
    group = f"{config.domain}-{config.task_name}-{ts}"
    wandb.init(
            run_name=run_name, 
            job_type="train", 
            config=config,
            tags=[config.task_name, config.domain],
            save_code=False
    )

def print_results(results, config):
    print("-" * 40)
    print(f"Task: {config.task_name.upper()}")
    print(f"{'Metric':<25} {'Value':>10}")
    print("-" * 40)
    print(f"{'Hypervolume (D(best))':<25} {results['hv_d_best']:10.4f}")
    print(f"{'Hypervolume (100th)':<25} {results['hv_100th']:10.4f}")
    print(f"{'Hypervolume (75th)':<25} {results['hv_75th']:10.4f}")
    print(f"{'Hypervolume (50th)':<25} {results['hv_50th']:10.4f}")
    print("-" * 40)
    print("Note: Higher HV = better diversity & convergence")


def main():
    config = parse_args()
    set_seed(config.seed)

    print("Configuration:")
    pprint(asdict(config))
    print()
        
    if config.use_wandb:
        setup_wandb(config)

    task, X, y, d_best = create_task(config)

    trainer = train_diffusion(config, X, y)
    ema_model = trainer.ema.ema_model

    res_x, res_y = sampling(task, config, ema_model, d_best)
    results = evaluation(task, config, res_y)
    if config.use_wandb:
        wandb.log(results)

    print()
    print_results(results, config)
    
    if config.save_dir is not None:
        with (config.save_dir / "results.json").open('w') as ofstream:
            # Ensure that the results do not contain e.g. numpy objects
            payload = {key: float(val) for key, val in results.items()} 
            json.dump(payload, ofstream)


if __name__ == "__main__":
    main()
