from pprint import pprint
from dataclasses import asdict
from typing import Tuple, Dict, List, Optional

import json
import pathlib
import gin
import wandb
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
    setup_wandb,
    reweight_multi_objective,
    sample_along_ref_dirs,
)


def create_task(config: TaskConfig) -> Tuple[
    object,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
    Optional[int],
]:
    """
    Create and prepare a task dataset based on the given configuration.

    Returns:
        task (object): Instantiated task object.
        X (np.ndarray): Input features (normalized and reshaped if needed).
        y (np.ndarray): Target objective values.
        cond_points (np.ndarray): Top k Non-dominated (Pareto-optimal) points,
                                  normalized if config.normalize_ys is True.
        n_dim (int): Number of decision variables (before flattening).
        n_classes (Optional[int]): Number of classes (only for discrete tasks).
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

    n_dim, n_classes = None, None

    if task.is_discrete:
        X = task.to_logits(X)
        data_size, n_dim, n_classes = tuple(X.shape)   # preserve original info
        X = X.reshape(-1, n_dim * n_classes)
    elif task.is_sequence:
        X = task.to_logits(X)
        data_size, n_dim = X.shape
    else:
        data_size, n_dim = X.shape

    n_obj = y.shape[1]

    print(
        f"Task: {task_name}\n"
        f"Data size: {data_size}\n"
        f"Number of objectives: {n_obj}\n"
        f"Number of dimensions: {n_dim}\n"
        f"Number of classes: {n_classes if n_classes is not None else 'N/A'}\n"
    )
    print()

    # Pick the top N non-dominated solutions for conditioning
    _, d_best = task.get_N_non_dominated_solutions(
        N=config.num_pareto_solutions, return_x=False, return_y=True
    )

    # Utopia point, ideal point for each objectives
    # utopia_point = np.min(y, axis=0).reshape(1, -1)

    if config.normalize_xs:
        task.map_normalize_x()
        X = task.normalize_x(X, config.normalize_method_xs)

    if config.normalize_ys:
        task.map_normalize_y()
        y = task.normalize_y(y, config.normalize_method_ys)
        d_best = task.normalize_y(d_best, config.normalize_method_ys)
        # utopia_point = task.normalize_y(utopia_point, config.normalize_method_ys)

    X = X.astype(np.float32)
    y = y.astype(np.float32)
    d_best = d_best.astype(np.float32)

    return task, X, y, d_best, n_dim, n_classes


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
            X, y, test_size=config.val_ratio,
            random_state=config.seed, shuffle=True,
        )
    else:
        X_train, y_train = X, y
        X_val, y_val = None, None

    print(f"Train data shape: {X_train.shape}, dtype: {X_train.dtype}")
    if X_val is not None:
        print(f"Validation data shape: {X_val.shape}, dtype: {X_val.dtype}")

    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).float()

    if config.reweight_loss:
        print("Using reweighted loss")
        weights = reweight_multi_objective(y_train)  # Hardcoded num_bins=20?
        print(
            (
                f" mu {weights.mean():.2f}, sd {np.std(weights):.2f}, "
                f"({weights.min():.2f}, {weights.max():.2f})"
            )
        )
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
        weights_tensor,
    )

    if X_val is not None and y_val is not None:
        X_val_tensor = torch.from_numpy(X_val).float()
        y_val_tensor = torch.from_numpy(y_val).float()
        val_weights = torch.ones(y_val.shape[0]).float()
        val_dataset = torch.utils.data.TensorDataset(
            X_val_tensor,
            y_val_tensor,
            val_weights,
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
        use_wandb=config.use_wandb,
        results_folder=config.model_dir,
    )

    trainer.train()

    return trainer


def sampling(
    task,
    config,
    diffusion_model,
    d_best: np.ndarray,
    n_dim: Optional[int],
    n_classes: Optional[int],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Generate samples from the
    diffusion model conditioned on extrapolated points.

    Args:
        config (TaskConfig): Sampling parameters and settings.
        cond_points (np.ndarray): Best objective values for conditioning.
        diffusion: Diffusion model with a sample method.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]:
            Generated samples and predicted targets per guidance scale

    Raises:
        ValueError:
            If num_pareto_solutions is not divisible by conditioning points.
    """
    # Use the reference directions for conditioning points
    cond_points = sample_along_ref_dirs(
            d_best=d_best,
            k=config.num_cond_points,
            num_points=config.num_pareto_solutions,
            noise_scale=config.sampling_noise_scale,
            seed=config.seed,
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

    res_x_all, res_y_all = [], []

    for scale in config.guidance_scales:
        print(
            f"Sampling {cond_points_tensor.shape[0]} solutions "
            f"with {cond_points_tensor.shape=} and guidance_scale={scale:.2f}"
        )

        res_x = diffusion_model.sample(
            batch_size=cond_points_tensor.shape[0],
            cond=cond_points_tensor,
            guidance_scale=scale,
            clamp=False,
        ).cpu().numpy()

        if config.normalize_xs:
            res_x = task.denormalize_x(res_x)

        if task.is_discrete:
            if n_dim is None or n_classes is None:
                raise ValueError(
                    "n_dim and n_classes must be provided for discrete tasks."
                )
            res_x = res_x.reshape(-1, n_dim, n_classes)
            res_x = task.to_integers(res_x)
        elif task.is_sequence:
            res_x = task.to_integers(res_x)

        res_y = task.predict(res_x)

        # Filter out invalid solutions
        visible_masks = np.ones(len(res_y))
        visible_masks[np.where(np.logical_or(np.isinf(res_y), np.isnan(res_y)))[0]] = 0
        visible_masks[np.where(np.logical_or(np.isinf(res_x), np.isnan(res_x)))[0]] = 0
        res_x = res_x[np.where(visible_masks == 1)[0]]
        res_y = res_y[np.where(visible_masks == 1)[0]]

        res_x_all.append(res_x)
        res_y_all.append(res_y)

    # Concatenate results from all guidance scales
    # Shape: (num_scales, num_solutions, ...)
    # res_x_all = np.array(res_x_all)
    # res_y_all = np.array(res_y_all)

    return res_x_all, res_y_all


def evaluation(
    task,
    config,
    res_y_all: List[np.ndarray],
) -> Dict[float, Dict[str, float]]:
    """
    Evaluate generated samples using various multi-objective metrics.

    Args:
        task: Task object with normalization and prediction methods.
        config (TaskConfig): Configuration object.
        res_y_all (np.ndarray): Corresponding predicted objectives.
            Expected shape: (num_scales, num_solutions, num_objectives)

    Returns:
        Dict[float, Dict[str, float]]: Dictionary containing HV metrics:
            - hv_d_best: HV of d_best
            - hv_100th: HV of all predicted samples
            - hv_75th: HV of top 75% solutions
            - hv_50th: HV of top 50% solutions
    """
    task_name = config.task_name.lower()
    if task_name in ALLTASKSDICT:
        task_name = ALLTASKSDICT[task_name]
    else:
        raise ValueError(
            f"Task '{config.task_name}' not found in ALLTASKSDICT. "
            f"Available tasks: {list(ALLTASKSDICT.keys())}"
        )

    # Get top 256 Pareto solutions (d_best) & nadir point
    _, d_best = task.get_N_non_dominated_solutions(
        N=256,
        return_x=False,
        return_y=True
    )
    nadir_point = task.nadir_point

    # For evaluation, we use the min-max normalization explicitly
    d_best = task.normalize_y(d_best, normalization_method="min-max")
    nadir_point = task.normalize_y(nadir_point, normalization_method="min-max")
    nadir_point = nadir_point.reshape(-1, )  # Ensure 1D array
    hv_d_best = hv(nadir_point, d_best, task_name)

    results_per_scale = {}
    for scale, res_y in zip(config.guidance_scales, res_y_all):
        res_y_norm = task.normalize_y(res_y, normalization_method="min-max")
        res_y_75_norm = get_quantile_solutions(res_y_norm, 0.75)
        res_y_50_norm = get_quantile_solutions(res_y_norm, 0.50)

        results_per_scale[scale] = {
            "hv_d_best": hv_d_best,
            "hv_100th": hv(nadir_point, res_y_norm, task_name),
            "hv_75th": hv(nadir_point, res_y_75_norm, task_name),
            "hv_50th": hv(nadir_point, res_y_50_norm, task_name),
        }

    return results_per_scale


def print_results(results, config):
    print("-" * 60)
    print(f"Task: {config.task_name.upper()}")
    print(f"{'Guidance Scale':<15} {'HV D(best)':>10} {'HV 100th':>10} {'HV 75th':>10} {'HV 50th':>10}")
    print("-" * 60)

    for scale, metrics in results.items():
        print(f"{scale:<15.2f} {metrics['hv_d_best']:10.4f} {metrics['hv_100th']:10.4f} "
              f"{metrics['hv_75th']:10.4f} {metrics['hv_50th']:10.4f}")

    print("-" * 60)
    print("Note: Higher HV = better diversity & convergence")


def make_json_serializable(obj):
    if isinstance(obj, pathlib.Path):
        return str(obj)
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    else:
        return obj


def main():
    config = parse_args()
    set_seed(config.seed)

    print("Configuration:")
    pprint(asdict(config))
    print()

    if config.use_wandb:
        setup_wandb(config)

    # Modify this as you wish!
    suffix = f"seed_{config.seed}_reweight_{config.reweight_loss}"

    if config.model_dir is not None:
        config.model_dir = config.model_dir / f"{config.task_name}_{suffix}"
        config.model_dir.mkdir(parents=True, exist_ok=True)

    # Create and prepare the task dataset
    (
        task,
        X,
        y,
        d_best,
        n_dim,
        n_classes,
    ) = create_task(config)

    # Load model config from gin files
    gin.parse_config_files_and_bindings(
        config.gin_config_files,
        config.gin_params
    )

    # Train the diffusion model
    trainer = train_diffusion(config, X, y)
    ema_model = trainer.ema.ema_model  # use the EMA model for evaluation

    # Sampling and evaluation with different guidance scales
    res_x_all, res_y_all = sampling(task, config, ema_model,
                                    d_best, n_dim, n_classes)
    results = evaluation(task, config, res_y_all)

    print()
    print_results(results, config)

    if config.use_wandb:
        wandb.log(results)

    if config.save_dir is not None:
        # Save the configuration
        exclude_list = (
            "gin_config_files",
            "gin_params",
            "use_wandb",
            "save_dir",
        )

        cfg_dct = asdict(
            config,
            dict_factory=lambda x: {k: v for (k, v) in x if k not in exclude_list},
        )

        # Query some params to the configuration
        cfg_dct = {
            "train_num_steps": gin.query_parameter("Trainer.train_num_steps"),
            "lr": gin.query_parameter("Trainer.train_lr"),
            "mlp_width": gin.query_parameter("ResidualMLPDenoiser.mlp_width"),
            "num_layers": gin.query_parameter("ResidualMLPDenoiser.num_layers"),
            "time_dim": gin.query_parameter("ResidualMLPDenoiser.dim_t"),
            "reweight_num_bins": gin.query_parameter(
                "reweight_multi_objective.num_bins"
            ),
            "reweight_k": gin.query_parameter("reweight_multi_objective.k"),
            "reweight_tau": gin.query_parameter("reweight_multi_objective.tau"),
            "reweight_normalize_dom_counts": gin.query_parameter(
                "reweight_multi_objective.normalize_dom_counts"
            ),
            **cfg_dct,
        }

        # suffix = f"seed_{config.seed}_reweight_{config.reweight_loss}"

        with (config.save_dir / f"{config.task_name}_{suffix}_config.json").open("w") as ofstream:
            json.dump(make_json_serializable(cfg_dct), ofstream, indent=2)

        with (config.save_dir / f"{config.task_name}_{suffix}_results.json").open("w") as ofstream:
            json.dump(make_json_serializable(results), ofstream, indent=2)

        np.savez(
            config.save_dir / f"{config.task_name}_{suffix}_data.npz",
            res_x=res_x_all,
            res_y=res_y_all,
        )


if __name__ == "__main__":
    main()
