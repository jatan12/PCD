import json
import pathlib
from dataclasses import asdict
from pprint import pprint
from typing import Dict, Tuple

import gin
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.model_selection import train_test_split

import offline_moo.off_moo_bench as ob
import wandb
from models import diffusion_utils, elucidated_diffusion
from models.model_helpers import (
    TaskConfig,
    get_slurm_job_id,
    get_slurm_task_id,
    parse_args,
    reweight_multi_objective,
    sample_along_ref_dirs,
    sample_uniform_direction,
    sample_uniform_toward_ideal,
    set_seed,
)
from offline_moo.off_moo_bench.evaluation.metrics import hv
from offline_moo.off_moo_bench.task_set import ALLTASKSDICT
from offline_moo.utils import get_quantile_solutions

# Configure evobench
try:
    from evobenchx.database.init import config as econfig

    off_moo_dir = (
        pathlib.Path(__file__) / "offline_moo" / "off_moo_bench" / "problem" / "mo_nas"
    ).resolve()

    db_path = off_moo_dir / "database"
    data_path = off_moo_dir / "data"
    if not db_path.exists():
        print(f"EvoBenchX: {str(db_path)!r} does not exist!")
        raise

    if not data_path.exists():
        print(f"EvoBenchX: {str(data_path)!r} does not exist!")
        raise
    #  econfig(str(db_path), str(data_path))
except Exception as e:
    print(f"Could  not configure EvoBenchX! ({e}) Continuing without it!")

palette = sns.color_palette("colorblind")
COLORS = {
    "blue": palette[0],
    "light-orange": palette[1],
    "green": palette[2],
    "orange": palette[3],
    "purple": palette[4],
    "brown": palette[5],
    "pink": palette[6],
    "grey": palette[7],
    "yellow": palette[8],
    "light-blue": palette[9],
}


def create_task(
    config: TaskConfig,
) -> Tuple[
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
        print("Normalizing x values!")
        # task.map_normalize_x()
        X = task.normalize_x(X, normalization_method=config.normalize_method_xs)

    if config.normalize_ys:
        print("Normalizing y values!")
        # task.map_normalize_y()
        y = task.normalize_y(y, normalization_method=config.normalize_method_ys)
        d_best = task.normalize_y(
            d_best, normalization_method=config.normalize_method_ys
        )

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

    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).float()
    if config.reweight_loss:
        print("Using reweighted loss")
        weights = reweight_multi_objective(y_train)
        print(
            f" mu {weights.mean():.2f}, sd {np.std(weights):.2f}, ({weights.min():.2f}, {weights.max():.2f})"
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
        X_train_tensor, y_train_tensor, weights_tensor
    )
    if X_val is not None and y_val is not None:
        X_val_tensor = torch.from_numpy(X_val).float()
        y_val_tensor = torch.from_numpy(y_val).float()
        val_weights = torch.ones(y_val.shape[0]).float()
        val_dataset = torch.utils.data.TensorDataset(
            X_val_tensor, y_val_tensor, val_weights
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
        results_folder=config.save_dir,
    )

    trainer.train()

    return trainer


def sampling(
    task,
    config,
    diffusion,
    guidance_scale: float,
    d_best: np.ndarray,
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
    # assert config.sampling_method in ("uniform-ideal", "uniform-direction", )

    if config.sampling_method == "uniform-ideal":
        cond_points = sample_uniform_toward_ideal(
            d_best=d_best, k=config.num_cond_points
        )
    elif config.sampling_method == "uniform-direction":
        cond_points = sample_uniform_direction(
            d_best=d_best, k=config.num_cond_points, alpha=0.4
        )
    elif config.sampling_method == "reference-direction":
        cond_points = sample_along_ref_dirs(
            d_best=d_best,
            k=config.num_cond_points,
            num_points=config.num_pareto_solutions,
        )
    else:
        assert False, config.sampling_method

    cond_points_tensor = torch.from_numpy(cond_points).float()

    if config.num_pareto_solutions % cond_points_tensor.shape[0] != 0:
        raise ValueError(
            f"num_pareto_solutions ({config.num_pareto_solutions}) must be "
            f"divisible by conditioning points ({cond_points_tensor.shape[0]})"
        )

    if cond_points_tensor.shape[0] != config.num_pareto_solutions:
        batch_interleave = config.num_pareto_solutions // cond_points_tensor.shape[0]
        cond_points_tensor = cond_points_tensor.repeat_interleave(
            batch_interleave, dim=0
        )

    print(
        f"Sampling for {cond_points_tensor.shape[0]} solutions! "
        f"with {cond_points_tensor.shape=} and scale {guidance_scale:.2f}"
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

    # Fix issue where some MONAS benchmarks will fail the conversion, so
    # run the results one by one
    if config.domain == "monas":
        res_y = []
        total_failed = 0
        for i in range(config.num_pareto_solutions):
            try:
                y_i = task.predict(res_x[i, :])
                res_y.append(y_i)
            except Exception:
                print(f"Failed to convert solution {i}")
                total_failed += 1
        print(
            f"In total {total_failed} / {config.num_pareto_solutions} solutions failed"
        )
        res_y = np.asarray(res_y)
    else:
        res_y = task.predict(res_x)

    return res_x, res_y, cond_points


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

    _, d_best = task.get_N_non_dominated_solutions(N=256, return_x=False, return_y=True)
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
    # now = datetime.datetime.now()
    # ts = now.strftime("%Y-%m-%dT%H-%M")
    exclude_list = (
        "gin_config_files",
        "gin_params",
        "use_wandb",
        "save_dir",
        "experiment_name",
    )

    cfg = asdict(
        config, dict_factory=lambda x: {k: v for (k, v) in x if k not in exclude_list}
    )

    cfg.update(
        {
            "slurm_job_id": get_slurm_job_id(),
            "slurm_array_task_id": get_slurm_task_id(),
            "reweight_num_bins": gin.query_parameter(
                "reweight_multi_objective.num_bins"
            ),
            "reweight_k": gin.query_parameter("reweight_multi_objective.k"),
            "reweight_tau": gin.query_parameter("reweight_multi_objective.tau"),
            "reweight_normalize_dom_counts": gin.query_parameter(
                "reweight_multi_objective.normalize_dom_counts"
            ),
        }
    )
    run_name = f"{config.task_name}-{config.seed}"
    experiment_name = f"{config.domain}-{config.task_name}-{config.experiment_name}"
    wandb.init(
        name=run_name,
        job_type="train",
        config=config,
        group=experiment_name,
        tags=[config.task_name, config.domain],
        save_code=False,
    )


def plot_results(d_best, cond_points, res_y, config, save_dir):
    print(f"D-best: {d_best.shape}")

    y_color = COLORS["purple"]
    d_best_color = COLORS["blue"]
    cond_point_color = COLORS["orange"]

    if d_best.shape[1] == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(
            d_best[:, 0],
            d_best[:, 1],
            d_best[:, 2],
            color=d_best_color,
            label="d-best",
        )
        ax.scatter(
            cond_points[:, 0],
            cond_points[:, 1],
            cond_points[:, 2],
            color=cond_point_color,
            label="cond-points",
        )

        ax.scatter(
            res_y[:, 0],
            res_y[:, 1],
            res_y[:, 2],
            color=y_color,
            label="y",
        )
        ax.legend(fontsize="large")
        ax.grid(True, alpha=0.25)
        ax.set_title(config.task_name, fontsize="x-large")
    elif d_best.shape[1] == 2:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.scatter(
            d_best[:, 0],
            d_best[:, 1],
            color=d_best_color,
            label="d-best",
        )
        ax.scatter(
            cond_points[:, 0],
            cond_points[:, 1],
            color=cond_point_color,
            label="cond-points",
        )

        ax.scatter(
            res_y[:, 0],
            res_y[:, 1],
            color=y_color,
            label="y",
        )
        ax.legend(fontsize="large")
        ax.grid(True, alpha=0.25)
        ax.set_title(config.task_name, fontsize="x-large")

    else:
        fig, axs = plt.subplots(
            d_best.shape[1], d_best.shape[1], figsize=(20, 20), constrained_layout=True
        )
        fig.suptitle(config.task_name)

        for i in range(d_best.shape[1]):
            for j in range(d_best.shape[1]):
                if i == j:
                    continue
                axs[i, j].scatter(
                    d_best[:, i], d_best[:, j], color=d_best_color, label="d_best"
                )
                axs[i, j].scatter(
                    cond_points[:, i],
                    cond_points[:, j],
                    color=cond_point_color,
                    label="cond-points",
                )
                axs[i, j].scatter(
                    res_y[:, i],
                    res_y[:, j],
                    color=y_color,
                    label="y",
                )
                axs[i, j].grid(True, alpha=0.25)
                axs[i, j].set_title(f"Obj {i + 1} vs {j + 1}")
                axs[i, j].legend(fontsize="large")
        fig.subplots_adjust(wspace=0.4, hspace=0.4)

    fig.savefig(save_dir / "pareto_front.png", dpi=400, bbox_inches="tight")
    fig.savefig(save_dir / "pareto_front.svg", transparent=True, bbox_inches="tight")


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

    # Setup configuration
    gin.parse_config_files_and_bindings(config.gin_config_files, config.gin_params)

    if config.use_wandb:
        setup_wandb(config)

    task, X, y, d_best = create_task(config)

    trainer = train_diffusion(config, X, y)
    ema_model = trainer.ema.ema_model

    # Sample the model with different guidance scales
    res_x, res_y, cond_points = sampling(
        task, config, ema_model, guidance_scale=config.guidance_scale, d_best=d_best
    )
    results = evaluation(task, config, res_y)
    # results["guidance_scale"] = scale
    results = {key: float(val) for key, val in results.items()}

    if config.use_wandb:
        wandb.log(results)

    print()
    print_results(results, config)

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

        with (config.save_dir / "config.json").open("w") as ofstream:
            json.dump(cfg_dct, ofstream)

        res_y = np.asarray(res_y)
        res_x = np.asarray(res_x)
        cond_points = np.asarray(cond_points)

        if config.normalize_ys:
            plot_y = task.normalize_y(res_y)
        else:
            plot_y = res_y

        # Save the results and plot the D-best paretoflow + the actual points
        plot_results(
            d_best,
            cond_points=cond_points,
            res_y=plot_y,
            config=config,
            save_dir=config.save_dir,
        )

        np.savez(
            config.save_dir / "data.npz",
            d_best=d_best,
            res_y=res_y,
            res_x=res_x,
            cond_points=cond_points,
        )

        with (config.save_dir / "results.json").open("w") as ofstream:
            # Ensure that the results do not contain e.g. numpy objects
            json.dump(results, ofstream)


if __name__ == "__main__":
    main()
