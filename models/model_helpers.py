import argparse
import datetime
import os
import pathlib
import random
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import gin
import numpy as np
import scipy
import torch
from pygmo import fast_non_dominated_sorting
from pymoo.algorithms.moo.nsga2 import calc_crowding_distance
from pymoo.factory import get_reference_directions
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


@dataclass
class TaskConfig:
    seed: int = 42
    task_name: str = ""
    domain: str = ""
    sampling_method: Literal["uniform-ideal", "uniform-angle"] = "uniform-ideal"
    guidance_scale: List[float] = field(default_factory=lambda: [1.0])
    reweight_loss: bool = False
    data_pruning: bool = False
    data_preserved_ratio: float = 0.2
    normalize_xs: bool = False
    normalize_ys: bool = False
    normalize_method_xs: str = "z-score"
    normalize_method_ys: str = "min-max"
    num_pareto_solutions: int = 256
    use_val_split: bool = True
    val_ratio: float = 0.2
    gin_config_files: List[str] = field(default_factory=list)
    gin_params: List[str] = field(default_factory=list)
    use_wandb: bool = False
    experiment_name: Optional[str] = None
    save_dir: Optional[pathlib.Path] = None


@dataclass
class SyntheticConfig(TaskConfig):
    task_name: str = "dtlz1"
    domain: str = "synthetic"
    normalize_xs: bool = True
    normalize_ys: bool = True
    gin_config_files: List[str] = field(
        default_factory=lambda: ["./config/synthetic.gin"]
    )
    gin_params: List[str] = field(default_factory=list)


@dataclass
class REConfig(TaskConfig):
    task_name: str = "re21"
    domain: str = "re"
    normalize_xs: bool = True
    normalize_ys: bool = True
    gin_config_files: List[str] = field(default_factory=lambda: ["./config/re.gin"])


@dataclass
class MORLConfig(TaskConfig):
    task_name = "mo_hopper_v2"
    domain: str = "morl"
    normalize_xs: bool = True
    normalize_ys: bool = True
    gin_config_files: List[str] = field(
        default_factory=lambda: ["./config/morl_v2.gin"]
    )
    gin_params: List[str] = field(default_factory=list)


@dataclass
class ScientificConfig(TaskConfig):
    task_name: str = "rfp"
    domain: str = "scientific"
    normalize_xs: bool = False
    normalize_ys: bool = True
    gin_config_files: List[str] = field(
        default_factory=lambda: ["./config/scientific.gin"]
    )
    gin_params: List[str] = field(default_factory=list)

    def __post_init__(self):
        # Override normalize_xs if the task is "molecule"
        if self.task_name.lower() == "molecule":
            self.normalize_xs = True


def get_task_config(domain: str):
    domain_to_config = {
        "synthetic": SyntheticConfig,
        "re": REConfig,
        "scientific": ScientificConfig,
        "morl": MORLConfig,
    }
    if domain not in domain_to_config:
        raise ValueError(
            f"Unknown domain: {domain}. "
            f"Available domains: {list(domain_to_config.keys())}"
        )
    return domain_to_config[domain]


def parse_args() -> TaskConfig:
    parser = argparse.ArgumentParser(description="Diffusion Model Configs")

    parser.add_argument(
        "--seed", type=int, default=1000, help="Random seed (default: %(default)s)"
    )
    parser.add_argument(
        "--task_name", type=str, default="dtlz1", help="Subtask name (e.g., dtlz1, rfp)"
    )
    parser.add_argument(
        "--domain",
        type=str,
        required=True,
        choices=["synthetic", "scientific", "morl", "re"],
        help="Task domain (eg. synthetic, scientific)",
    )
    parser.add_argument(
        "--reweight-loss",
        action="store_true",
        help="Enable loss reweighting based on dominance number",
    )
    parser.add_argument(
        "--data_pruning", action="store_true", help="Enable pruning of dominated data"
    )
    parser.add_argument(
        "--data_preserved_ratio",
        type=float,
        default=0.2,
        help=("Fraction of data to preserve when pruning (default: %(default)s)"),
    )
    parser.add_argument(
        "--sampling-method",
        type=str,
        choices=["uniform-ideal", "uniform-direction"],
        default="uniform-ideal",
    )
    parser.add_argument(
        "--sampling-guidance-scale", type=float, nargs="+", default=[1.0]
    )

    parser.add_argument(
        "--use_wandb", action="store_true", help="Enables logging to Weights and biases"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help='The name of the experiment. Used only if "--use_wandb" is set',
    )
    parser.add_argument("--save_dir", type=pathlib.Path, default=None)

    args = parser.parse_args()
    ConfigClass = get_task_config(args.domain)


    exp_name = (
        "experiment"
        if args.experiment_name is None
        else args.experiment_name
    )

    if args.save_dir is not None:
        save_dir = (
            args.save_dir / args.domain / args.task_name / exp_name / str(args.seed)
        )
    else:
        save_dir = args.save_dir  # args.save_dir is none!

    # Ensure that the directory exists
    save_dir.mkdir(parents=True, exist_ok=True)


    config = ConfigClass(
        seed=args.seed,
        task_name=args.task_name,
        sampling_method=args.sampling_method,
        guidance_scale=args.sampling_guidance_scale,
        reweight_loss=args.reweight_loss,
        data_pruning=args.data_pruning,
        data_preserved_ratio=args.data_preserved_ratio,
        use_wandb=args.use_wandb,
        experiment_name=args.experiment_name,
        save_dir=save_dir,
    )

    return config


def get_slurm_job_id():
    """Retrieve job-id from slurm if applicable"""
    job_id = os.environ.get("SLURM_ARRAY_JOB_ID", None)
    if job_id is None:
        job_id = os.environ.get("SLURM_JOB_ID", None)
    return int(job_id) if job_id is not None else job_id


def get_slurm_task_id():
    """Retrieve task array id from slurm if applicable"""
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", None)
    return int(task_id) if task_id is not None else task_id


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    global now_seed
    now_seed = seed


def get_pareto_front(y):
    """
    Estimates the Pareto front using NonDominatedSorting from pymoo.

    Args:
        y (np.ndarray): Array of shape (n_samples, n_objectives) with
                        objective values for multi-objective optimization.

    Returns:
        tuple: (pareto_points, pareto_indices)
            - pareto_points (np.ndarray): Non-dominated objective vectors.
            - pareto_indices (np.ndarray): Idx of the non-dominated points in y
    """
    nds = (
        NonDominatedSorting()
    )  # Use "efficient_non_dominated_sort" for larger datasets
    front_indices = nds.do(y, only_non_dominated_front=True)
    return y[front_indices], front_indices


@gin.configurable(denylist=["scores", "maximize"])
def reweight_multi_objective(
    scores: np.ndarray,
    num_bins: int = 20,
    k: float = 10,
    tau: float = 0.05,
    maximize: bool = False,
    normalize_dom_counts: bool = True,
) -> np.ndarray:
    """
    Compute sample weights for multi-objective optimization problems using
    dominance rank and bin frequency to encourage Pareto-optimality & coverage.

    Parameters
    ----------
    scores : np.ndarray
        A (N, D) array of objective values where N is the number of samples
        and D is the number of objectives.
    bins : int
        Number of bins for each dimension in histogramdd.
    k : float
        Small constant controlling the penalty for overrepresented bins.
    tau : float
        Temperature parameter controlling how strongly the dominance depth
        affects weights.
    maximize : bool
        If True, assumes maximization; otherwise, minimization.
    normalize_dom_counts: bool
        If True, min-max normalize dominance numbers (
                easier to setup task independent hyperparams
        )
    Returns
    -------
    weights : np.ndarray
        A (N,) array of sample weights.
    """
    assert scores.ndim == 2, f"Expected 2D array for scores, got shape {scores.shape}"
    scores_proc = scores.copy()

    # Pygmo assumes minimization; invert if maximizing
    if maximize:
        scores_proc = -scores_proc

    _, _, dc, _ = fast_non_dominated_sorting(points=scores_proc)
    if normalize_dom_counts:
        dc = (dc - dc.min()) / (dc.max() - dc.min())

    hist, _, binnum = scipy.stats.binned_statistic_dd(
        scores_proc,
        values=None,
        statistic="count",
        bins=num_bins,
        expand_binnumbers=False,
    )
    weights = np.zeros(scores_proc.shape[0])
    unique_bins = np.unique(binnum)
    for i in range(unique_bins.shape[0]):
        mask = binnum == unique_bins[i]
        n_items = mask.sum()
        weights[mask] = n_items / (n_items + k) * np.exp(-dc[mask].mean() / tau)

    return weights


def sample_uniform_direction(
    d_best: np.ndarray, k: int, alpha: float = 0.4, noise_scale: float = 0.05
):
    idx = np.random.choice(len(d_best), size=k, replace=True)
    base = d_best[idx]

    # TODO: Make the total  number of points configurable
    # Tile the same points for easier math
    num_tiles = 256 // k
    base = np.tile(base, (num_tiles, 1))

    # Sample projection angles from 0 to 180 deg
    angles = np.random.uniform(size=base.shape[0]) * 0.5 * np.pi
    # Extrapolation factor
    x_i = base[:, 0] - np.sin(angles) * alpha
    y_i = base[:, 1] - np.cos(angles) * alpha
    return np.stack([x_i, y_i], axis=1)


def sample_uniform_toward_ideal(
    d_best: np.ndarray,
    k: int,
    alpha_range: Tuple[float, float] = (0.1, 0.4),
    noise_scale: float = 0.05,
) -> np.ndarray:
    """
    Uniformly interpolates between d_best points and pareto ideal
    then adds noise for exploration.

    The ideal is computed as the min of d_best.
    The nadir is computed as the max of d_best.

    Parameters:
        d_best (np.ndarray): Array of current best solutions (e.g., PF).
                             Shape (num_points, num_objectives).

        k (int): Number of conditioning points to generate.

        alpha_range (tuple[float, float]): Tuple specifying (min, max)
                                           for interpolation scalar.

        noise_scale (float): Standard deviation of the Gaussian noise to add.
                             This is a key hyperparameter for exploration.
    Returns:
        np.ndarray: Noisy conditioning points,
                    clipped within the ideal and nadir bounds.
    """
    ideal_point = d_best.min(axis=0, keepdims=True)
    nadir_point = d_best.max(axis=0, keepdims=True)

    idx = np.random.choice(len(d_best), size=k, replace=True)
    base = d_best[idx]

    min_alpha, max_alpha = alpha_range
    alphas = np.random.uniform(min_alpha, max_alpha, size=(k, 1))

    directions = ideal_point - base  # minimization direction
    cond_points = base + alphas * directions

    noise = np.random.normal(loc=0.0, scale=noise_scale, size=cond_points.shape)
    noisy_points = cond_points + noise

    return np.clip(noisy_points, a_min=ideal_point, a_max=nadir_point)


def sample_along_reference_vectors(
    d_best: np.ndarray,
    k: int,
    seed: int = 42,
    method: str = "energy",
    alpha_range: Tuple[float, float] = (0.1, 0.3),
    bounds: Tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """
    Sample k points by moving inward from a normalized Pareto front
    along reference directions.

    Args:
        d_best (np.ndarray): Normalized Pareto front, shape (N, M).
        k (int): Number of points to sample.
        seed (int): Random seed.
        method (str): Reference direction generation method.
        alpha_range (tuple): Range for step sizes toward the ideal point.
        bounds (tuple): Clipping bounds for sampled points.

    Returns:
        np.ndarray: Sampled points of shape (k, M).
    """
    np.random.seed(seed)
    n_obj = d_best.shape[1]

    # Generate reference directions (assumed normalized)
    ref_dirs = get_reference_directions(method, n_obj, k, seed=seed)

    # Compute crowding distances and sanitize them
    crowding_dist = calc_crowding_distance(d_best)
    finite_max = np.nanmax(crowding_dist[np.isfinite(crowding_dist)])
    crowding_dist = np.where(np.isinf(crowding_dist), finite_max * 10, crowding_dist)
    crowding_dist = np.nan_to_num(crowding_dist, nan=0.0)

    # Normalize to get sampling probabilities or fallback to uniform
    total_dist = np.sum(crowding_dist)
    prob = crowding_dist / total_dist if total_dist > 0 else None

    # Sample base points from d_best weighted by crowding distance
    idx = np.random.choice(len(d_best), size=k, replace=True, p=prob)
    base_points = d_best[idx]

    # Sample step sizes and move inward along reference directions
    alphas = np.random.uniform(*alpha_range, size=(k, 1))
    sampled_points = base_points - alphas * ref_dirs

    # Clip points within bounds
    return np.clip(sampled_points, bounds[0], bounds[1])
