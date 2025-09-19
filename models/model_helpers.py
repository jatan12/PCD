import os
import random
import pathlib
import argparse
from dataclasses import asdict
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import torch
import gin
import wandb

from pygmo import fast_non_dominated_sorting
from pymoo.factory import get_reference_directions
from pymoo.util.function_loader import load_function
from pymoo.algorithms.moo.nsga3 import calc_niche_count
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


@dataclass
class TaskConfig:
    seed: int = 42
    task_name: str = ""
    domain: str = ""
    reweight_loss: bool = False
    data_pruning: bool = False
    data_preserved_ratio: float = 0.2
    normalize_xs: bool = False
    normalize_ys: bool = False
    normalize_method_xs: str = "z-score"
    normalize_method_ys: str = "z-score"
    num_cond_points: int = 32  # 1, 2, 4, 8, 16, 32, 64, 128, 256
    sampling_noise_scale: float = 0.05
    # extrapolation_factor: float = 0.0
    num_pareto_solutions: int = 256
    guidance_scales: List[float] = field(default_factory=lambda: [1.0, 2.0, 2.5, 5.0, 8.0])
    use_val_split: bool = True
    val_ratio: float = 0.2
    use_wandb: bool = False
    experiment_name: Optional[str] = None
    save_dir: Optional[pathlib.Path] = None
    model_dir: Optional[pathlib.Path] = None
    gin_config_files: List[str] = field(default_factory=list)
    gin_params: List[str] = field(default_factory=list)


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
class MORLConfig(TaskConfig):
    task_name: str = "mo_hopper_v2"
    domain: str = "morl"
    normalize_xs: bool = True
    normalize_ys: bool = True
    gin_config_files: List[str] = field(
        default_factory=lambda: ["./config/morl.gin"]
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


@dataclass
class MONASConfig(TaskConfig):
    task_name: str = "c10mop1"
    domain: str = "monas"
    normalize_xs: bool = False
    normalize_ys: bool = True
    gin_config_files: List[str] = field(
        default_factory=lambda: ["./config/monas.gin"]
    )
    gin_params: List[str] = field(default_factory=list)


@dataclass
class REConfig(TaskConfig):
    task_name: str = "re21"
    domain: str = "re"
    normalize_xs: bool = True
    normalize_ys: bool = True
    gin_config_files: List[str] = field(
        default_factory=lambda: ["./config/re.gin"]
    )
    gin_params: List[str] = field(default_factory=list)


def get_task_config(domain: str):
    domain_to_config = {
        "synthetic": SyntheticConfig,
        "morl": MORLConfig,
        "scientific": ScientificConfig,
        "monas": MONASConfig,
        "re": REConfig,
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
        choices=["synthetic", "morl", "scientific", "monas", "re"],
        help="Task domain (eg. synthetic, scientific)",
    )
    parser.add_argument(
        "--reweight_loss",
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
    # parser.add_argument(
    #     "--extrapolation-factor",
    #     type=float,
    #     default=0.0,
    #     help="Factor for extrapolating conditioning points (default: %(default)s)",
    # )
    parser.add_argument(
        "--num_cond_points",
        type=int,
        default=32,
        help="Number of non-dominated solutions to use for conditioning (default: %(default)s)",
    )
    parser.add_argument(
        "--sampling_noise_scale",
        type=float,
        default=0.05,
        help="Sampling noise for conditional points (default: %(default)s)"
    )
    parser.add_argument(
        "--num_pareto_solutions",
        type=int,
        default=256,
        help="Number of Pareto solutions to generate during sampling (default: %(default)s)",
    )
    parser.add_argument(
        "--guidance_scales",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 2.5, 5.0, 8.0],
        help="Guidance scales for diffusion sampling",
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
    parser.add_argument("--model_dir", type=pathlib.Path, default=None)
    parser.add_argument("--gin_params", nargs="*", default=[])

    args = parser.parse_args()
    ConfigClass = get_task_config(args.domain)

    # Create the save directory
    exp_name = (
        "experiment"
        if args.experiment_name is None
        else args.experiment_name
    )

    # Determine the save directory path
    save_dir = None
    if args.save_dir is not None:
        save_dir = (
            args.save_dir / args.domain / args.task_name / exp_name / str(args.seed)
        )
        # Ensure that the directory exists
        save_dir.mkdir(parents=True, exist_ok=True)

    config = ConfigClass(
        seed=args.seed,
        task_name=args.task_name,
        reweight_loss=args.reweight_loss,
        data_pruning=args.data_pruning,
        data_preserved_ratio=args.data_preserved_ratio,
        num_cond_points=args.num_cond_points,
        sampling_noise_scale=args.sampling_noise_scale,
        # extrapolation_factor=args.extrapolation_factor,
        num_pareto_solutions=args.num_pareto_solutions,
        guidance_scales=args.guidance_scales,
        use_wandb=args.use_wandb,
        experiment_name=args.experiment_name,
        gin_params=args.gin_params,
        save_dir=save_dir,
        model_dir=args.model_dir,
    )

    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    global now_seed
    now_seed = seed


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


def extrapolate_cond_points(
    cond_points: np.ndarray,
    utopia_point: np.ndarray,
    extrapolation_factor: float = 0.1,
) -> np.ndarray:
    """
    Extrapolates a set of points towards a given utopia point.

    Args:
        cond_points (np.ndarray):
            The non-dominated points to be extrapolated (shape: [N, D]).
        utopia_point (np.ndarray):
            The ideal objective vector, e.g., np.min(all_y, axis=0).
            (shape: [D]).
        extrapolation_factor (float):
            The improvement factor (e.g., 0.1 for a 10% push towards the ideal).
            Defaults to 0.1.

    Returns:
        np.ndarray: The new, extrapolated target points (shape: [N, D]).
    """
    if extrapolation_factor == 0:
        return cond_points

    # Calculate the direction vector from each point to the utopia point
    direction_to_utopia = utopia_point - cond_points

    # Apply the extrapolation formula
    extrapolated_points = cond_points + extrapolation_factor * direction_to_utopia

    return extrapolated_points


def _niching(
    pop,
    n_remaining,
    niche_count,
    niche_of_individuals,
    dist_to_niche,
):
    # Very slightly modified procedure for niching from Pymoo
    survivors = []

    # boolean array of elements that are considered for each iteration
    mask = np.full(len(pop), True)

    while len(survivors) < n_remaining:
        # If all points from this set of points have been considered, stop
        if np.all(~mask):
            break
        # number of individuals to select in this iteration
        n_select = n_remaining - len(survivors)

        # all niches where new individuals can be assigned to and the
        # corresponding niche count
        next_niches_list = np.unique(niche_of_individuals[mask])
        next_niche_count = niche_count[next_niches_list]

        # the minimum niche count
        min_niche_count = next_niche_count.min()

        # all niches with the minimum niche count
        # (truncate randomly if there are more niches than remaining individuals)
        next_niches = next_niches_list[np.where(next_niche_count == min_niche_count)[0]]
        next_niches = next_niches[np.random.permutation(len(next_niches))[:n_select]]

        for next_niche in next_niches:
            # indices of individuals that are considered and assign to next_niche
            next_ind = np.where(
                np.logical_and(niche_of_individuals == next_niche, mask)
            )[0]

            # shuffle to break random tie (equal perp. dist) or select randomly
            np.random.shuffle(next_ind)

            if niche_count[next_niche] == 0:
                next_ind = next_ind[np.argmin(dist_to_niche[next_ind])]
            else:
                # already randomized through shuffling
                next_ind = next_ind[0]

            # add the selected individual to the survivors
            mask[next_ind] = False
            survivors.append(int(next_ind))

            # increase the corresponding niche count
            niche_count[next_niche] += 1

    return survivors


def sample_along_ref_dirs(
    d_best: np.ndarray,
    k: int,
    num_points: int,
    alpha_range: Tuple[float, float] = (0.1, 0.4),
    noise_scale: float = 0.05,
    seed: int = 42,
) -> np.ndarray:
    ref_dirs = get_reference_directions("energy", d_best.shape[1], k, seed=seed)
    d_best = d_best.astype(np.float64)
    fronts, rank = NonDominatedSorting().do(
        d_best, return_rank=True, n_stop_if_ranked=k
    )  # We need at most k values

    # Use non-dominated points as a starting point
    points = d_best[fronts[0]]
    point_indices = np.unique(
        fronts[0]
    )  # Keep count of the indices of the selected points

    # If there are more non-dominated points than k, select subset of these points
    if points.shape[0] > k:
        inds = np.random.choice(np.arange(points.shape[0]), size=k, replace=False)
        points = points[inds]
        point_indices = point_indices[inds]

    # Associate each point with a reference direction
    dist_mtx = load_function("calc_perpendicular_distance")(d_best, ref_dirs)
    niche_of_inds = np.argmin(dist_mtx, axis=1)
    dist_to_niche = dist_mtx[np.arange(d_best.shape[0]), niche_of_inds]

    # If there are less than k non-dominating points, choose points from other fronts
    # such that reference directions with minimal number of points are considerd
    front_index = 1
    while points.shape[0] < k:
        # If we have used all fronts, stop
        if front_index > len(fronts):
            # Gone through all fronts
            break

        current_front = fronts[front_index]
        if front_index == 1:
            prev_fronts = fronts[:front_index-1]
        else:
            prev_fronts = np.concatenate(fronts[:front_index-1])

        front_index += 1
        # We will always use all points from previous fronts
        # before moving to the next one, so one can just use indices of all previous fronts
        niche_count = calc_niche_count(len(ref_dirs), niche_of_inds[prev_fronts])
        num_remaining = k - points.shape[0]

        # assing points to the reference directions
        S = _niching(
            d_best[current_front],
            num_remaining,
            niche_count,
            niche_of_inds[current_front],
            dist_to_niche[current_front],
        )
        point_indices = np.concatenate((point_indices, current_front[S].tolist()))
        points = d_best[point_indices]

    assert len(points) == k

    # Tile the directions for each chosen point
    tiling_factor = num_points // k
    point_dirs = niche_of_inds[point_indices]

    directions = np.tile(ref_dirs[point_dirs], (tiling_factor, 1))
    # Interpolate each point towards the closest reference direction with
    # different length
    min_alpha, max_alpha = alpha_range
    alphas = np.random.uniform(min_alpha, max_alpha, size=num_points)
    points = np.tile(points, (tiling_factor, 1)) - np.einsum(
        "i,ij->ij", alphas, directions
    )

    # TODO: Add clipping
    return (
        points
        if abs(noise_scale) < 1e-10
        else np.random.normal(points, scale=noise_scale)
    )


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
