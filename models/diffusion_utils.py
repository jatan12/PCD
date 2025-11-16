import pathlib
from typing import List, Optional, Union

import gin
import torch
import torch.nn as nn

from models.denoiser_network import ResidualMLPDenoiser
from models.elucidated_diffusion import ElucidatedDiffusion
from models.normalization import normalizer_factory


@gin.configurable
def construct_diffusion_model(
    inputs: torch.Tensor,
    normalizer_type: str,
    denoising_network: nn.Module,
    disable_terminal_norm: bool = False,
    skip_dims: Optional[List[int]] = None,
    cond_dim: Optional[int] = None,
) -> ElucidatedDiffusion:
    """
    Create and return a configured diffusion model with specified
    denoiser and normalizer.
    """
    if skip_dims is None:
        skip_dims = []

    event_dim = inputs.shape[1]

    model = denoising_network(d_in=event_dim, cond_dim=cond_dim)

    if disable_terminal_norm:
        terminal_dim = event_dim - 1
        if terminal_dim not in skip_dims:
            skip_dims.append(terminal_dim)

    if skip_dims:
        print(f"Skipping normalization for dimensions {skip_dims}.")

    normalizer = normalizer_factory(
        normalizer_type, inputs, skip_dims=skip_dims
    )

    return ElucidatedDiffusion(
        net=model,
        normalizer=normalizer,
        event_shape=[event_dim],
    )


def load_diffusion_model(
    filepath: Union[str, pathlib.Path],
    inputs: torch.Tensor,
    normalizer_type: str,
    disable_terminal_norm: bool = False,
    skip_dims: Optional[List[int]] = None,
    cond_dim: Optional[int] = None,
) -> ElucidatedDiffusion:
    """Load a diffusion model from checkpoint and initialize it."""
    if skip_dims is None:
        skip_dims = []

    filepath = pathlib.Path(filepath)
    if not filepath.is_file():
        raise FileNotFoundError(
            f"{str(filepath)!r} does not point to a model file!"
        )

    model = construct_diffusion_model(
        inputs=inputs,
        cond_dim=cond_dim,
    )

    state = torch.load(filepath, weights_only=True)
    print(state.keys())
    model.load_state_dict(state["model"])
    model.eval()
    return model
