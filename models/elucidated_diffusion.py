"""
Main diffusion code.
Code was adapted from https://github.com/lucidrains/denoising-diffusion-pytorch
"""
import math
import multiprocessing as mp
import os
import pathlib
from typing import Literal, Optional, Sequence, Tuple, Union

import gin
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from einops import reduce
from torch.utils.data import DataLoader
from torchdiffeq import odeint
from tqdm import tqdm

from ema_pytorch import EMA
from models.normalization import BaseNormalizer


def get_cpu_count() -> int:
    """
    Slurm aware CPU count that recognizes how many CPU cores is allocated
    to the current node
    """
    allocated_cores = os.environ.get("SLURM_CPUS_PER_TASK", None)
    return mp.cpu_count() if allocated_cores is None else int(allocated_cores)


# Returns True if the value is not None
def exists(val):
    return val is not None


# Returns the provided value if it exists, otherwise returns the default value
def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


# Cycles through a DataLoader indefinitely
def cycle(dl):
    while True:
        for data in dl:
            yield data


# Computes the logarithm of a tensor with numerical stability
def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


# Main Class
@gin.configurable
class ElucidatedDiffusion(nn.Module):
    def __init__(
        self,
        net: nn.Module,
        normalizer: BaseNormalizer,
        event_shape: Sequence[int],  # shape of the input and output
        cond_drop_prob: float = 0.15,  # probability of dropping the condition
        num_sample_steps: int = 32,  # number of sampling steps
        sigma_min: float = 0.002,  # min noise level
        sigma_max: float = 80,  # max noise level
        sigma_data: float = 1.0,  # standard deviation of data distribution
        rho: float = 7,  # controls the sampling schedule
        P_mean: float = -1.2,  # mean of log-normal distribution from which noise is drawn for training
        P_std: float = 1.2,  # standard deviation of log-normal distribution from which noise is drawn for training
        S_churn: float = 80,  # parameters for stochastic sampling - depends on dataset, Table 5 in paper
        S_tmin: float = 0.05,
        S_tmax: float = 50,
        S_noise: float = 1.003,
    ):
        super().__init__()
        assert net.random_or_learned_sinusoidal_cond

        self.net = net
        self.normalizer = normalizer

        # input dimensions
        self.event_shape = event_shape

        # CFG parameters
        self.cond_drop_prob = cond_drop_prob

        # Sampling parameters
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.P_mean = P_mean
        self.P_std = P_std
        self.num_sample_steps = (
            num_sample_steps  # otherwise known as N in the paper
        )
        self.S_churn = S_churn
        self.S_tmin = S_tmin
        self.S_tmax = S_tmax
        self.S_noise = S_noise

    @property
    def device(self):
        return next(self.net.parameters()).device

    # Derived preconditioning params - Table 1
    def c_skip(self, sigma: Union[torch.Tensor, float]) -> torch.Tensor:
        return (self.sigma_data**2) / (sigma**2 + self.sigma_data**2)

    def c_out(self, sigma: Union[torch.Tensor, float]) -> torch.Tensor:
        return sigma * self.sigma_data * (self.sigma_data**2 + sigma**2) ** -0.5

    def c_in(self, sigma: Union[torch.Tensor, float]) -> float:
        return 1 * (sigma**2 + self.sigma_data**2) ** -0.5

    def c_noise(self, sigma: Union[torch.Tensor, float]) -> float:
        return log(sigma) * 0.25

    def _denoise(
        self, x: torch.Tensor, sigma: torch.Tensor, net_out: torch.Tensor
    ) -> torch.Tensor:
        return self.c_skip(sigma) * x + self.c_out(sigma) * net_out

    # Preconditioned network output, equation (7) in the paper
    def preconditioned_network_forward(
        self,
        noised_inputs: torch.Tensor,
        sigma: Union[torch.Tensor, float],
        guidance_scale: float,
        clamp: bool = False,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, device = noised_inputs.shape[0], noised_inputs.device

        if isinstance(sigma, float):
            sigma = torch.full((batch,), sigma, device=device)

        padded_sigma = sigma.view(batch, *([1] * len(self.event_shape)))

        # Classifier-free Guidance
        if guidance_scale != 1.0 and cond is not None:
            # Unconditional model: D_θ(x;σ)
            uncond_net_out = self.net(
                self.c_in(padded_sigma) * noised_inputs,
                self.c_noise(sigma),
                cond=None,
            )

            # Conditional model: D_θ(x|y;σ)
            cond_net_out = self.net(
                self.c_in(padded_sigma) * noised_inputs,
                self.c_noise(sigma),
                cond=cond,
            )

            # CFG: dxdσ = -((γ*D_θ(x|y;σ) + (1-γ)*D_θ(x;σ) - x)/σ)
            net_out = (
                guidance_scale * cond_net_out
                + (1.0 - guidance_scale) * uncond_net_out
            )

            out = self._denoise(noised_inputs, padded_sigma, net_out)
        else:
            # Standard denoising (no guidance)
            net_out = self.net(
                self.c_in(padded_sigma) * noised_inputs,
                self.c_noise(sigma),
                cond=cond,
            )

            out = self._denoise(noised_inputs, padded_sigma, net_out)

        if clamp:
            out = out.clamp(-1.0, 1.0)

        return out

    # Sample schedule, equation (5) in the paper
    def sample_schedule(
        self, num_sample_steps: Optional[int] = None
    ) -> torch.Tensor:
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)

        N = num_sample_steps
        inv_rho = 1 / self.rho

        steps = torch.arange(
            num_sample_steps, device=self.device, dtype=torch.float32
        )
        sigmas = (
            self.sigma_max**inv_rho
            + steps
            / (N - 1)
            * (self.sigma_min**inv_rho - self.sigma_max**inv_rho)
        ) ** self.rho

        sigmas = F.pad(
            sigmas, (0, 1), value=0.0
        )  # last step is sigma value of 0.

        return sigmas

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 16,
        guidance_scale: float = 1.0,
        num_sample_steps: Optional[int] = None,
        clamp: bool = False,
        cond: Optional[torch.Tensor] = None,
        disable_tqdm: bool = False,
    ) -> torch.Tensor:
        self.eval()  # ensure the model is in eval mode
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)
        shape = (batch_size, *self.event_shape)

        if cond is not None:
            cond = cond.to(self.device)

        # get the schedule, which is returned as (sigma, gamma) tuple, and pair up with the next sigma and gamma
        sigmas = self.sample_schedule(num_sample_steps)
        gammas = torch.where(
            (sigmas >= self.S_tmin) & (sigmas <= self.S_tmax),
            min(self.S_churn / num_sample_steps, math.sqrt(2) - 1),
            0.0,
        )

        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[:-1]))

        # inputs are noise at the beginning
        init_sigma = sigmas[0]
        inputs = init_sigma * torch.randn(shape, device=self.device)

        # gradually denoise
        for sigma, sigma_next, gamma in tqdm(
            sigmas_and_gammas,
            desc="sampling time step",
            mininterval=1,
            disable=disable_tqdm,
        ):
            sigma, sigma_next, gamma = map(
                lambda t: t.item(), (sigma, sigma_next, gamma)
            )

            eps = self.S_noise * torch.randn(
                shape, device=self.device
            )  # stochastic sampling

            sigma_hat = sigma + gamma * sigma
            inputs_hat = inputs + math.sqrt(sigma_hat**2 - sigma**2) * eps

            # Pass the guidance_scale to the score function
            denoised_over_sigma = self.score_fn(
                inputs_hat, sigma_hat, clamp=clamp,
                cond=cond, guidance_scale=guidance_scale
            )
            inputs_next = (
                inputs_hat + (sigma_next - sigma_hat) * denoised_over_sigma
            )

            # second order correction, if not the last timestep
            if sigma_next != 0:
                denoised_prime_over_sigma = self.score_fn(
                    inputs_next, sigma_next, clamp=clamp,
                    cond=cond, guidance_scale=guidance_scale
                )
                inputs_next = inputs_hat + 0.5 * (sigma_next - sigma_hat) * (
                    denoised_over_sigma + denoised_prime_over_sigma
                )

            inputs = inputs_next

        if clamp:
            inputs = inputs.clamp(-1.0, 1.0)

        return self.normalizer.unnormalize(inputs)

    # This is known as 'denoised_over_sigma' in the lucidrains repo.
    def score_fn(
        self,
        x: torch.Tensor,
        sigma: Union[torch.Tensor, float],
        guidance_scale: float,
        clamp: bool = False,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        denoised = self.preconditioned_network_forward(
            x, sigma, clamp=clamp, cond=cond, guidance_scale=guidance_scale
        )
        denoised_over_sigma = (x - denoised) / sigma
        return denoised_over_sigma

    # Adapted from https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/sampling.py
    @torch.no_grad()
    def log_likelihood(
        self,
        x: torch.Tensor,
        atol: float = 1e-4,
        rtol: float = 1e-4,
        clamp: bool = False,
        normalize: bool = True,
        cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        # Input to the ODE solver must be in normalized space.
        if normalize:
            x = self.normalizer.normalize(x)

        v = torch.randint_like(x, 2) * 2 - 1
        s_in = x.new_ones([x.shape[0]])
        fevals = 0

        def ode_fn(sigma, x):
            nonlocal fevals
            with torch.enable_grad():
                x = x[0].detach().requires_grad_()
                sigma = sigma * s_in
                padded_sigma = sigma.view(
                    x.shape[0], *([1] * len(self.event_shape))
                )
                denoised = self.preconditioned_network_forward(
                    x, sigma, clamp=clamp, cond=cond
                )
                denoised_over_sigma = (x - denoised) / padded_sigma
                fevals += 1
                grad = torch.autograd.grad((denoised_over_sigma * v).sum(), x)[
                    0
                ]
                d_ll = (v * grad).flatten(1).sum(1)
            return denoised_over_sigma.detach(), d_ll

        x_min = x, x.new_zeros([x.shape[0]])
        t = x.new_tensor([self.sigma_min, self.sigma_max])
        sol = odeint(ode_fn, x_min, t, atol=atol, rtol=rtol, method="dopri5")
        latent, delta_ll = sol[0][-1], sol[1][-1]
        ll_prior = (
            torch.distributions.Normal(0, self.sigma_max)
            .log_prob(latent)
            .flatten(1)
            .sum(1)
        )

        return ll_prior + delta_ll, {"fevals": fevals}

    # training
    def loss_weight(self, sigma: Union[torch.Tensor, float]) -> torch.Tensor:
        return (sigma**2 + self.sigma_data**2) * (sigma * self.sigma_data) ** -2

    def noise_distribution(self, batch_size: int) -> torch.Tensor:
        return (
            self.P_mean
            + self.P_std * torch.randn((batch_size,), device=self.device)
        ).exp()

    # Forward pass for training
    def forward(
        self,
        inputs: torch.Tensor,
        weights: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.train()  # ensure the model is in training mode
        inputs = self.normalizer.normalize(inputs)

        batch_size, *event_shape = inputs.shape
        assert event_shape == self.event_shape, (
            f"mismatch of event shape, "
            f"expected {self.event_shape}, got {event_shape}"
        )

        sigmas = self.noise_distribution(batch_size)
        padded_sigmas = sigmas.view(batch_size, *([1] * len(self.event_shape)))

        noise = torch.randn_like(inputs)
        noised_inputs = (
            inputs + padded_sigmas * noise
        )  # alphas are 1. in the paper

        # Apply conditional dropout if using CFG
        if cond is not None and self.cond_drop_prob > 0:
            cond_mask = (
                torch.rand(batch_size, device=self.device)
                >= self.cond_drop_prob
            )
            cond_mask = cond_mask.float().unsqueeze(1)
            cond = cond * cond_mask  # works if cond shape is [batch_size, d]

        # Explicitly pass guidance_scale=1.0 during training
        denoised = self.preconditioned_network_forward(
            noised_inputs, sigmas, cond=cond, guidance_scale=1.0
        )
        losses = F.mse_loss(denoised, inputs, reduction="none")
        losses = reduce(losses, "b ... -> b", "mean")
        losses = losses * self.loss_weight(
            sigmas
        )  # Reweighted loss goes over here
        losses = losses * weights
        return losses.mean()


@gin.configurable(denylist=["results_folder"])
class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset: Optional[torch.utils.data.Dataset] = None,
        val_dataset: Optional[torch.utils.data.Dataset] = None,
        train_batch_size: int = 16,
        small_batch_size: int = 16,
        val_batch_size: int = 64,
        gradient_accumulate_every: int = 1,
        train_lr: float = 1e-4,
        lr_scheduler: Optional[str] = None,
        train_num_steps: int = 100000,
        ema_update_every: int = 10,
        ema_decay: float = 0.995,
        adam_betas: Tuple[float, float] = (0.9, 0.99),
        sample_every: int = 10000,
        weight_decay: float = 0.0,
        results_folder: str = "./results",
        amp: bool = False,
        fp16: bool = False,
        split_batches: bool = True,
        use_wandb: bool = False,
    ):
        super().__init__()
        
        self.use_wandb = use_wandb
        if self.use_wandb:
            wandb.config.update(
                {
                    "split_batches": split_batches,
                    "train_batch_size": train_batch_size,
                    "lr": train_lr,
                    "train_num_steps": train_num_steps,
                    "ema_update_every": ema_update_every,
                    "ema_decay": ema_decay,
                }
            )

        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision="fp16" if fp16 else "no",
        )
        self.accelerator.native_amp = amp
        self.model = diffusion_model

        num_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f"Number of trainable parameters: {num_params}.")

        self.sample_every = sample_every
        self.train_num_steps = train_num_steps
        self.gradient_accumulate_every = gradient_accumulate_every

        if dataset is not None:
            # If dataset size is less than 800K use the small batch size
            self.batch_size = (
                small_batch_size
                if len(dataset) < int(8e5)
                else train_batch_size
            )
            print(f"Using batch size: {self.batch_size}")
            # dataset and dataloader
            dl = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=get_cpu_count(),
            )
            dl = self.accelerator.prepare(dl)
            self.dl = cycle(dl)
        else:
            # No dataloader, train batch by batch
            self.batch_size = train_batch_size
            self.dl = None

        if val_dataset is not None:
            val_dl = DataLoader(
                val_dataset,
                batch_size=val_batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=get_cpu_count(),
            )
            val_dl = self.accelerator.prepare(val_dl)
            self.val_dl = val_dl
        else:
            self.val_dl = None

        # optimizer, make sure that the bias & layer-norm weights are not decayed
        no_decay = ["bias", "LayerNorm.weight", "norm.weight", ".g"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        self.opt = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=train_lr, betas=adam_betas
        )

        # for logging results in a folder periodically
        if self.accelerator.is_main_process:
            self.ema = EMA(
                diffusion_model, beta=ema_decay, update_every=ema_update_every
            )
            self.results_folder = pathlib.Path(results_folder)
            self.results_folder.mkdir(exist_ok=True, parents=True)

        # step counter state
        self.step = 0

        # prepare model, dataloader, optimizer with accelerator
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        if lr_scheduler == "linear":
            print("using linear learning rate scheduler")
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.opt, lambda step: max(0, 1 - step / train_num_steps)
            )
        elif lr_scheduler == "cosine":
            print("using cosine learning rate scheduler")
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.opt, train_num_steps
            )
        else:
            self.lr_scheduler = None

        self.model.normalizer.to(self.accelerator.device)
        self.ema.ema_model.normalizer.to(self.accelerator.device)

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            "step": self.step,
            "model": self.accelerator.get_state_dict(self.model),
            "opt": self.opt.state_dict(),
            "ema": self.ema.state_dict(),
            "scaler": self.accelerator.scaler.state_dict()
            if exists(self.accelerator.scaler)
            else None,
        }

        torch.save(data, str(self.results_folder / f"model-{milestone}.pt"))

    def load(self, milestone: int):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(
            str(self.results_folder / f"model-{milestone}.pt"),
            map_location=device,
        )

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data["model"])

        self.step = data["step"]
        self.opt.load_state_dict(data["opt"])
        self.ema.load_state_dict(data["ema"])

        if exists(self.accelerator.scaler) and exists(data["scaler"]):
            self.accelerator.scaler.load_state_dict(data["scaler"])

    @torch.no_grad()
    def validate(self):
        if self.val_dl is None:
            return None

        self.model.eval()
        total_val_loss = 0.0
        count = 0

        for batch in self.val_dl:
            x, y, w= (t.to(self.accelerator.device) for t in batch)
            loss = self.model(x, weights=w, cond=y)
            total_val_loss += loss.item()
            count += 1

        self.model.train()
        avg_val_loss = total_val_loss / count if count > 0 else None
        return avg_val_loss

    # Train for the full number of steps.
    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(
            initial=self.step,
            total=self.train_num_steps,
            disable=not accelerator.is_main_process,
        ) as pbar:
            while self.step < self.train_num_steps:
                total_loss = 0.0

                for _ in range(self.gradient_accumulate_every):
                    x, y, w = (
                        t.to(device) for t in next(self.dl)
                    )  # data = (next(self.dl)[0]).to(device)
                    with self.accelerator.autocast():
                        loss = self.model(x, weights=w, cond=y)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()
                    self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                pbar.set_description(f"loss: {total_loss:.4f}")
                if self.use_wandb:
                    wandb.log(
                        {
                            "step": self.step,
                            "loss": total_loss,
                            "lr": self.opt.param_groups[0]["lr"],
                        }
                    )

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.to(device)
                    self.ema.update()

                    if (
                        self.step != 0
                        and self.step % self.sample_every == 0
                    ):
                        #self.save(self.step)
                        val_loss = self.validate()
                        if val_loss is not None:
                            print(
                                f"Validation loss at step {self.step}: "
                                f"{val_loss:.4f}"
                            )
                            if self.use_wandb: 
                                wandb.log(
                                    {
                                        "step": self.step,
                                        "val_loss": val_loss,
                                    }
                                )
                pbar.update(1)

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

        accelerator.print("training complete")
        self.save("final") # <- Save only the final model

    # Allow user to pass in external data.
    def train_on_batch(
        self,
        data: Tuple[torch.Tensor, torch.Tensor],
        use_wandb=False,
        splits=1,  # number of splits to split the batch into
        **kwargs,
    ):
        assert len(data) == 2, "Input data should be a tuple of (x, y) tensors"
        accelerator = self.accelerator
        device = accelerator.device
        x, y = (t.to(device) for t in data)  # data = data.to(device)

        total_loss = 0.0

        if splits == 1:
            with self.accelerator.autocast():
                loss = self.model(x, cond=y, **kwargs)
                total_loss += loss.item()
            self.accelerator.backward(loss)
        else:
            assert splits > 1 and x.shape[0] % splits == 0

            split_x = torch.split(x, x.shape[0] // splits)
            split_y = torch.split(y, y.shape[0] // splits)

            for x_chunk, y_chunk in zip(split_x, split_y):
                with self.accelerator.autocast():
                    loss = self.model(x_chunk, cond=y_chunk, **kwargs)
                    loss = loss / splits
                    total_loss += loss.item()
                self.accelerator.backward(loss)

        accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
        if use_wandb:
            wandb.log(
                {
                    "step": self.step,
                    "loss": total_loss,
                    "lr": self.opt.param_groups[0]["lr"],
                }
            )

        accelerator.wait_for_everyone()

        self.opt.step()
        self.opt.zero_grad()

        accelerator.wait_for_everyone()

        self.step += 1
        if accelerator.is_main_process:
            self.ema.to(device)
            self.ema.update()

            if self.step != 0 and self.step % self.sample_every == 0:
                pass
                #self.save(self.step)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return total_loss
