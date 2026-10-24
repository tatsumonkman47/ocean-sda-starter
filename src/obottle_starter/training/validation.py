# SPDX-License-Identifier: Apache-2.0
"""Held-out denoising loss, by channel and noise level.

The training loss mixes a random noise level per sample, so its value
jumps around and hides which noise levels or variables are learned well.
:func:`denoising_loss_by_sigma` instead evaluates a fixed set of held-out
samples at fixed noise levels with fixed noise, so passes at different
steps are directly comparable. It uses the same EDM weighting as training,
``λ(σ) = (σ² + σ_data²) / (σ σ_data)²``, so values near 1 at large σ mean
"no better than predicting the mean".
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from obottle_starter.core.obs_conditioning import is_conditional, prepend
from obottle_starter.training.obs_simulation import training_condition

NET_KWARGS = (
    "condition",
    "class_labels",
    "second_of_day",
    "day_of_year",
    "time_offsets",
)


@torch.no_grad()
def denoising_loss_by_sigma(
    net: torch.nn.Module,
    samples: Sequence[dict],
    sigmas: Sequence[float],
    cfg,
    *,
    normalizer=None,
    batch_size: int = 8,
    seed: int = 0,
) -> dict[float, list[float]]:
    """Mean weighted denoising error per channel at each noise level.

    Args:
        net: Network (normally the EMA copy), in eval mode.
        samples: Fixed samples from
            :func:`obottle_starter.training.data.validation_samples`.
        sigmas: Noise levels (model units).
        cfg: Training config.
        normalizer: Needed for conditional models, whose synthetic
            observations are drawn (with fixed randomness) as in training.
        batch_size: Samples per forward pass.
        seed: Seed for the noise, so every pass uses the same noise.

    Returns:
        ``{sigma: [loss of channel 0, loss of channel 1, ...]}``.
    """
    device = next(net.parameters()).device
    sigma_data = float(cfg.sigma_data)
    out = {}
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        for sigma in sigmas:
            torch.manual_seed(seed)
            total, count = None, 0
            for i in range(0, len(samples), batch_size):
                chunk = samples[i : i + batch_size]
                batch = {
                    k: torch.stack([s[k] for s in chunk]).to(device) for k in chunk[0]
                }
                kwargs = {k: batch[k] for k in NET_KWARGS if k in batch}
                x = batch["target"]
                if is_conditional(cfg):
                    obs = training_condition(cfg, x, normalizer)
                    kwargs["condition"] = prepend(obs, kwargs.get("condition"))
                s = torch.full((x.shape[0],), float(sigma), device=device)
                noisy = x + float(sigma) * torch.randn_like(x)
                denoised = net(noisy, s, force_fp32=True, **kwargs).out
                weight = (sigma**2 + sigma_data**2) / (sigma * sigma_data) ** 2
                err = weight * (denoised - x).square().mean(dim=(2, 3)).sum(0)
                total = err if total is None else total + err
                count += x.shape[0]
            out[float(sigma)] = (total / count).cpu().tolist()
    return out


def format_validation(step: int, losses: dict, names: Sequence[str]) -> str:
    """One line per noise level, e.g. ``sigma 0.1: SSH 0.21  SST 0.35``."""
    lines = [f"validation at step {step} (EMA weights, held-out samples):"]
    for sigma, values in losses.items():
        parts = "  ".join(f"{n} {v:.4g}" for n, v in zip(names, values))
        lines.append(f"  sigma {sigma:g}: {parts}")
    return "\n".join(lines)
