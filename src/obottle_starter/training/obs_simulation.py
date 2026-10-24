# SPDX-License-Identifier: Apache-2.0
"""Synthetic observations for training conditional models.

A conditional model only learns the kinds of observations it is trained
on, so this is where its observing system is decided. Each training sample
gets fresh masks and noise for every channel in ``conditioning.observe``,
drawn on the GPU from the ``conditioning.training_observations`` settings::

    training_observations:
      masks: bernoulli               # bernoulli | stripes
      coverage: {SSH: 0.1, SST: 0.3} # bernoulli: fraction observed
      stripe_width: {SSH: 30, SST: 60}   # stripes: width in grid cells
      stripes_per_frame: {SSH: 1, SST: 2}
      frame_probability: 1.0         # chance a channel is observed at all in a frame
      noise_std: {SSH: 0.005, SST: 0.1}  # physical units (m, degC)
      null_probability: 0.0          # chance the whole condition is dropped (CFG / M_4)

``stripes`` are straight bands at a random angle and offset, a crude stand-in
for satellite swaths. Neither option reproduces real SWOT, nadir or cloud
coverage; replace :func:`simulate_masks` to train on real footprints.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from obottle_starter.core.normalization import Normalizer
from obottle_starter.core.obs_conditioning import (
    build_obs_condition,
    observed_names,
)

MASK_KINDS = ("bernoulli", "stripes")


def _per_channel(value, name: str, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return float(value.get(name, default))


def simulate_masks(settings, names, B: int, T: int, H: int, W: int, device) -> Tensor:
    """Random masks ``(B, len(names), T, H*W)``, one per sample, channel and frame."""
    kind = settings.get("masks", "bernoulli")
    if kind not in MASK_KINDS:
        raise ValueError(f"training_observations.masks must be one of {MASK_KINDS}")
    out = []
    for name in names:
        if kind == "bernoulli":
            p = _per_channel(settings.get("coverage"), name, 0.1)
            m = torch.rand(B, T, H, W, device=device) < p
        else:
            width = _per_channel(settings.get("stripe_width"), name, 30.0)
            count = int(_per_channel(settings.get("stripes_per_frame"), name, 1))
            rows = torch.arange(H, device=device, dtype=torch.float32)[:, None]
            cols = torch.arange(W, device=device, dtype=torch.float32)[None, :]
            m = torch.zeros(B, T, H, W, dtype=torch.bool, device=device)
            for _ in range(count):
                angle = torch.rand(B, T, 1, 1, device=device) * math.pi
                # Distance of each cell from a line through a random point.
                centre_r = torch.rand(B, T, 1, 1, device=device) * H
                centre_c = torch.rand(B, T, 1, 1, device=device) * W
                dist = (rows - centre_r) * torch.cos(angle) - (
                    cols - centre_c
                ) * torch.sin(angle)
                m |= dist.abs() < width / 2
        p_frame = _per_channel(settings.get("frame_probability"), name, 1.0)
        if p_frame < 1.0:
            m &= torch.rand(B, T, 1, 1, device=device) < p_frame
        out.append(m.reshape(B, T, H * W))
    return torch.stack(out, dim=1)


def training_condition(cfg, target: Tensor, normalizer: Normalizer) -> Tensor:
    """Observation condition for a training batch ``target`` ``(B, C, T, H*W)``.

    Draws masks and noise, builds the condition, and replaces it with the
    null condition for a ``null_probability`` fraction of samples.
    """
    settings = dict((cfg.get("conditioning") or {}).get("training_observations") or {})
    names = observed_names(cfg)
    B, _, T, _ = target.shape
    masks = simulate_masks(settings, names, B, T, cfg.H, cfg.W, target.device)
    values = target.clone()
    for name in names:
        c = normalizer.channels.index(name)
        std = normalizer.sigma_to_model(
            c, _per_channel(settings.get("noise_std"), name, 0.0)
        )
        if std > 0:
            values[:, c] += std * torch.randn_like(values[:, c])
    condition = build_obs_condition(cfg, values, masks)
    p_null = float(settings.get("null_probability", 0.0))
    if p_null > 0:
        keep = torch.rand(B, 1, 1, 1, device=target.device) >= p_null
        condition = condition * keep.to(condition.dtype)
    return condition
