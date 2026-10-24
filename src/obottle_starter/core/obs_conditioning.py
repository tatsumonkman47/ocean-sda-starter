# SPDX-License-Identifier: Apache-2.0
"""Observations as network inputs (conditional models, M_3 / M_4).

A *conditional* model takes the observations as extra input channels and
learns ``p(x | y)`` directly (Suresh Babu, Sadam & Lermusiaux, 2026: their
M_3; with condition dropout and classifier-free guidance, their M_4). It is
set up by the ``conditioning`` section of the training config::

    conditioning:
      mode: observations      # none (default): an unconditional prior
      observe: [SSH, SST]     # channels whose observations are inputs
      mask_channel: true      # also input a 0/1 "observed here" map per channel

For each observed channel the network gets the observed values in model
units, zero where unobserved, and, with ``mask_channel``, the mask itself.
Without a mask channel, "unobserved" and "observed value 0" look the same,
which in model units is common. The order of the condition channels is

    [values of observe[0], ..., values of observe[-1],
     mask of observe[0], ..., mask of observe[-1]]          (masks optional)

followed by the frame-time encodings (``temporal_offset_channels``). An
all-zero block (no values, no masks) is the *null* condition: "nothing
observed", which is what CFG training substitutes.

Nothing here depends on training or inference; both build their conditions
with :func:`build_obs_condition`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor

from obottle_starter.core.config import channel_names

MODES = ("none", "observations")


def settings(cfg) -> dict:
    """The ``conditioning`` section of a config, with defaults filled in."""
    section = cfg.get("conditioning") or {}
    mode = section.get("mode", "none") or "none"
    if mode not in MODES:
        raise ValueError(f"conditioning.mode must be one of {MODES}; got {mode!r}")
    names = channel_names(cfg)
    observe = list(section.get("observe") or names) if mode != "none" else []
    unknown = set(observe) - set(names)
    if unknown:
        raise ValueError(f"conditioning.observe has unknown channels {unknown}")
    return {
        "mode": mode,
        "observe": [c for c in names if c in observe],  # model channel order
        "mask_channel": bool(section.get("mask_channel", True)),
    }


def is_conditional(cfg) -> bool:
    """Whether the model takes observations as inputs."""
    return settings(cfg)["mode"] != "none"


def observation_channels(cfg) -> int:
    """Number of input channels the observations occupy (0 if unconditional)."""
    s = settings(cfg)
    return len(s["observe"]) * (2 if s["mask_channel"] else 1)


def build_obs_condition(
    cfg,
    values: Tensor | None,
    masks: Mapping[str, Tensor] | Tensor | None,
) -> Tensor | None:
    """The observation part of the condition.

    Args:
        cfg: Model config.
        values: ``(..., C, T, H*W)`` fields in model units containing the
            observed values, with ``C`` the model's channels; ``None`` for
            the null condition. Unobserved entries may hold anything,
            including NaN: they are replaced by 0. Observed entries must be
            finite.
        masks: ``{channel: (..., T, H*W) bool}``, or a ``(..., C_obs, T, H*W)``
            bool tensor in ``observe`` order. Channels missing from a mapping
            are unobserved.

    Returns:
        ``(..., observation_channels(cfg), T, H*W)``, or ``None`` for an
        unconditional model.
    """
    s = settings(cfg)
    if s["mode"] == "none":
        return None
    names = channel_names(cfg)
    if values is None:
        return None
    lead = values.shape[:-3]
    T, HW = values.shape[-2:]
    if isinstance(masks, Tensor):
        stack = masks.to(torch.bool)
    else:
        masks = masks or {}
        stack = torch.stack(
            [
                torch.as_tensor(
                    masks[c], dtype=torch.bool, device=values.device
                ).reshape(*lead, T, HW)
                if c in masks
                else torch.zeros(*lead, T, HW, dtype=torch.bool, device=values.device)
                for c in s["observe"]
            ],
            dim=-3,
        )
    idx = [names.index(c) for c in s["observe"]]
    observed = values[..., idx, :, :]
    if bool((stack & ~torch.isfinite(observed)).any()):
        raise ValueError(
            "observed values must be finite; found NaN or inf where the mask "
            "says observed"
        )
    # torch.where, not values * mask: NaN * 0 is still NaN.
    parts = [torch.where(stack, observed, torch.zeros_like(observed))]
    if s["mask_channel"]:
        parts.append(stack.to(values.dtype))
    return torch.cat(parts, dim=-3)


def null_obs_condition(cfg, like: Tensor) -> Tensor | None:
    """The all-zero ("nothing observed") condition shaped like ``like``'s batch.

    ``like`` is ``(..., C, T, H*W)``.
    """
    n = observation_channels(cfg)
    if n == 0:
        return None
    return torch.zeros(
        *like.shape[:-3], n, *like.shape[-2:], dtype=like.dtype, device=like.device
    )


def prepend(obs_condition: Tensor | None, condition: Tensor | None) -> Tensor | None:
    """Join the observation condition and the time encodings, in that order."""
    if obs_condition is None:
        return condition
    if condition is None:
        return obs_condition
    return torch.cat([obs_condition, condition.to(obs_condition.dtype)], dim=-3)


def observed_names(cfg) -> Sequence[str]:
    """Channels the model can take observations of, in model order."""
    return settings(cfg)["observe"]
