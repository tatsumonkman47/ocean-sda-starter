# SPDX-License-Identifier: Apache-2.0
"""Build, load and export ``EDMPrecond(SongUNet)`` ocean diffusion models.

A checkpoint is a ``torch.save`` dict with a ``config`` (a plain dict of the
training options, see ``configs/train_base.yaml``) and an ``ema_state_dict``.
Training checkpoints also carry ``model_state_dict`` and
``optimizer_state_dict``; :func:`export_checkpoint` drops those to make a
smaller inference-only file. The network is always rebuilt from ``config``,
so nothing but tensors and plain Python types is pickled.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

from obottle_starter.core.config import (
    ARCHITECTURE_KEYS,
    channel_names,
    num_channels,
    upgrade,
    validate_model,
    variables,
)
from obottle_starter.core.nn.domain import Plane
from obottle_starter.core.nn.networks import EDMPrecond, SongUNet

__all__ = [
    "build_model",
    "channel_names",
    "export_checkpoint",
    "init_differences",
    "load_checkpoint",
    "load_weights_padded",
]


def build_model(cfg, device: torch.device | str = "cpu") -> EDMPrecond:
    """Construct ``EDMPrecond(SongUNet)`` on a 2-D ``H x W`` grid.

    The network works on tensors of shape ``(B, C, T, H*W)``. Its input is
    the noisy state, then (for a conditional model) the observation channels
    (:mod:`obottle_starter.core.obs_conditioning`), then
    ``temporal_offset_channels`` sinusoidal encodings of each frame's time
    offset (:func:`obottle_starter.core.conditioning.frame_conditioning`).
    """
    # Imported here: obs_conditioning itself imports this module.
    from obottle_starter.core.obs_conditioning import observation_channels

    validate_model(cfg)
    if cfg.get("per_channel_offsets", False):
        raise NotImplementedError("per_channel_offsets is not supported")

    C = num_channels(cfg)
    n_obs = observation_channels(cfg)
    domain = Plane(nx=cfg.W, ny=cfg.H)
    label_dim = cfg.label_dim + cfg.get("span_label_channels", 0)
    temporal = cfg.time_length > 1
    # CalendarEmbedding needs a per-pixel longitude tensor at construction;
    # the real tile longitude enters through second_of_day (local solar time).
    calendar_lon = (
        torch.zeros(cfg.H * cfg.W) if cfg.calendar_embed_channels > 0 else None
    )
    unet = SongUNet(
        domain=domain,
        in_channels=C + n_obs + cfg.temporal_offset_channels,
        out_channels=C,
        label_dim=label_dim,
        label_dropout=cfg.label_dropout,
        model_channels=cfg.model_channels,
        channel_mult=list(cfg.channel_mult),
        num_blocks=cfg.num_blocks,
        attn_resolutions=list(cfg.attn_resolutions),
        dropout=cfg.dropout,
        time_length=cfg.time_length,
        mixing_type=cfg.get("mixing_type", "spatial"),
        calendar_embed_channels=cfg.calendar_embed_channels,
        calendar_lon=calendar_lon,
        temporal_attention_resolutions=(
            list(cfg.temporal_attn_resolutions) if temporal else None
        ),
        decoder_start_with_temporal_attention=(
            cfg.decoder_start_temporal_attn and temporal
        ),
        upsample_temporal_attention=cfg.upsample_temporal_attn and temporal,
        use_per_frame_adagn=cfg.use_per_frame_adagn,
        per_frame_adaptive_scale=cfg.get("per_frame_adaptive_scale", False),
    )
    net = EDMPrecond(
        model=unet,
        domain=domain,
        img_channels=C,
        time_length=cfg.time_length,
        label_dim=label_dim,
        condition_channels=n_obs + cfg.temporal_offset_channels,
        sigma_data=cfg.sigma_data,
    )
    return net.to(device)


def load_checkpoint(
    path: str | Path,
    device: torch.device | str = "cuda",
    *,
    weights: str = "ema_state_dict",
) -> tuple[EDMPrecond, DictConfig]:
    """Load a checkpoint as a frozen network in eval mode.

    Args:
        path: ``.pt`` file from training or :func:`export_checkpoint`.
        device: Where to put the network.
        weights: ``ema_state_dict`` (default) or ``model_state_dict``.

    Returns:
        ``(net, cfg)``. ``cfg`` is in the current format (older checkpoint
        configs are translated by :func:`obottle_starter.core.config.upgrade`).
        Gradients are disabled on the parameters (the network's input can
        still be differentiated, as SDA guidance needs) and fp16 is off.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "config" not in ckpt:
        raise ValueError(f"{path} has no 'config'; cannot rebuild the network")
    if weights not in ckpt:
        raise ValueError(f"{path} has no {weights!r}")
    cfg = upgrade(OmegaConf.create(ckpt["config"]))
    net = build_model(cfg, "cpu")
    net.load_state_dict(ckpt[weights])
    net = net.to(device).eval().requires_grad_(False)
    net.use_fp16 = False
    return net, cfg


def export_checkpoint(
    src: str | Path,
    dst: str | Path,
    *,
    extra: Mapping | None = None,
    drop: tuple[str, ...] = (),
) -> dict:
    """Write an inference-only copy of ``src`` (config + EMA weights).

    ``extra`` entries are added to the saved config, e.g. a ``description``
    (or, for an older flat config, ``channels``); ``drop`` keys are removed
    from it. The saved config is in the current format. Returns the saved
    dict.
    """
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    config = {k: v for k, v in ckpt["config"].items() if k not in drop}
    config.update(extra or {})
    config = OmegaConf.to_container(upgrade(config))
    out = {
        "config": config,
        "ema_state_dict": ckpt["ema_state_dict"],
        "step": ckpt.get("step"),
    }
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, dst)
    return out


def init_differences(cfg, source_cfg) -> list[str]:
    """Why weights from ``source_cfg`` would not mean the same in ``cfg``.

    Starting a run from another model's weights only makes sense when both
    see the same variables, in the same order and normalization, through the
    same network. The new model may add observation inputs to an
    unconditional one; otherwise the conditioning must match too.
    """
    from obottle_starter.core.obs_conditioning import settings

    diffs = []

    def strip(specs):
        out = []
        for s in specs:
            s = {k: s[k] for k in ("name", "units", "normalization")}
            clim = s["normalization"]["climatology"]
            if clim is not None:
                s["normalization"] = dict(
                    s["normalization"],
                    climatology={k: clim[k] for k in ("variable", "lookup")},
                )
            out.append(s)
        return out

    new, old = strip(variables(cfg)), strip(variables(source_cfg))
    if new != old:
        diffs.append(f"variables: {old} -> {new}")
    for key in ARCHITECTURE_KEYS:
        # Older configs lack per_frame_adaptive_scale; absent means false.
        default = False if key == "per_frame_adaptive_scale" else None
        a, b = cfg.get(key, default), source_cfg.get(key, default)
        a = list(a) if isinstance(a, (list, ListConfig)) else a
        b = list(b) if isinstance(b, (list, ListConfig)) else b
        if a != b:
            diffs.append(f"{key}: {b!r} -> {a!r}")
    s_new, s_old = settings(cfg), settings(source_cfg)
    if s_old["mode"] != "none" and s_new != s_old:
        diffs.append(f"conditioning: {s_old} -> {s_new}")
    return diffs


def load_weights_padded(
    net: EDMPrecond, state: Mapping, cfg, source_cfg=None
) -> list[str]:
    """Load ``state`` into ``net``, zero-filling new observation input channels.

    For starting a conditional model from an unconditional checkpoint. Only
    the first convolution (the input projection) may differ in shape: the
    observation channels are inserted, as zeros, right after the noisy-state
    channels, where the observation condition sits. The new model then
    initially ignores its observations and reproduces the old one. Every
    other tensor must match exactly; nothing is newly initialized.

    Args:
        net: The new network.
        state: The source ``state_dict``.
        cfg: The new network's config.
        source_cfg: The source's config. When given, :func:`init_differences`
            must be empty.

    Returns:
        Names of the padded tensors (empty or the input projection).
    """
    from obottle_starter.core.obs_conditioning import observation_channels

    if source_cfg is not None:
        diffs = init_differences(cfg, source_cfg)
        if diffs:
            raise ValueError(
                "init_from: the source model is not compatible:\n  "
                + "\n  ".join(diffs)
            )
        n_new = observation_channels(cfg) - observation_channels(source_cfg)
    else:
        n_new = observation_channels(cfg)
    own = net.state_dict()
    missing = sorted(set(own) - set(state))
    unexpected = sorted(set(state) - set(own))
    if missing or unexpected:
        raise ValueError(
            f"init_from: parameter names differ (missing {missing[:3]}, "
            f"unexpected {unexpected[:3]}); the networks are built differently"
        )
    start = num_channels(cfg)  # observation channels follow the noisy state
    padded, merged = [], {}
    for key, value in state.items():
        target = own[key]
        if target.shape != value.shape:
            extra = target.shape[1] - value.shape[1] if value.ndim > 1 else -1
            same_rest = value.ndim == target.ndim and all(
                a == b
                for i, (a, b) in enumerate(zip(value.shape, target.shape))
                if i != 1
            )
            input_projection = key.endswith("_conv_embed.module.weight")
            if not (input_projection and same_rest and 0 < extra == n_new):
                raise ValueError(f"cannot pad {key}: {value.shape} -> {target.shape}")
            new = torch.zeros_like(target)
            new[:, :start] = value[:, :start]
            new[:, start + extra :] = value[:, start:]
            value = new
            padded.append(key)
        merged[key] = value
    net.load_state_dict(merged)
    return padded
