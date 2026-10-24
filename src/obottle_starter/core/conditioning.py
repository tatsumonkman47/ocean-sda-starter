# SPDX-License-Identifier: Apache-2.0
"""Per-window conditioning: frame time offsets, calendar, labels.

The models see time in three ways, all derived from the frame timestamps:

* ``condition``: ``temporal_offset_channels`` sinusoidal encodings of each
  frame's offset from frame 0 (hours), broadcast over the grid and
  concatenated with the noisy state;
* ``time_offsets``: the raw offsets, used by temporal attention and the
  per-frame adaptive group norm;
* ``second_of_day`` (local solar time, from the tile's central longitude)
  and ``day_of_year`` (UTC, 0-based), used by the calendar embedding.

Frames may be irregularly spaced. The released checkpoints were trained on
offsets drawn on a 1-hour grid with a total span of up to 336 h.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor


def sinusoidal_time_encoding(
    offsets: Tensor,
    n_channels: int,
    period_min_hours: float = 7.2,
    period_max_hours: float = 7200.0,
) -> Tensor:
    """Encode ``(T,)`` offsets in hours as ``(n_channels, T)`` sin/cos features.

    Frequencies are log-spaced so the periods span
    ``[period_min_hours, period_max_hours]``. The defaults reproduce the
    encoder the released checkpoints were trained with, bit for bit.
    """
    if n_channels % 2:
        raise ValueError("n_channels must be even (sin/cos pairs)")
    n_freq = n_channels // 2
    if period_min_hours == 7.2 and period_max_hours == 7200.0:
        freqs = torch.logspace(-1, 2, n_freq)
        angles = 2.0 * torch.pi * freqs[:, None] * offsets[None, :] / 720.0
    else:
        periods = torch.logspace(
            math.log10(period_max_hours), math.log10(period_min_hours), n_freq
        )
        angles = 2.0 * torch.pi * offsets[None, :] / periods[:, None]
    return torch.cat([angles.sin(), angles.cos()], dim=0)


def span_label_encoding(
    span_hours: float, n_channels: int, max_hours: float = 720.0
) -> Tensor:
    """Encode a window span (hours) as ``(n_channels,)`` sin/cos label features."""
    if n_channels % 2:
        raise ValueError("n_channels must be even (sin/cos pairs)")
    span = max(0.0, min(float(span_hours), float(max_hours)))
    norm = math.log1p(span) / math.log1p(float(max_hours))
    freqs = torch.logspace(0, 1.5, n_channels // 2)
    angles = 2.0 * torch.pi * freqs * norm
    return torch.cat([angles.sin(), angles.cos()], dim=0)


def frame_conditioning(
    times,
    center_lon: float,
    cfg,
    device: torch.device | str | None = None,
) -> dict[str, Tensor | None]:
    """Conditioning for one window of frames, without a batch dimension.

    Args:
        times: ``(T,)`` frame times (anything ``numpy.datetime64`` accepts),
            in increasing order.
        center_lon: Tile-centre longitude in degrees east, for local solar
            time.
        cfg: Model config (``H``, ``W``, ``temporal_offset_channels``,
            ``label_dim``, ``span_label_channels``).
        device: Where to put the tensors.

    Returns:
        ``condition`` ``(C_time, T, H*W)`` or ``None``, ``class_labels``
        ``(label_dim,)`` or ``None``, and ``(T,)`` ``second_of_day``,
        ``day_of_year`` and ``time_offsets``. These are the keyword arguments
        of the network's forward pass.
    """
    times = np.asarray(times, dtype="datetime64[ns]")
    if len(times) != cfg.time_length:
        raise ValueError(
            f"model expects {cfg.time_length} frames; got {len(times)} times"
        )
    seconds = (times - times[0]) / np.timedelta64(1, "s")
    if np.any(np.diff(seconds) <= 0):
        raise ValueError("frame times must be strictly increasing")
    offsets = torch.from_numpy((seconds / 3600.0).astype(np.float32))

    utc_seconds = (times - times.astype("datetime64[D]")) / np.timedelta64(1, "s")
    local_seconds = (utc_seconds + center_lon * 240.0) % 86400.0
    day_of_year = (times.astype("datetime64[D]") - times.astype("datetime64[Y]")) / (
        np.timedelta64(1, "D")
    )

    condition = None
    if cfg.temporal_offset_channels > 0:
        enc = sinusoidal_time_encoding(offsets, cfg.temporal_offset_channels)
        condition = enc.unsqueeze(-1).expand(-1, -1, cfg.H * cfg.W)

    labels = None
    n_span = cfg.get("span_label_channels", 0)
    if cfg.label_dim + n_span > 0:
        labels = torch.zeros(cfg.label_dim + n_span)
        if n_span:
            labels[cfg.label_dim :] = span_label_encoding(
                float(offsets[-1]), n_span, cfg.get("span_label_max_hours", 720.0)
            )

    out = {
        "condition": condition,
        "class_labels": labels,
        "second_of_day": torch.from_numpy(local_seconds.astype(np.float32)),
        "day_of_year": torch.from_numpy(day_of_year.astype(np.float32)),
        "time_offsets": offsets,
    }
    if device is not None:
        out = {k: None if v is None else v.to(device) for k, v in out.items()}
    return out
