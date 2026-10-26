# SPDX-License-Identifier: Apache-2.0
r"""Long sequences from a short-window prior: sliding windows (Rozet & Louppe, 2023).

A prior is trained on windows of ``T`` frames (5 or 11 for the released
models), but an assimilation period is often longer. Rozet & Louppe (2023,
"Score-based Data Assimilation", section 3.1 and Algorithm 2) build the
score of an ``L``-frame sequence from a ``T``-frame one: run it on every
window of ``T`` consecutive frames and keep, for each frame, the output of
the window in which it is most central. For ``T = 2k + 1``, frame ``i``
comes from the window centred on it, and the first and last ``k`` frames
from the first and last windows. For a Markov chain, a frame's score
depends only on its neighbours (its Markov blanket); for noisy frames the
``k`` frames on each side are a "pseudo" blanket, an approximation that is
exact at the two ends of the noise range and unbiased in between. How good
it is for ocean surface fields is an empirical question.

This module writes that construction for denoisers instead of scores. The
two are equivalent through Tweedie's formula, ``score = (D(x_t) - x_t) / σ_t²``,
applied frame by frame. The assembled denoiser is differentiable, so
guidance applies to the whole sequence unchanged: Rozet guidance
differentiates through it, and MMPS takes its Jacobian as the covariance.
That Jacobian is assembled from the windows' Jacobians and is not exactly
symmetric, so use the ``gmres`` solver (the default), not ``cg``.

``step`` generalizes the layout: consecutive windows start ``step`` frames
apart, and each window contributes its central ``step`` frames. ``step = 1``
is Rozet & Louppe's construction; ``step = T`` splits the sequence into
independent blocks, a baseline that shows the seams sliding windows avoid.
Cost per denoiser call is one network evaluation per window,
``(L - T) / step + 1`` of them.

Each window gets its own conditioning: frame offsets from its own first
frame, and the calendar of its own frames, as in training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from obottle_starter.core.conditioning import frame_conditioning
from obottle_starter.inference.sda._azula.denoise import Denoiser, DiracPosterior
from obottle_starter.inference.sda._azula.noise import Schedule

__all__ = ["SlidingWindows", "WindowedDenoiser", "sliding_windows", "window_layout"]

COND_KEYS = (
    "condition",
    "class_labels",
    "second_of_day",
    "day_of_year",
    "time_offsets",
)


def window_layout(
    n_frames: int, window: int, step: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Where each frame of a sequence comes from.

    Args:
        n_frames: Sequence length ``L``.
        window: The model's window length ``T``.
        step: Frames between consecutive window starts, ``1 <= step <= T``.

    Returns:
        ``starts`` ``(n_windows,)``: first frame of each window; and, per
        sequence frame, ``source`` ``(L,)``: the window it is taken from,
        and ``local`` ``(L,)``: its index inside that window.

    Raises:
        ValueError: If the windows cannot tile the sequence exactly, with
            the nearest lengths that work.
    """
    L, T, s = int(n_frames), int(window), int(step)
    if not 1 <= s <= T:
        raise ValueError(
            f"window step must be between 1 and the window length {T}; got {s}"
        )
    if L < T:
        raise ValueError(
            f"the sequence has {L} frames, fewer than the model's window of {T}"
        )
    if (L - T) % s:
        shorter = L - (L - T) % s
        raise ValueError(
            f"{L} frames cannot be covered by windows of {T} frames starting every "
            f"{s} frames: (frames - {T}) must be a multiple of {s}. "
            f"Use {shorter} or {shorter + s} frames, or another step."
        )
    starts = np.arange(0, L - T + 1, s)
    lo = (T - s) // 2  # first central frame of a window
    source = np.empty(L, dtype=np.int64)
    local = np.empty(L, dtype=np.int64)
    for w, start in enumerate(starts):
        keep = np.arange(lo, lo + s)
        if w == 0:
            keep = np.arange(0, lo + s)
        if w == len(starts) - 1:
            keep = np.arange(keep[0], T)
        source[start + keep] = w
        local[start + keep] = keep
    return starts, source, local


@dataclass
class SlidingWindows:
    """A window layout and the per-window conditioning for one sequence.

    Built by :func:`sliding_windows`. Conditioning tensors carry a leading
    window dimension: ``condition`` ``(n_windows, C_time, T, H*W)``,
    ``second_of_day`` ``(n_windows, T)``, and so on, or ``None``.
    """

    window: int
    step: int
    starts: Tensor
    source: Tensor
    local: Tensor
    condition: Optional[Tensor] = None
    class_labels: Optional[Tensor] = None
    second_of_day: Optional[Tensor] = None
    day_of_year: Optional[Tensor] = None
    time_offsets: Optional[Tensor] = None

    @property
    def n_windows(self) -> int:
        return len(self.starts)

    @property
    def n_frames(self) -> int:
        return len(self.source)

    def describe(self) -> str:
        return (
            f"sliding windows: {self.n_frames} frames, {self.n_windows} windows of "
            f"{self.window} frames, step {self.step}"
            + (" (Rozet & Louppe)" if self.step == 1 else "")
            + (" (independent blocks)" if self.step == self.window else "")
        )


def sliding_windows(
    times, center_lon: float, cfg, step: int = 1, device=None
) -> SlidingWindows:
    """Lay out windows of ``cfg.time_length`` frames over ``times``.

    Args:
        times: ``(L,)`` frame times, strictly increasing, ``L >= T``.
        center_lon: Window-centre longitude (local solar time).
        cfg: Model config.
        step: See :func:`window_layout`.
        device: Where to put the tensors.
    """
    times = np.asarray(times, dtype="datetime64[ns]")
    T = int(cfg.time_length)
    starts, source, local = window_layout(len(times), T, step)
    per_window = [frame_conditioning(times[a : a + T], center_lon, cfg) for a in starts]
    stacked = {
        key: None
        if per_window[0][key] is None
        else torch.stack([c[key] for c in per_window]).to(device)
        for key in COND_KEYS
    }
    as_tensor = lambda a: torch.as_tensor(a, device=device)  # noqa: E731
    return SlidingWindows(
        window=T,
        step=int(step),
        starts=as_tensor(starts),
        source=as_tensor(source),
        local=as_tensor(local),
        **stacked,
    )


class WindowedDenoiser(Denoiser):
    """The denoiser of an ``L``-frame sequence, assembled from ``T``-frame windows.

    Args:
        net: A network from :func:`obottle_starter.core.model.load_checkpoint`
            (an unconditional prior).
        schedule: Noise schedule, as for
            :class:`obottle_starter.inference.sda.denoiser.EDMPrecondDenoiser`.
        windows: Layout and conditioning from :func:`sliding_windows`, on the
            network's device.
        window_batch: Most windows (times ensemble members) per network
            call, or ``None`` for all at once. Lowers the peak memory of a
            call; with guidance, the activations of every window are still
            kept for the backward pass.

    ``forward`` takes ``x_t`` ``(B, C, L, H*W)`` and returns the assembled
    estimate of the same shape. With one window (``L == T``) it equals
    :class:`~obottle_starter.inference.sda.denoiser.EDMPrecondDenoiser`.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        schedule: Schedule,
        windows: SlidingWindows,
        window_batch: Optional[int] = None,
    ) -> None:
        super().__init__()
        if window_batch is not None and int(window_batch) < 1:
            raise ValueError(f"window_batch must be >= 1; got {window_batch}")
        self.net = net
        self.schedule = schedule
        self.windows = windows
        self.window_batch = None if window_batch is None else int(window_batch)
        T = windows.window
        # (n_windows, T): sequence frame of each window frame.
        self._frames = windows.starts[:, None] + torch.arange(
            T, device=windows.starts.device
        )

    def forward(self, x_t: Tensor, t: Tensor, **kwargs) -> DiracPosterior:
        w = self.windows
        B, C, L, HW = x_t.shape
        if L != w.n_frames:
            raise ValueError(f"x_t has {L} frames; the windows cover {w.n_frames}")
        n, T = w.n_windows, w.window
        # Unfold: (B, C, n, T, HW) -> (B * n, C, T, HW), window index fastest.
        windows = (
            x_t[:, :, self._frames].permute(0, 2, 1, 3, 4).reshape(B * n, C, T, HW)
        )
        _, sigma_t = self.schedule(t)
        sigma = sigma_t.reshape(-1).expand(B).repeat_interleave(n)
        sigma = sigma.to(device=x_t.device, dtype=x_t.dtype)
        cond = {}
        for key in COND_KEYS:
            value = getattr(w, key)
            if value is not None:
                cond[key] = (
                    value.unsqueeze(0)
                    .expand(B, *value.shape)
                    .reshape(B * n, *value.shape[1:])
                )
        chunk = self.window_batch or B * n
        outs = []
        for a in range(0, B * n, chunk):
            part = {k: v[a : a + chunk] for k, v in cond.items()}
            outs.append(
                self.net(
                    windows[a : a + chunk],
                    sigma[a : a + chunk],
                    force_fp32=True,
                    **part,
                ).out
            )
        out = torch.cat(outs).reshape(B, n, C, T, HW).permute(0, 2, 1, 3, 4)
        # Fold: frame j of the sequence is frame local[j] of window source[j].
        return DiracPosterior(mean=out[:, :, w.source, w.local])
