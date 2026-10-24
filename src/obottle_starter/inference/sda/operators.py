# SPDX-License-Identifier: Apache-2.0
"""General linear observation operators and observation-error covariances.

An opt-in alternative to the mask-based :class:`ObservationTerm`: here an
observation does not have to sit at a grid pixel. Every observation ``i`` is
a weighted sum of state values plus a constant,

    y_i = Σ_k w_ik · x[c_ik, t_ik, p_ik] + offset_i + e_i,

stored as a sparse matrix (``index`` and ``weight``, each ``(N, K)``, into
the flattened ``(C, T, H*W)`` state). That one form covers:

* off-grid points: :func:`point_observations` (bilinear in space, linear in
  time between frames), e.g. along-track altimetry;
* area and footprint averages: :func:`block_average` (coarse-to-fine) and
  :func:`footprint_observations` (any kernel, e.g. :func:`gaussian_kernel`);
* combinations of variables and time averages, by adding and scaling
  observations with the same number of rows: ``0.5 * (obs_t0 + obs_t1)``.

The errors ``e`` have covariance :class:`ObservationCovariance`: diagonal,
or block-diagonal with dense blocks for correlated errors (e.g. one block per
track, :func:`exponential_covariance`).

:class:`ObservationSet` concatenates several observations. Build the
inference input with :func:`build_general_observation` (model units) or
:func:`obottle_starter.inference.general.model_unit_observation` (physical
units, scaled consistently). Both plug into
:func:`obottle_starter.inference.sda.run_inference` like the mask-based
observations. Operators are linear (affine), so MMPS's linearization is
exact for them; nonlinear operators are not provided.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from obottle_starter.inference.sda.observations import ObservationSpec

__all__ = [
    "ObservationCovariance",
    "ObservationSet",
    "SparseObservation",
    "as_covariance",
    "block_average",
    "build_general_observation",
    "exponential_covariance",
    "footprint_observations",
    "gaussian_kernel",
    "point_observations",
]


# ── Operators ─────────────────────────────────────────────────────────────


class SparseObservation:
    """``N`` observations, each a weighted sum of state values plus a constant.

    Args:
        name: Label for diagnostics (one group).
        state_shape: ``(C, T, H, W)`` of the state it observes.
        index: ``(N, K)`` indices into the flattened ``(C, T, H*W)`` state
            (``((c * T + t) * H + row) * W + col``).
        weight: ``(N, K)`` weights; unused slots have weight 0.
        offset: Constant added to each observation, scalar or ``(N,)``.
        coords: Optional ``{name: (N,) array}`` metadata (e.g. ``frame``,
            ``row``, ``col``, or real latitudes), carried to the output.
    """

    def __init__(
        self,
        name: str,
        state_shape: Sequence[int],
        index,
        weight,
        offset=0.0,
        coords: Optional[Mapping[str, np.ndarray]] = None,
    ) -> None:
        self.name = str(name)
        self.state_shape = tuple(int(s) for s in state_shape)
        if len(self.state_shape) != 4:
            raise ValueError("state_shape must be (C, T, H, W)")
        self.index = torch.as_tensor(index, dtype=torch.long)
        self.weight = torch.as_tensor(weight, dtype=torch.float32)
        if self.index.ndim != 2 or self.index.shape != self.weight.shape:
            raise ValueError("index and weight must both be (N, K)")
        size = math.prod(self.state_shape)
        if self.index.numel() and (self.index.min() < 0 or self.index.max() >= size):
            raise ValueError(f"index outside the state of shape {self.state_shape}")
        n = self.index.shape[0]
        self.offset = torch.as_tensor(offset, dtype=torch.float32).expand(n).clone()
        self.coords = {k: np.asarray(v) for k, v in (coords or {}).items()}
        for k, v in self.coords.items():
            if v.shape != (n,):
                raise ValueError(f"coords[{k!r}] must have shape ({n},)")

    @property
    def n_obs(self) -> int:
        return int(self.index.shape[0])

    def __call__(self, x: Tensor) -> Tensor:
        """``(..., C, T, H*W)`` state to ``(..., N)`` observations."""
        flat = x.reshape(*x.shape[:-3], -1)
        values = flat[..., self.index] * self.weight.to(flat.dtype)
        return values.sum(-1) + self.offset.to(flat.dtype)

    def _like(self, index, weight, offset, name=None) -> SparseObservation:
        return SparseObservation(
            name or self.name, self.state_shape, index, weight, offset, self.coords
        )

    def __mul__(self, scale: float) -> SparseObservation:
        return self._like(self.index, self.weight * scale, self.offset * scale)

    __rmul__ = __mul__

    def __add__(self, other: SparseObservation) -> SparseObservation:
        """Row-wise sum: observation ``i`` of the result is ``self_i + other_i``."""
        if not isinstance(other, SparseObservation):
            return NotImplemented
        if other.n_obs != self.n_obs or other.state_shape != self.state_shape:
            raise ValueError("can only add observations with the same rows and state")
        return self._like(
            torch.cat([self.index, other.index.to(self.index.device)], dim=1),
            torch.cat([self.weight, other.weight.to(self.weight.device)], dim=1),
            self.offset + other.offset.to(self.offset.device),
        )

    def renamed(self, name: str) -> SparseObservation:
        return self._like(self.index, self.weight, self.offset, name)

    # The interface run_inference and the guidance classes use.
    @property
    def sizes(self) -> list[int]:
        return [self.n_obs]

    @property
    def names(self) -> list[str]:
        return [self.name]

    @property
    def groups(self) -> Tensor:
        return torch.zeros(self.n_obs, dtype=torch.long, device=self.index.device)

    def footprint(self, n_channels: int) -> Tensor:
        """``(C, T, H*W)`` mask of the state values any observation uses."""
        C, T, H, W = self.state_shape
        out = torch.zeros(C * T * H * W, dtype=torch.bool, device=self.index.device)
        out[self.index[self.weight != 0]] = True
        return out.reshape(C, T, H * W)[:n_channels]

    def to(self, device) -> SparseObservation:
        out = self._like(
            self.index.to(device), self.weight.to(device), self.offset.to(device)
        )
        return out


class ObservationSet:
    """Several observations concatenated, in order, into one vector."""

    def __init__(self, parts: Sequence[SparseObservation]) -> None:
        if not parts:
            raise ValueError("an observation set needs at least one part")
        shapes = {p.state_shape for p in parts}
        if len(shapes) != 1:
            raise ValueError(f"parts observe different state shapes: {shapes}")
        self.parts = list(parts)
        self.state_shape = parts[0].state_shape

    def __call__(self, x: Tensor) -> Tensor:
        return torch.cat([p(x) for p in self.parts], dim=-1)

    @property
    def n_obs(self) -> int:
        return sum(p.n_obs for p in self.parts)

    @property
    def sizes(self) -> list[int]:
        return [p.n_obs for p in self.parts]

    @property
    def names(self) -> list[str]:
        return [p.name for p in self.parts]

    @property
    def groups(self) -> Tensor:
        device = self.parts[0].index.device
        return torch.cat(
            [
                torch.full((p.n_obs,), k, dtype=torch.long, device=device)
                for k, p in enumerate(self.parts)
            ]
        )

    @property
    def coords(self) -> dict[str, np.ndarray]:
        """Coordinates of all parts, NaN where a part lacks one."""
        keys = sorted({k for p in self.parts for k in p.coords})
        return {
            k: np.concatenate(
                [
                    p.coords[k].astype(float)
                    if k in p.coords
                    else np.full(p.n_obs, np.nan)
                    for p in self.parts
                ]
            )
            for k in keys
        }

    def footprint(self, n_channels: int) -> Tensor:
        out = self.parts[0].footprint(n_channels)
        for p in self.parts[1:]:
            out = out | p.footprint(n_channels)
        return out

    def to(self, device) -> ObservationSet:
        return ObservationSet([p.to(device) for p in self.parts])


def _as_set(operator) -> ObservationSet:
    return (
        operator if isinstance(operator, ObservationSet) else ObservationSet([operator])
    )


def _channel_weights(channel) -> dict[int, float]:
    if isinstance(channel, Mapping):
        return {int(c): float(w) for c, w in channel.items()}
    return {int(channel): 1.0}


def _flat(shape, c, t, r, col):
    _, T, H, W = shape
    return ((c * T + t) * H + r) * W + col


def _lerp_axis(pos: np.ndarray, n: int, what: str):
    """Lower index and fraction for linear interpolation on 0..n-1."""
    pos = np.asarray(pos, dtype=float)
    if np.any(pos < 0) or np.any(pos > n - 1) or not np.all(np.isfinite(pos)):
        raise ValueError(f"{what} positions must lie in [0, {n - 1}]")
    if n == 1:
        return np.zeros(pos.shape, dtype=int), np.zeros(pos.shape)
    lo = np.minimum(np.floor(pos).astype(int), n - 2)
    return lo, pos - lo


def point_observations(
    state_shape: Sequence[int],
    channel,
    frame,
    row,
    col,
    *,
    name: str = "points",
    coords: Optional[Mapping[str, np.ndarray]] = None,
) -> SparseObservation:
    """Observations at arbitrary positions, interpolated from the grid.

    Args:
        state_shape: ``(C, T, H, W)``.
        channel: Variable index, or ``{index: coefficient}`` for a
            combination of variables at the same points.
        frame, row, col: ``(N,)`` fractional positions in grid units (cell
            centres at integers; frame 2.5 is halfway between frames 2 and 3).
            Space is bilinear, time linear in the frame index; convert
            observation times to fractional frames yourself, using the
            window's frame times.
        name: Group label.
        coords: Extra metadata (e.g. latitude), ``{name: (N,)}``.
    """
    shape = tuple(int(s) for s in state_shape)
    _, T, H, W = shape
    t0, ft = _lerp_axis(frame, T, "frame")
    r0, fr = _lerp_axis(row, H, "row")
    c0, fc = _lerp_axis(col, W, "col")
    index, weight = [], []
    for ch, coeff in _channel_weights(channel).items():
        for dt, wt in ((0, 1 - ft), (1, ft)):
            for dr, wr in ((0, 1 - fr), (1, fr)):
                for dc, wc in ((0, 1 - fc), (1, fc)):
                    t = np.minimum(t0 + dt, T - 1)
                    r = np.minimum(r0 + dr, H - 1)
                    c = np.minimum(c0 + dc, W - 1)
                    index.append(_flat(shape, ch, t, r, c))
                    weight.append(coeff * wt * wr * wc)
    meta = {
        "frame": np.asarray(frame, float),
        "row": np.asarray(row, float),
        "col": np.asarray(col, float),
        **(coords or {}),
    }
    return SparseObservation(
        name, shape, np.stack(index, 1), np.stack(weight, 1), coords=meta
    )


def block_average(
    state_shape: Sequence[int],
    channel,
    block: int | tuple[int, int],
    *,
    frames: Optional[Sequence[int]] = None,
    name: str = "block_average",
) -> SparseObservation:
    """The mean over each ``block``-sized tile of the grid, in every frame.

    The coarse-to-fine operator: a coarse product whose cells are ``block``
    fine cells on a side. One observation per coarse cell and frame, ordered
    by frame, then coarse row, then coarse column. Coordinates are the
    coarse-cell centres in fine-grid units.
    """
    shape = tuple(int(s) for s in state_shape)
    _, T, H, W = shape
    bh, bw = (block, block) if isinstance(block, int) else block
    if H % bh or W % bw:
        raise ValueError(f"block {bh}x{bw} does not tile the {H}x{W} grid")
    frames = list(range(T)) if frames is None else [int(f) for f in frames]
    t, R, Cc = np.meshgrid(
        frames, np.arange(H // bh), np.arange(W // bw), indexing="ij"
    )
    t, R, Cc = t.ravel(), R.ravel(), Cc.ravel()
    dr, dc = np.meshgrid(np.arange(bh), np.arange(bw), indexing="ij")
    dr, dc = dr.ravel(), dc.ravel()
    index, weight = [], []
    for ch, coeff in _channel_weights(channel).items():
        index.append(
            _flat(shape, ch, t[:, None], R[:, None] * bh + dr, Cc[:, None] * bw + dc)
        )
        weight.append(np.full(index[-1].shape, coeff / (bh * bw)))
    coords = {
        "frame": t.astype(float),
        "row": R * bh + (bh - 1) / 2,
        "col": Cc * bw + (bw - 1) / 2,
    }
    return SparseObservation(
        name, shape, np.concatenate(index, 1), np.concatenate(weight, 1), coords=coords
    )


def gaussian_kernel(
    sigma_cells: float, radius_cells: Optional[int] = None
) -> np.ndarray:
    """A normalized 2-D Gaussian footprint, ``(2r+1, 2r+1)``, r = 3 sigma by default."""
    r = math.ceil(3 * sigma_cells) if radius_cells is None else int(radius_cells)
    d = np.arange(-r, r + 1)
    k = np.exp(-(d[:, None] ** 2 + d[None, :] ** 2) / (2 * sigma_cells**2))
    return k / k.sum()


def footprint_observations(
    state_shape: Sequence[int],
    channel,
    frame,
    row,
    col,
    kernel: np.ndarray,
    *,
    name: str = "footprints",
) -> SparseObservation:
    """Kernel-weighted averages centred at integer grid cells.

    ``kernel`` (odd sizes) is placed at each ``(frame, row, col)``. Parts of
    it outside the grid are dropped and the rest renormalized to the
    kernel's sum, so edge observations are still averages.
    """
    shape = tuple(int(s) for s in state_shape)
    _, _, H, W = shape
    kernel = np.asarray(kernel, dtype=float)
    kh, kw = kernel.shape
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError("kernel sizes must be odd so it has a centre")
    frame, row, col = (np.asarray(a, dtype=int) for a in (frame, row, col))
    dr, dc = np.meshgrid(
        np.arange(kh) - kh // 2, np.arange(kw) - kw // 2, indexing="ij"
    )
    r = row[:, None] + dr.ravel()
    c = col[:, None] + dc.ravel()
    inside = (r >= 0) & (r < H) & (c >= 0) & (c < W)
    base = np.where(inside, kernel.ravel()[None, :], 0.0)
    base *= kernel.sum() / np.maximum(base.sum(1, keepdims=True), 1e-30)
    r, c = np.clip(r, 0, H - 1), np.clip(c, 0, W - 1)
    index, weight = [], []
    for ch, coeff in _channel_weights(channel).items():
        index.append(_flat(shape, ch, frame[:, None], r, c))
        weight.append(coeff * base)
    coords = {
        "frame": frame.astype(float),
        "row": row.astype(float),
        "col": col.astype(float),
    }
    return SparseObservation(
        name, shape, np.concatenate(index, 1), np.concatenate(weight, 1), coords=coords
    )


# ── Covariances ───────────────────────────────────────────────────────────


class ObservationCovariance:
    """Block-diagonal observation-error covariance ``R``.

    Each block covers a contiguous run of observations and is either a
    ``(n,)`` vector of variances (independent errors) or an ``(n, n)``
    symmetric positive definite matrix (correlated errors, e.g. along one
    track). The blocks follow the observation order. A single vector block
    is the usual diagonal ``R`` and behaves exactly like the plain ``cov_y``
    tensor the guidance classes accepted before.
    """

    def __init__(self, blocks: Sequence[Tensor]) -> None:
        self.blocks = [torch.as_tensor(b) for b in blocks]
        single_scalar = len(self.blocks) == 1 and self.blocks[0].ndim == 0
        for b in self.blocks:
            if single_scalar:
                break  # one variance for every observation (legacy cov_y)
            if b.ndim not in (1, 2) or (b.ndim == 2 and b.shape[0] != b.shape[1]):
                raise ValueError("blocks must be (n,) variances or (n, n) matrices")
        self._chol: list[Optional[Tensor]] = [None] * len(self.blocks)

    @property
    def n(self) -> int:
        return sum(b.shape[0] for b in self.blocks)

    def validate(self, what: str = "covariance") -> ObservationCovariance:
        """Fail unless every block is a finite, valid covariance; returns self.

        Variances must be finite and positive; dense blocks symmetric and
        positive definite (checked by a Cholesky factorization).
        """
        for k, b in enumerate(self.blocks):
            if not bool(torch.isfinite(b).all()):
                raise ValueError(f"{what}: block {k} has NaN or inf entries")
            if b.ndim <= 1:
                if not bool((b > 0).all()):
                    raise ValueError(f"{what}: block {k} has variances <= 0")
                continue
            if not torch.allclose(b, b.T, rtol=1e-5, atol=1e-12):
                raise ValueError(f"{what}: block {k} is not symmetric")
            try:
                self._factor(k)
            except torch.linalg.LinAlgError:
                raise ValueError(
                    f"{what}: block {k} is not positive definite; add an "
                    "uncorrelated part (e.g. exponential_covariance(nugget=...))"
                ) from None
        return self

    @property
    def is_diagonal(self) -> bool:
        return all(b.ndim == 1 for b in self.blocks)

    def _slices(self):
        start = 0
        for b in self.blocks:
            yield slice(start, start + b.shape[0]), b
            start += b.shape[0]

    def diagonal(self) -> Tensor:
        """``diag(R)``, ``(N,)``."""
        if len(self.blocks) == 1 and self.blocks[0].ndim <= 1:
            return self.blocks[0]
        return torch.cat([b if b.ndim == 1 else b.diagonal() for b in self.blocks])

    def matvec(self, v: Tensor) -> Tensor:
        """``R v`` for ``v`` of shape ``(..., N)``."""
        if len(self.blocks) == 1 and self.blocks[0].ndim <= 1:
            return self.blocks[0] * v
        parts = []
        for sl, b in self._slices():
            u = v[..., sl]
            parts.append(b * u if b.ndim == 1 else u @ b.to(u.dtype).T)
        return torch.cat(parts, dim=-1)

    def _factor(self, k: int) -> Tensor:
        if self._chol[k] is None:
            self._chol[k] = torch.linalg.cholesky(self.blocks[k])
        return self._chol[k]

    def solve(self, v: Tensor) -> Tensor:
        """``R⁻¹ v`` (Cholesky per dense block)."""
        if len(self.blocks) == 1 and self.blocks[0].ndim <= 1:
            return v / self.blocks[0]
        parts = []
        for k, (sl, b) in enumerate(self._slices()):
            u = v[..., sl]
            if b.ndim == 1:
                parts.append(u / b)
            else:
                L = self._factor(k).to(u.dtype)
                parts.append(torch.cholesky_solve(u.unsqueeze(-1), L).squeeze(-1))
        return torch.cat(parts, dim=-1)

    def whiten(self, v: Tensor) -> Tensor:
        """``L⁻¹ v`` with ``R = L Lᵀ``: residuals in units of their error."""
        if len(self.blocks) == 1 and self.blocks[0].ndim <= 1:
            return v / self.blocks[0].sqrt()
        parts = []
        for k, (sl, b) in enumerate(self._slices()):
            u = v[..., sl]
            if b.ndim == 1:
                parts.append(u / b.sqrt())
            else:
                L = self._factor(k).to(u.dtype)
                parts.append(
                    torch.linalg.solve_triangular(
                        L, u.unsqueeze(-1), upper=False
                    ).squeeze(-1)
                )
        return torch.cat(parts, dim=-1)

    def shifted(self, s) -> ObservationCovariance:
        """``R + s I`` (``s`` a scalar or 0-d tensor)."""
        out = []
        for b in self.blocks:
            if b.ndim <= 1:
                out.append(b + s)
            else:
                out.append(
                    b + s * torch.eye(b.shape[0], dtype=b.dtype, device=b.device)
                )
        return ObservationCovariance(out)

    def scaled(self, q: Tensor) -> ObservationCovariance:
        """``Q R Q`` for a diagonal scaling ``q`` (``(N,)``)."""
        out = []
        for sl, b in self._slices():
            qs = q[sl].to(b.dtype)
            out.append(b * qs**2 if b.ndim == 1 else qs[:, None] * b * qs[None, :])
        return ObservationCovariance(out)

    def sample(self, generator: Optional[torch.Generator] = None) -> Tensor:
        """One draw of ``e ~ N(0, R)``, ``(N,)``, on the CPU."""
        parts = []
        for k, (_, b) in enumerate(self._slices()):
            eps = torch.randn(b.shape[0], generator=generator, dtype=torch.float64)
            if b.ndim == 1:
                parts.append(b.double().cpu().sqrt() * eps)
            else:
                parts.append(torch.linalg.cholesky(b.double().cpu()) @ eps)
        return torch.cat(parts)

    def to(self, device=None, dtype=None) -> ObservationCovariance:
        return ObservationCovariance(
            [b.to(device=device, dtype=dtype) for b in self.blocks]
        )

    @staticmethod
    def concat(covs: Sequence[ObservationCovariance]) -> ObservationCovariance:
        return ObservationCovariance([b for c in covs for b in c.blocks])


def as_covariance(cov) -> ObservationCovariance:
    """An :class:`ObservationCovariance` from itself or a ``(N,)`` variance tensor."""
    if isinstance(cov, ObservationCovariance):
        return cov
    return ObservationCovariance([torch.as_tensor(cov)])


def exponential_covariance(
    positions, sigma: float, length: float, nugget: float = 0.0
) -> Tensor:
    """``sigma² exp(-|d| / length) + nugget² I`` for 1-D ``positions``.

    For errors correlated along a track: ``positions`` are along-track
    distances (any unit, the same as ``length``). Use one block per track.
    ``nugget`` adds an uncorrelated part, which also keeps the matrix well
    conditioned when points are close.
    """
    p = torch.as_tensor(np.asarray(positions, dtype=float))
    d = (p[:, None] - p[None, :]).abs()
    return sigma**2 * torch.exp(-d / length) + nugget**2 * torch.eye(
        len(p), dtype=p.dtype
    )


# ── Building the inference input ──────────────────────────────────────────


def build_general_observation(
    operator: SparseObservation | ObservationSet,
    covariance,
    *,
    truth: Optional[Tensor] = None,
    y: Optional[Tensor] = None,
    instrument_noise: bool | ObservationCovariance = False,
    seed: Optional[int] = None,
) -> ObservationSpec:
    """The general-operator counterpart of ``build_observation`` (model units).

    Args:
        operator: The observations.
        covariance: ``R`` as an :class:`ObservationCovariance`, a ``(N,)``
            variance tensor, or a list with one per part of an
            :class:`ObservationSet`.
        truth: ``(C, T, H, W)`` or ``(C, T, H*W)`` truth, for an OSSE.
        y: ``(N,)`` real observations instead.
        instrument_noise: Noise added to synthesized ``y``: ``False``,
            ``True`` (drawn from ``covariance``), or another covariance.
        seed: Seed for the noise.
    """
    operator = _as_set(operator)
    C, T, H, W = operator.state_shape
    if isinstance(covariance, (list, tuple)):
        covariance = ObservationCovariance.concat(
            [as_covariance(c) for c in covariance]
        )
    covariance = as_covariance(covariance)
    covariance.validate("covariance")
    if covariance.n != operator.n_obs:
        raise ValueError(
            f"covariance has {covariance.n} rows for {operator.n_obs} observations"
        )
    noise = None
    if truth is not None:
        truth = torch.as_tensor(truth, dtype=torch.float32).reshape(C, T, H * W)
    if y is None:
        if truth is None:
            raise ValueError("Pass truth to synthesize observations, or pass y")
        y = operator(truth)
        if instrument_noise is not False:
            source = (
                covariance
                if instrument_noise is True
                else as_covariance(instrument_noise)
            )
            gen = torch.Generator()
            if seed is not None:
                gen.manual_seed(int(seed))
            else:
                gen.seed()
            noise = source.validate("instrument_noise").sample(gen).to(y.dtype)
            y = y + noise
    else:
        y = torch.as_tensor(y, dtype=torch.float32)
        if y.shape != (operator.n_obs,):
            raise ValueError(
                f"y must have shape ({operator.n_obs},); got {tuple(y.shape)}"
            )
    if not bool(torch.isfinite(y).all()):
        raise ValueError("observations y must be finite (NaN or inf found)")
    return ObservationSpec(
        y=y,
        cov_y=covariance.to(dtype=y.dtype),
        operator=operator,
        H=H,
        W=W,
        truth=truth,
        noise=noise,
    )
