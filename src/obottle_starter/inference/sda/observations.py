# SPDX-License-Identifier: Apache-2.0
"""Observation operators and synthetic (OSSE) observations.

States use the network layout ``(..., C, T, H*W)`` in normalized units. An
observation is built from :class:`ObservationTerm` objects. Each term is a
weighted sum of channels plus an optional offset, sampled where its
``(T, H*W)`` mask is true:

    y_term = (Σ_c w_c L_c(x_c) + offset)[mask]

where ``L_c`` is the identity unless the term gives a linear spatial
transform for channel ``c`` (``transforms``), such as a smoothing or
along-track filter. With one, an observed pixel depends on the whole frame
of that channel; ``footprint`` still marks the observed pixels.

Typical terms, all in model (normalized) units:

* a plain per-channel mask: ``weights={c: 1.0}``
  (:meth:`obottle_starter.core.normalization.Normalizer.observation_terms`);
* a smoothed or footprint-averaged channel: ``transforms={c: L}``;
* a linear combination of channels, e.g. a sum of two SSH components.

Weights and offsets could also express observations in physical units (e.g.
``weights={0: std_ssh}``), and the exact posterior would be the same. But
MMPS truncates its Krylov solve, which then favours whichever term has the
largest residuals in its own units, so keep ``y`` and ``sigma_y`` in model
units (see :mod:`obottle_starter.core.normalization`). The offset does not
depend on the state, so the operator stays linear in ``x`` up to a constant
and MMPS differentiates it like any other.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Optional

import torch
from torch import Tensor


@dataclass
class ObservationTerm:
    """One observed quantity: a weighted channel sum sampled on a mask.

    Attributes:
        name: Label, e.g. ``"SSHA"`` or ``"SST"``. Used for diagnostics.
        mask: Boolean ``(T, H*W)`` or ``(T, H, W)``; true where observed.
        weights: ``{channel index: coefficient}``.
        offset: Scalar or tensor broadcastable to ``(T, H*W)`` added before
            sampling.
        transforms: Optional ``{channel index: L}``, a linear map applied to
            that channel's ``(..., T, H*W)`` field before its weight. It must
            be linear for MMPS to stay exact.
    """

    name: str
    mask: Tensor
    weights: Mapping[int, float]
    offset: float | Tensor = 0.0
    transforms: Mapping[int, Callable[[Tensor], Tensor]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.weights:
            raise ValueError(f"Observation term {self.name!r} has no channels")
        self.weights = {int(c): float(w) for c, w in self.weights.items()}
        self.transforms = {int(c): f for c, f in self.transforms.items()}
        if not set(self.transforms) <= set(self.weights):
            raise ValueError(f"Term {self.name!r} transforms an unweighted channel")
        mask = torch.as_tensor(self.mask, dtype=torch.bool)
        if mask.ndim == 3:
            mask = mask.reshape(mask.shape[0], -1)
        if mask.ndim != 2:
            raise ValueError(
                f"Term {self.name!r} mask must be (T, H*W) or (T, H, W); "
                f"got {tuple(mask.shape)}"
            )
        self.mask = mask

    def evaluate(self, x: Tensor) -> Tensor:
        """Return the unsampled ``(..., T, H*W)`` field for state ``x``."""
        out = None
        for channel, weight in self.weights.items():
            field_c = x[..., channel, :, :]
            if channel in self.transforms:
                field_c = self.transforms[channel](field_c)
            part = weight * field_c
            out = part if out is None else out + part
        if not (isinstance(self.offset, float) and self.offset == 0.0):
            out = out + self.offset
        return out

    def to(self, device) -> ObservationTerm:
        offset = self.offset.to(device) if torch.is_tensor(self.offset) else self.offset
        return replace(self, mask=self.mask.to(device), offset=offset)


class ObservationOperator:
    """Concatenate :class:`ObservationTerm` samples into one vector.

    Calling it maps ``(..., C, T, H*W)`` to ``(..., N)``; observations are
    ordered term by term, and within a term by frame, then pixel.
    """

    def __init__(self, terms: Sequence[ObservationTerm]) -> None:
        if not terms:
            raise ValueError("An observation operator needs at least one term")
        self.terms = list(terms)
        shapes = {tuple(term.mask.shape) for term in self.terms}
        if len(shapes) != 1:
            raise ValueError(f"Observation term masks differ in shape: {shapes}")

    def __call__(self, x: Tensor) -> Tensor:
        parts = [term.evaluate(x)[..., term.mask] for term in self.terms]
        return torch.cat(parts, dim=-1)

    @property
    def sizes(self) -> list[int]:
        return [int(term.mask.sum()) for term in self.terms]

    @property
    def names(self) -> list[str]:
        return [term.name for term in self.terms]

    @property
    def groups(self) -> Tensor:
        """``(N,)`` term index of every observation."""
        device = self.terms[0].mask.device
        return torch.cat(
            [
                torch.full((n,), k, dtype=torch.long, device=device)
                for k, n in enumerate(self.sizes)
            ]
        )

    def footprint(self, n_channels: int) -> Tensor:
        """``(C, T, H*W)`` mask of the pixels each channel contributes to."""
        mask = self.terms[0].mask
        out = torch.zeros(
            (n_channels, *mask.shape), dtype=torch.bool, device=mask.device
        )
        for term in self.terms:
            for channel in term.weights:
                out[channel] |= term.mask
        return out

    def to(self, device) -> ObservationOperator:
        return ObservationOperator([term.to(device) for term in self.terms])


@dataclass
class ObservationSpec:
    """Everything inference needs about one set of observations.

    Attributes:
        y: Observed values, shape ``(N,)``.
        cov_y: Observation noise variances, shape ``(N,)``, in the units of
            ``y``.
        operator: The forward operator that produced ``y``.
        H, W: Spatial size, for unflattening.
        truth: Optional ``(C, T, H*W)`` truth in normalized units (OSSE only).
        noise: Optional ``(N,)`` synthetic noise that was added to ``y``.
    """

    y: Tensor
    cov_y: Tensor
    operator: ObservationOperator
    H: int
    W: int
    truth: Optional[Tensor] = None
    noise: Optional[Tensor] = None
    extras: dict = field(default_factory=dict)

    @property
    def mask(self) -> Tensor:
        """``(C, T, H*W)`` footprint; ``C`` comes from ``truth`` or the terms."""
        if self.truth is not None:
            n_channels = self.truth.shape[0]
        else:
            n_channels = 1 + max(c for t in self.operator.terms for c in t.weights)
        return self.operator.footprint(n_channels)

    def to(self, device) -> ObservationSpec:
        def move(x):
            return None if x is None else x.to(device)

        return replace(
            self,
            y=self.y.to(device),
            cov_y=self.cov_y.to(device),
            operator=self.operator.to(device),
            truth=move(self.truth),
            noise=move(self.noise),
        )


def _per_term(value, terms: Sequence[ObservationTerm], what: str) -> list:
    if isinstance(value, Mapping):
        missing = [t.name for t in terms if t.name not in value]
        if missing:
            raise ValueError(f"{what} is missing terms {missing}")
        return [value[t.name] for t in terms]
    if isinstance(value, (int, float)) or (torch.is_tensor(value) and value.ndim == 0):
        return [float(value)] * len(terms)
    value = list(value)
    if len(value) != len(terms):
        raise ValueError(f"{what} has {len(value)} entries for {len(terms)} terms")
    return value


def _per_obs(value, operator: ObservationOperator, device, what: str) -> Tensor:
    """Expand a per-term standard deviation to one value per observation."""
    values = _per_term(value, operator.terms, what)
    return torch.cat(
        [
            torch.as_tensor(v, dtype=torch.float32, device=device).expand(n)
            for v, n in zip(values, operator.sizes)
        ]
    )


def build_observation(
    terms: Sequence[ObservationTerm],
    sigma_y,
    *,
    truth: Optional[Tensor] = None,
    y: Optional[Tensor] = None,
    grid: Optional[tuple[int, int]] = None,
    instrument_noise=False,
    seed: Optional[int] = None,
) -> ObservationSpec:
    """Build observations from terms, synthesizing ``y`` from ``truth`` if needed.

    Args:
        terms: The observed quantities.
        sigma_y: Observation-error standard deviation the sampler assumes, per
            term and in that term's units (``cov_y = sigma_y²``): a scalar, a
            sequence in term order, or ``{name: value}``.
        truth: ``(C, T, H, W)`` or ``(C, T, H*W)`` truth in normalized units.
            Required unless ``y`` is given.
        y: Real observations, ``(N,)`` in operator order. When omitted,
            ``y = A(truth)`` plus any instrument noise.
        grid: ``(H, W)``; needed only without a 4-D ``truth`` on a
            non-square grid.
        instrument_noise: White Gaussian noise added to synthesized ``y``.
            ``False`` (default) adds none; ``True`` uses ``sigma_y``; a
            scalar, sequence or ``{name: value}`` sets its own standard
            deviation per term, which may differ from the assumed
            ``sigma_y``. Ignored when ``y`` is given.
        seed: Seed for the noise draw.
    """
    operator = ObservationOperator(terms)
    T, HW = operator.terms[0].mask.shape
    H, W = grid if grid is not None else (None, None)
    if truth is not None:
        if truth.ndim == 4:
            H, W = truth.shape[-2:]
            truth = truth.reshape(*truth.shape[:2], -1)
        if truth.ndim != 3 or tuple(truth.shape[1:]) != (T, HW):
            raise ValueError(
                f"truth must be (C, {T}, H, W) or (C, {T}, {HW}); "
                f"got {tuple(truth.shape)}"
            )
        operator = operator.to(truth.device)
    if H is None:
        H = W = int(round(HW**0.5))
        if H * W != HW:
            raise ValueError("Pass grid=(H, W) or a (C, T, H, W) truth")

    device = operator.terms[0].mask.device
    std = _per_obs(sigma_y, operator, device, "sigma_y")
    noise = None
    if y is None:
        if truth is None:
            raise ValueError("Pass truth to synthesize observations, or pass y")
        y = operator(truth)
        if instrument_noise is not False:
            noise_std = (
                std
                if instrument_noise is True
                else _per_obs(instrument_noise, operator, device, "instrument_noise")
            )
            gen = torch.Generator(device="cpu")
            if seed is not None:
                gen.manual_seed(int(seed))
            else:
                gen.seed()
            noise = torch.randn(y.shape, generator=gen).to(device, y.dtype) * noise_std
            y = y + noise
    elif y.shape != (int(std.numel()),):
        raise ValueError(f"y must have shape ({std.numel()},); got {tuple(y.shape)}")

    return ObservationSpec(
        y=y,
        cov_y=std.to(y.dtype) ** 2,
        operator=operator,
        H=int(H),
        W=int(W),
        truth=truth,
        noise=noise,
    )


def channel_terms(
    mask: Tensor, channel_names: Optional[Sequence[str]] = None
) -> list[ObservationTerm]:
    """One unit-weight term per channel with any observed pixel.

    Args:
        mask: Boolean ``(C, T, H, W)`` or ``(C, T, H*W)`` mask.
        channel_names: Names for the terms; defaults to ``ch0``, ``ch1``, ...
    """
    mask = torch.as_tensor(mask, dtype=torch.bool)
    names = channel_names or [f"ch{c}" for c in range(mask.shape[0])]
    return [
        ObservationTerm(names[c], mask[c], {c: 1.0})
        for c in range(mask.shape[0])
        if bool(mask[c].any())
    ]


def build_osse_observation(
    truth: Tensor,
    mask: Tensor,
    sigma_y,
    seed: Optional[int] = None,
    channel_names: Optional[Sequence[str]] = None,
) -> tuple[ObservationSpec, ObservationOperator]:
    """Per-channel masked observations of a truth tile (normalized units).

    The original ``ocean_inference`` entry point, kept for the notebooks. Unlike
    :func:`build_observation`, it always adds noise of std ``sigma_y``, as it
    always has.

    Args:
        truth: ``(C, T, H, W)`` truth in normalized units.
        mask: Boolean ``(C, T, H, W)`` observation mask.
        sigma_y: Noise std per channel in normalized units: a scalar or a
            length-``C`` sequence.
        seed: Seed for the noise draw.
        channel_names: Optional channel labels for diagnostics.

    Returns:
        ``(spec, A)`` with ``A = spec.operator``.
    """
    if truth.ndim != 4 or truth.shape != mask.shape:
        raise ValueError(
            f"truth and mask must both be (C, T, H, W); got {tuple(truth.shape)} "
            f"and {tuple(mask.shape)}"
        )
    C = truth.shape[0]
    terms = channel_terms(mask.to(truth.device), channel_names)
    if isinstance(sigma_y, (int, float)) or (
        torch.is_tensor(sigma_y) and sigma_y.ndim == 0
    ):
        sigmas = [float(sigma_y)] * C
    else:
        sigmas = [float(s) for s in sigma_y]
        if len(sigmas) != C:
            raise ValueError(f"sigma_y must be a scalar or length-{C} sequence")
    sigma_by_term = {term.name: sigmas[next(iter(term.weights))] for term in terms}
    spec = build_observation(
        terms, sigma_by_term, truth=truth, instrument_noise=True, seed=seed
    )
    return spec, spec.operator


def build_random_mask(
    C: int,
    T: int,
    H: int,
    W: int,
    coverage: float,
    *,
    seed: Optional[int] = None,
    device=None,
) -> Tensor:
    """Bernoulli mask with the given observation coverage, for smoke tests."""
    if not 0.0 < coverage <= 1.0:
        raise ValueError(f"coverage must be in (0, 1]; got {coverage}")
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(int(seed))
    mask = torch.rand((C, T, H, W), generator=gen) < coverage
    return mask.to(device) if device is not None else mask
