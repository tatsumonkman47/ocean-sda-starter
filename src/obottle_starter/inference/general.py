# SPDX-License-Identifier: Apache-2.0
"""SDA with general observation operators, in physical units (opt-in).

The mask-based :func:`obottle_starter.assimilate` covers per-variable masks.
This module is for everything else that is linear: off-grid points, area or
footprint averages (coarse-to-fine), combinations of variables, time
averages, and correlated errors. The operators come from
:mod:`obottle_starter.inference.sda.operators` and are written in
*physical* units:

    y_phys = H u + offset + e,      e ~ N(0, R_phys),

with ``u`` the state in physical units (m, °C). The model works in
normalized units, ``u = S x + b(t)`` (per-variable scale ``S`` and offset
``b``, from the checkpoint's normalization). :func:`model_unit_observation`
rewrites the whole observation equation consistently, with a per-observation
scaling ``Q``:

    A_model(x) = Q (H S x + H b + offset),   y_model = Q y_phys,
    R_model = Q R_phys Qᵀ.

The exact posterior does not depend on ``Q``, but MMPS truncates its Krylov
solve, which reduces the largest residuals first. ``scaling`` chooses ``Q``:

* ``"error"`` (default): ``Q_i = 1 / sqrt(R_ii)``, every observation in units
  of its own error. Residuals of different products are then comparable,
  whatever their units or accuracy.
* ``"row_norm"``: ``Q_i = 1 / Σ_k |(H S)_ik|``, every row of ``A_model`` of
  unit L1 norm. For a point or an average of one variable that is ``1 / S``,
  what ``assimilate`` uses for masks. Fine for one product; with mixed
  products of very different accuracy it lets the less accurate one
  dominate a truncated solve (in examples/03, SST footprints with 0.15 °C
  errors were fitted to 5 error stds beside SSH tracks, against 1.1 with
  ``"error"``).
* a list with one array or scalar per part, to set ``Q`` yourself.

Only unconditional priors are supported here (observation inputs of
conditional models are mask-shaped).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Optional

import numpy as np
import torch
import xarray as xr

from obottle_starter.core.conditioning import frame_conditioning
from obottle_starter.core.config import channel_names
from obottle_starter.core.normalization import Normalizer
from obottle_starter.core.obs_conditioning import is_conditional
from obottle_starter.inference.sda.inference import run_inference
from obottle_starter.inference.sda.observations import ObservationSpec
from obottle_starter.inference.sda.operators import (
    ObservationCovariance,
    ObservationSet,
    SparseObservation,
    _as_set,
    as_covariance,
)

__all__ = ["assimilate_general", "model_unit_observation"]


def _truth_array(truth, names, shape) -> np.ndarray:
    C, T, H, W = shape
    if isinstance(truth, Mapping):
        truth = np.stack([np.asarray(truth[c], dtype=np.float32) for c in names])
    truth = np.asarray(truth, dtype=np.float32)
    if truth.shape != (C, T, H, W):
        raise ValueError(
            f"truth must be (C, T, H, W) = {(C, T, H, W)}; got {truth.shape}"
        )
    return truth


def _covariances(covariance, operator: ObservationSet) -> list[ObservationCovariance]:
    """One covariance per part of ``operator``."""
    if isinstance(covariance, (list, tuple)):
        covs = [as_covariance(c) for c in covariance]
    else:
        whole = as_covariance(covariance)
        if len(operator.parts) == 1:
            covs = [whole]
        else:
            # Split block-wise; blocks must not straddle parts.
            covs, blocks, sizes = [], list(whole.blocks), operator.sizes
            for n in sizes:
                taken, count = [], 0
                while count < n and blocks:
                    b = blocks.pop(0)
                    taken.append(b)
                    count += b.shape[0]
                if count != n:
                    raise ValueError(
                        "covariance blocks do not line up with the observation "
                        "parts; pass one covariance per part"
                    )
                covs.append(ObservationCovariance(taken))
    for cov, part in zip(covs, operator.parts):
        if cov.n != part.n_obs:
            raise ValueError(
                f"covariance for {part.name!r} has {cov.n} rows for {part.n_obs} "
                "observations"
            )
    if len(covs) != len(operator.parts):
        raise ValueError("pass one covariance per observation part")
    return covs


def _to_model_part(part: SparseObservation, scale, offset_table, q, cov):
    """The model-unit version of a physical-unit part, and its scaling ``Q``."""
    _, T, H, W = part.state_shape
    channel = part.index // (T * H * W)
    frame = (part.index // (H * W)) % T
    weight = part.weight * scale[channel]
    hb = (part.weight * offset_table[channel, frame]).sum(-1)
    norm = weight.abs().sum(-1)
    if bool((norm == 0).any()):
        raise ValueError(f"{part.name!r} has observations with no weight")
    if isinstance(q, str) and q == "error":
        q = 1.0 / cov.diagonal().double().sqrt().float()
    elif isinstance(q, str) and q == "row_norm":
        q = 1.0 / norm
    elif isinstance(q, str):
        raise ValueError(f"scaling must be 'error', 'row_norm' or a list; got {q!r}")
    q = torch.as_tensor(q, dtype=torch.float32).expand(part.n_obs)
    model = SparseObservation(
        part.name,
        part.state_shape,
        part.index,
        q[:, None] * weight,
        q * (part.offset + hb),
        part.coords,
    )
    return model, q


def model_unit_observation(
    operator: SparseObservation | ObservationSet,
    covariance,
    normalizer: Normalizer,
    times,
    *,
    truth=None,
    y=None,
    instrument_noise: bool | ObservationCovariance = False,
    seed: Optional[int] = None,
    scaling="error",
) -> ObservationSpec:
    """Physical-unit observations rewritten in model units (module docstring).

    Args:
        operator: Physical-unit observations of the ``(C, T, H, W)`` state.
        covariance: ``R_phys``: an
            :class:`~obottle_starter.inference.sda.operators.ObservationCovariance`,
            a variance tensor, or one per part.
        normalizer: The model's normalization.
        times: ``(T,)`` frame times (time-dependent offsets).
        truth: Physical truth, ``{name: (T, H, W)}`` or ``(C, T, H, W)``, for
            an OSSE; ``y`` is then synthesized from it.
        y: ``(N,)`` real physical observations instead.
        instrument_noise: Noise added to synthesized ``y`` (physical units):
            ``False``, ``True`` (drawn from ``covariance``) or a covariance.
        seed: Seed for the noise.
        scaling: ``Q``: ``"error"`` (default), ``"row_norm"``, or one
            array or scalar per part (module docstring).

    Returns:
        A model-unit ``ObservationSpec`` for ``run_inference``. Its
        ``extras`` hold ``scaling`` (``Q``), ``y_phys``, ``noise_phys``,
        ``operator_phys`` and ``covariance_phys``.
    """
    operator = _as_set(operator)
    C, T, H, W = operator.state_shape
    names = normalizer.channels
    if len(names) != C:
        raise ValueError(f"operator observes {C} variables; the model has {names}")
    covs = _covariances(covariance, operator)
    cov_phys = ObservationCovariance.concat(covs).validate("covariance")
    scale = torch.tensor([normalizer.scale(c) for c in range(C)], dtype=torch.float32)
    offsets = torch.from_numpy(
        np.stack([normalizer.offset(c, times) for c in range(C)]).astype(np.float32)
    )
    if not isinstance(scaling, str) and len(scaling) != len(operator.parts):
        raise ValueError("scaling needs one entry per observation part")
    parts, qs = [], []
    for k, part in enumerate(operator.parts):
        q_k = scaling if isinstance(scaling, str) else scaling[k]
        model, q = _to_model_part(part, scale, offsets, q_k, covs[k])
        parts.append(model)
        qs.append(q)
    q = torch.cat(qs)
    model_op = ObservationSet(parts)

    truth_model, noise = None, None
    if y is None:
        if truth is None:
            raise ValueError("Pass truth (OSSE) or y (real observations)")
        u = _truth_array(truth, names, (C, T, H, W))
        y_phys = operator(torch.from_numpy(u).reshape(C, T, H * W))
        if instrument_noise is not False:
            source = (
                cov_phys
                if instrument_noise is True
                else as_covariance(instrument_noise)
            )
            gen = torch.Generator()
            if seed is not None:
                gen.manual_seed(int(seed))
            else:
                gen.seed()
            noise = source.validate("instrument_noise").sample(gen).to(y_phys.dtype)
            y_phys = y_phys + noise
    else:
        y_phys = torch.as_tensor(np.asarray(y), dtype=torch.float32)
        if y_phys.shape != (operator.n_obs,):
            raise ValueError(f"y must have shape ({operator.n_obs},)")
    if not bool(torch.isfinite(y_phys).all()):
        raise ValueError(
            "observations must be finite (NaN or inf found); drop missing "
            "values from the operator instead of passing them"
        )
    if truth is not None:
        u = _truth_array(truth, names, (C, T, H, W))
        truth_model = torch.from_numpy(
            np.stack([normalizer.normalize(c, u[c], times) for c in range(C)])
        ).reshape(C, T, H * W)
    spec = ObservationSpec(
        y=q * y_phys,
        cov_y=cov_phys.scaled(q).to(dtype=torch.float32),
        operator=model_op,
        H=H,
        W=W,
        truth=truth_model,
        noise=None if noise is None else q * noise,
    )
    spec.extras.update(
        scaling=q,
        y_phys=y_phys,
        noise_phys=noise,
        operator_phys=operator,
        covariance_phys=cov_phys,
    )
    return spec


def assimilate_general(
    net: torch.nn.Module,
    cfg,
    operator: SparseObservation | ObservationSet,
    covariance,
    *,
    times,
    center_lon: float,
    truth=None,
    y=None,
    normalizer: Optional[Normalizer] = None,
    instrument_noise: bool | ObservationCovariance = False,
    obs_seed: Optional[int] = 0,
    scaling="error",
    guidance: str = "mmps",
    **sampler_kwargs,
) -> xr.Dataset:
    """Sample the posterior for one window given general linear observations.

    The general-operator counterpart of :func:`obottle_starter.assimilate`:
    physical units in and out, solved in model units.

    Args:
        net, cfg: From :func:`obottle_starter.core.model.load_checkpoint`
            (an unconditional prior).
        operator: Physical-unit observations
            (:mod:`obottle_starter.inference.sda.operators`).
        covariance: ``R_phys`` (see :func:`model_unit_observation`).
        times, center_lon: As for ``assimilate``.
        truth, y, instrument_noise, obs_seed, scaling: See
            :func:`model_unit_observation`.
        normalizer: Defaults to the checkpoint's.
        guidance: ``"mmps"`` (default) or ``"rozet"``. Note that Rozet's
            variance heuristic is added in observation space, which differs
            from MMPS's for averaging operators.
        **sampler_kwargs: Passed to ``run_inference`` (``n_members``,
            ``steps``, ``mmps_iterations``, ...; defaults as in ``sda.yaml``).

    Returns:
        Dataset with ``samples`` (and ``truth``) on
        ``(sample, channel, time, row, col)`` in physical units, and in
        observation space: ``y`` (physical observations), ``predicted``
        (``H u + offset`` for each member), ``error_std`` (``sqrt(diag
        R_phys)``), ``scaling`` (``Q``), the operator's coordinates as
        ``obs_*``, and ``misfit`` per member: the RMS of residuals whitened
        by ``R_phys`` (a diagnostic, not a target).
    """
    if is_conditional(cfg):
        raise ValueError(
            "assimilate_general supports unconditional priors only; a "
            "conditional model's observation inputs are mask-shaped"
        )
    if guidance not in ("mmps", "rozet"):
        raise ValueError(f"guidance must be 'mmps' or 'rozet'; got {guidance!r}")
    names = channel_names(cfg)
    normalizer = normalizer or Normalizer.from_config(cfg)
    operator = _as_set(operator)
    expected = (len(names), int(cfg.time_length), int(cfg.H), int(cfg.W))
    if operator.state_shape != expected:
        raise ValueError(
            f"operator observes a state of shape {operator.state_shape}; the "
            f"model's is {expected}"
        )
    times = np.asarray(times, dtype="datetime64[ns]")
    spec = model_unit_observation(
        operator,
        covariance,
        normalizer,
        times,
        truth=truth,
        y=y,
        instrument_noise=instrument_noise,
        seed=obs_seed,
        scaling=scaling,
    )
    device = next(net.parameters()).device
    cond = frame_conditioning(times, center_lon, cfg, device)
    ds = run_inference(
        net,
        spec,
        method=guidance,
        sigma_min=float(cfg.sigma_min),
        sigma_max=float(cfg.sigma_max),
        channel_names=names,
        **cond,
        **sampler_kwargs,
    )

    C, T, H, W = expected
    samples = normalizer.to_physical(ds.x_sample_members.values, times)
    flat = torch.from_numpy(samples).reshape(samples.shape[0], C, T, H * W)
    predicted = operator(flat)
    cov_phys = spec.extras["covariance_phys"]
    y_phys = spec.extras["y_phys"]
    misfit = cov_phys.whiten((predicted - y_phys).double()).square().mean(-1).sqrt()

    dims = ("channel", "time", "row", "col")
    data = {
        "samples": (("sample", *dims), samples),
        "y": (("obs",), y_phys.numpy()),
        "predicted": (("sample", "obs"), predicted.numpy()),
        "error_std": (("obs",), cov_phys.diagonal().double().sqrt().numpy()),
        "scaling": (("obs",), spec.extras["scaling"].numpy()),
        "obs_group": (("obs",), operator.groups.numpy()),
        "misfit": (("sample",), misfit.numpy()),
    }
    for key, values in operator.coords.items():
        data[f"obs_{key}"] = (("obs",), values)
    if truth is not None:
        u = _truth_array(truth, names, expected)
        data["truth"] = (dims, u)
        data["y_true"] = (
            ("obs",),
            operator(torch.from_numpy(u).reshape(C, T, H * W)).numpy(),
        )
    for key in [k for k in ds.data_vars if k.startswith("solver_")]:
        data[key] = ds[key]
    out = xr.Dataset(
        data,
        coords={"sample": ds.sample.values, "channel": names, "time": times},
    )
    out.attrs.update(ds.attrs)
    out.attrs.update(normalizer.as_attrs())
    out.attrs.update(
        units=",".join(f"{c}={normalizer.units(c)}" for c in names),
        groups=",".join(operator.names),
        route=f"prior + SDA ({guidance}, general operator)",
        correlated_errors=int(not cov_phys.is_diagonal),
        center_lon=float(center_lon),
    )
    return out
