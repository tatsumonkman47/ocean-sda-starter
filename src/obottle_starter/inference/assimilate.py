# SPDX-License-Identifier: Apache-2.0
"""One-call SDA on a window: physical-unit inputs, physical-unit ensemble.

:func:`assimilate` converts the truth or observations and their errors to
model units with a :class:`~obottle_starter.core.normalization.Normalizer`
(by default the checkpoint's own normalization), runs MMPS-guided DDIM in
model units, and returns the ensemble in physical units. The output records
the normalization used. See :mod:`obottle_starter.core.normalization` for why
the solve is done in model units.

Routes for observations (``guidance``):

=========================  ============================  ======================
route                      network input                 guidance
=========================  ============================  ======================
prior + SDA                frame times only              ``mmps`` / ``rozet``
learned conditional (M_3)  frame times and observations  ``conditional``
CFG (M_4)                  conditional and null branch   ``cfg``
hybrid (unvalidated)       frame times and observations  ``mmps`` / ``rozet``
=========================  ============================  ======================

**Longer sequences.** Pass more frame times than the model's window
(``cfg.time_length``) and the prior + SDA route assimilates the whole
sequence at once with sliding windows (Rozet & Louppe, 2023; see
:mod:`obottle_starter.inference.sda.windows`). ``window_step`` sets how far
apart consecutive windows start: 1 (default) is Rozet & Louppe's
construction, ``cfg.time_length`` splits the sequence into independent
blocks.

The hybrid route (a conditional model with ``mmps`` or ``rozet``) uses the
same observations twice. A model that already approximates ``p(x | y)`` is
not automatically improved by applying the likelihood of ``y`` again; it
can over-concentrate the ensemble. It is available for experiments and is
reported as such.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
import xarray as xr

from obottle_starter.core.conditioning import frame_conditioning
from obottle_starter.core.config import channel_names, trained_null_probability
from obottle_starter.core.normalization import Normalizer
from obottle_starter.core.obs_conditioning import (
    build_obs_condition,
    is_conditional,
    null_obs_condition,
    observed_names,
    prepend,
)
from obottle_starter.inference.sda.defaults import DEFAULTS
from obottle_starter.inference.sda.inference import run_inference
from obottle_starter.inference.sda.observations import (
    ObservationTerm,
    build_observation,
)
from obottle_starter.inference.sda.windows import sliding_windows


def assimilate(
    net: torch.nn.Module,
    cfg,
    *,
    times,
    center_lon: float,
    masks: Mapping[str, np.ndarray] | None = None,
    truth: Mapping[str, np.ndarray] | None = None,
    y: Mapping[str, np.ndarray] | None = None,
    sigma_y: Mapping[str, float] | float | None = None,
    instrument_noise=True,
    normalizer: Normalizer | None = None,
    obs_seed: int | None = 0,
    guidance: str = "mmps",
    window_step: int = DEFAULTS.window_step,
    **sampler_kwargs,
) -> xr.Dataset:
    """Sample the posterior for a window of ``cfg.time_length`` frames or longer.

    Args:
        net, cfg: From :func:`obottle_starter.core.model.load_checkpoint`.
        times: ``(T,)`` frame times. ``T`` is the model's ``cfg.time_length``,
            or more for a longer sequence assimilated with sliding windows
            (unconditional priors; ``masks``, ``truth`` and ``y`` then have
            ``T`` frames too).
        center_lon: Window-centre longitude, degrees east.
        masks: ``{channel: (T, H, W) bool}`` observed pixels. Channels left
            out, or ``None`` / all-false masks, are unobserved. With no
            observations at all, the prior is sampled and ``truth`` / ``y``
            are optional.
        truth: ``{channel: (T, H, W)}`` truth in physical units, for an
            OSSE: observations are taken from it on ``masks``.
        y: ``{channel: (T, H, W)}`` real observations in physical units
            instead of ``truth`` (only values on ``masks`` are used).
        sigma_y: Observation-error std assumed by the guidance, physical units
            (m for SSH, °C for SST): one value or ``{channel: value}``.
            Required when anything is observed (there is no sensible default
            across variables).
        instrument_noise: Noise added to OSSE observations: ``False``,
            ``True`` (std ``sigma_y``), or its own std per channel.
        normalizer: Defaults to ``Normalizer.from_config(cfg)``.
        obs_seed: Seed for the instrument noise.
        guidance: How observations steer sampling. ``"mmps"`` (default) or
            ``"rozet"`` (Rozet & Louppe 2023; set ``rozet_gamma``) work for
            any model. For conditional models only: ``"conditional"`` (the
            observations enter only as network inputs; M_3) or ``"cfg"``
            (classifier-free guidance with ``cfg_weight``; M_4, needs a model
            trained with ``null_probability > 0``). With a conditional model,
            ``mmps`` and ``rozet`` use both routes (unvalidated; see the
            module docstring). Ignored when nothing is observed.
        window_step: Sequences longer than the model's window only: frames
            between consecutive window starts, from 1 (Rozet & Louppe's
            sliding windows) to ``cfg.time_length`` (independent blocks).
            ``(T - cfg.time_length)`` must be a multiple of it.
        **sampler_kwargs: Passed to
            :func:`obottle_starter.inference.sda.inference.run_inference`
            (``n_members``, ``n_batch``, ``steps``, ``mmps_iterations``,
            ``rozet_gamma``, ``window_batch``, ``seed``, ...). Defaults are
            :data:`obottle_starter.inference.sda.defaults.DEFAULTS`, the same
            as ``configs/sda.yaml``.

    Returns:
        Dataset on ``(sample, channel, time, row, col)`` with ``samples``,
        ``mask``, ``observations`` (the values the guidance used, NaN where
        unobserved) and, for an OSSE, ``truth``, all in physical units;
        ``time`` holds the frame times.
    """
    names = channel_names(cfg)
    normalizer = normalizer or Normalizer.from_config(cfg)
    H, W = cfg.H, cfg.W
    times = np.asarray(times, dtype="datetime64[ns]")
    T = len(times)
    windowed = T != cfg.time_length
    masks = {c: m for c, m in (masks or {}).items() if m is not None}
    unknown = set(masks) - set(names)
    if unknown:
        raise ValueError(f"Model has channels {names}; cannot observe {unknown}")

    truth_norm = None
    if truth is not None:
        truth_norm = torch.from_numpy(
            np.stack([normalizer.normalize(c, truth[c], times) for c in names])
        )

    # Observations, sigma_y and instrument noise all go to model units.
    terms = normalizer.observation_terms(masks)
    observed = [t.name for t in terms]
    conditional = is_conditional(cfg)
    if windowed and conditional:
        raise ValueError(
            f"the model takes {cfg.time_length} frames; got {T}. Sliding windows "
            "over longer sequences are implemented for unconditional priors only"
        )
    if guidance not in ("mmps", "rozet", "conditional", "cfg"):
        raise ValueError(
            "guidance must be 'mmps', 'rozet', 'conditional' or 'cfg'; "
            f"got {guidance!r}"
        )
    if guidance in ("conditional", "cfg"):
        if not conditional:
            raise ValueError(f"guidance={guidance!r} needs a conditional model")
        if guidance == "cfg" and trained_null_probability(cfg) <= 0:
            raise ValueError(
                "guidance='cfg' needs a model trained with "
                "conditioning.training_observations.null_probability > 0; this "
                "one never saw the null condition, so CFG's unconditional "
                "branch is untrained"
            )
        ignored = set(observed) - set(observed_names(cfg))
        if ignored:
            print(
                f"WARNING: the model takes no {sorted(ignored)} inputs; "
                "those observations are ignored"
            )
    elif conditional and observed and _verbose(sampler_kwargs):
        print(
            f"NOTE: guidance={guidance!r} with a conditional model uses the "
            "observations twice (network inputs and guidance); this hybrid is "
            "unvalidated"
        )
    method = guidance if observed else "unconditional"
    if observed and sigma_y is None:
        raise ValueError("pass sigma_y (observation-error std, physical units)")
    if not isinstance(sigma_y, Mapping):
        sigma_y = {c: float(sigma_y) for c in observed}
    missing = [c for c in observed if c not in sigma_y]
    if missing:
        raise ValueError(f"sigma_y has no value for observed {missing}")
    bad = [c for c in observed if not 0 < float(sigma_y[c]) < float("inf")]
    if bad:
        raise ValueError(f"sigma_y must be finite and positive for {bad}")
    sigma = {c: normalizer.sigma_to_model(c, sigma_y[c]) for c in observed}
    if not terms:
        # The sampler still wants an operator; give it one with no rows.
        terms = [ObservationTerm(names[0], torch.zeros(T, H * W, dtype=bool), {0: 1.0})]
        sigma = {names[0]: 1.0}

    if not observed:
        obs = build_observation(terms, sigma, y=torch.zeros(0), grid=(H, W))
        obs.truth = truth_norm
    elif y is not None:
        parts = []
        for t in terms:
            values = normalizer.normalize(t.name, y[t.name], times)
            parts.append(torch.from_numpy(values).reshape(T, -1)[t.mask])
        obs = build_observation(terms, sigma, y=torch.cat(parts), grid=(H, W))
        obs.truth = truth_norm
    else:
        if truth_norm is None:
            raise ValueError("Pass truth (OSSE) or y (real observations)")
        noise = instrument_noise
        if noise is not True and noise is not False:
            if not isinstance(noise, Mapping):
                noise = {c: float(noise) for c in observed}
            noise = {c: normalizer.sigma_to_model(c, noise[c]) for c in observed}
        obs = build_observation(
            terms, sigma, truth=truth_norm, instrument_noise=noise, seed=obs_seed
        )

    device = next(net.parameters()).device
    if windowed:
        # One conditioning per window, from its own frames.
        cond = {"windows": sliding_windows(times, center_lon, cfg, window_step, device)}
    else:
        cond = frame_conditioning(times, center_lon, cfg, device)
    # Observed values as model-unit fields (NaN where unobserved): recorded in
    # the output, and the network's inputs for a conditional model.
    obs_field = torch.full((len(names), T, H * W), float("nan"))
    if observed:
        for t, part in zip(terms, torch.split(obs.y.cpu(), obs.operator.sizes)):
            obs_field[names.index(t.name)][t.mask] = part.to(obs_field.dtype)
    if conditional:
        values = obs_field.to(device)
        obs_masks = {t.name: t.mask.to(device) for t in terms if t.name in observed}
        time_condition = cond["condition"]
        cond["condition"] = prepend(
            build_obs_condition(cfg, values, obs_masks), time_condition
        )
        if method == "cfg":
            sampler_kwargs["null_condition"] = prepend(
                null_obs_condition(cfg, values), time_condition
            )
    ds = run_inference(
        net,
        obs,
        method=method,
        sigma_min=float(cfg.sigma_min),
        sigma_max=float(cfg.sigma_max),
        channel_names=names,
        **cond,
        **sampler_kwargs,
    )

    dims = ("channel", "time", "row", "col")
    out = xr.Dataset(
        {
            "samples": (
                ("sample", *dims),
                normalizer.to_physical(ds.x_sample_members.values, times),
            ),
            "mask": (dims, ds.mask_in.values),
            "observations": (
                dims,
                normalizer.to_physical(obs_field.reshape(-1, T, H, W).numpy(), times),
            ),
        },
        coords={"sample": ds.sample.values, "channel": names, "time": times},
    )
    if truth_norm is not None:
        out["truth"] = (dims, normalizer.to_physical(ds.x_true.values, times))
    for key in [k for k in ds.data_vars if k.startswith("solver_")]:
        out[key] = ds[key]
    out.attrs["units"] = ",".join(f"{c}={normalizer.units(c)}" for c in names)
    out.attrs.update(ds.attrs)
    out.attrs.update(normalizer.as_attrs())
    out.attrs.update(
        method=method,
        observed=",".join(observed),
        sigma_y=",".join(f"{c}={sigma_y[c]}" for c in observed),
        route=_route(method, conditional),
        conditional_model=int(conditional),
        center_lon=float(center_lon),
    )
    return out


def _verbose(sampler_kwargs) -> bool:
    return bool(sampler_kwargs.get("verbose", True))


def _route(method: str, conditional: bool) -> str:
    if method == "unconditional":
        return "prior sample (nothing observed)"
    if method in ("conditional", "cfg"):
        return f"learned conditional ({method})"
    if conditional:
        return f"hybrid: conditional model + {method} guidance (unvalidated)"
    return f"prior + SDA ({method})"
