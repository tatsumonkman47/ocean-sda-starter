# SPDX-License-Identifier: Apache-2.0
"""Ensemble posterior sampling packaged as an ``xr.Dataset``."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import xarray as xr
from torch import Tensor

from obottle_starter.inference.sda._azula.noise import VESchedule
from obottle_starter.inference.sda.cfg import CFGDenoiser
from obottle_starter.inference.sda.defaults import DEFAULTS
from obottle_starter.inference.sda.denoiser import EDMPrecondDenoiser
from obottle_starter.inference.sda.observations import ObservationSpec
from obottle_starter.inference.sda.operators import as_covariance
from obottle_starter.inference.sda.sampling import build_sampler
from obottle_starter.inference.sda.windows import SlidingWindows, WindowedDenoiser


def run_inference(
    net: torch.nn.Module,
    obs_spec: ObservationSpec,
    *,
    condition: Optional[Tensor] = None,
    class_labels: Optional[Tensor] = None,
    second_of_day: Optional[Tensor] = None,
    day_of_year: Optional[Tensor] = None,
    time_offsets: Optional[Tensor] = None,
    sigma_min: float = 0.0002,
    sigma_max: float = 200.0,
    method: str = "mmps",
    n_members: int = DEFAULTS.n_members,
    n_batch: int = DEFAULTS.n_batch,
    steps: int = DEFAULTS.steps,
    eta: float = 0.0,
    start: float = 1.0,
    stop: float = 0.0,
    timestep_spacing: str = DEFAULTS.timestep_spacing,
    rho: float = 7.0,
    mmps_solver: str = DEFAULTS.mmps_solver,
    mmps_iterations: int = DEFAULTS.mmps_iterations,
    mmps_warm_start: bool = False,
    mmps_preconditioner: str = "none",
    cov_y_inflation: float = 0.0,
    record_diagnostics: bool = False,
    rozet_gamma: float = DEFAULTS.rozet_gamma,
    null_condition: Optional[Tensor] = None,
    cfg_weight: float = DEFAULTS.cfg_weight,
    windows: Optional[SlidingWindows] = None,
    window_batch: Optional[int] = None,
    channel_names: Optional[Sequence[str]] = None,
    time_slice: Optional[Sequence[int]] = None,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> xr.Dataset:
    """Run posterior sampling and package the result as an ``xr.Dataset``.

    Args:
        net: Network from :func:`obottle_starter.core.model.load_checkpoint`.
        obs_spec: Observations, from :func:`obottle_starter.inference.sda.observations.
            build_observation` or ``build_osse_observation``.
        condition, class_labels, second_of_day, day_of_year, time_offsets:
            Per-tile conditioning, without a batch dimension, as returned by
            :func:`obottle_starter.core.conditioning.frame_conditioning`.
        sigma_min, sigma_max: Schedule bounds; use the checkpoint's
            ``cfg.sigma_min`` and ``cfg.sigma_max``.
        method: ``"mmps"``, ``"rozet"``, ``"unconditional"``, or, for a
            conditional model whose ``condition`` already contains the
            observations, ``"conditional"`` (no guidance; M_3) or ``"cfg"``
            (classifier-free guidance; M_4).
        n_members, n_batch: Ensemble size and members per batch.
        steps, eta, start, stop, timestep_spacing, rho: Sampler settings; see
            :func:`obottle_starter.inference.sda.sampling.build_sampler`.
        mmps_solver: ``"cg"``, ``"gmres"`` or ``"minres"``.
        mmps_iterations, mmps_warm_start, mmps_preconditioner,
        cov_y_inflation, record_diagnostics: MMPS settings; see
            :class:`obottle_starter.inference.sda.mmps.MMPSDenoiser`.
            ``record_diagnostics`` also applies to ``"rozet"``.
        rozet_gamma: ``γ`` for ``method="rozet"``; see
            :class:`obottle_starter.inference.sda.rozet.RozetDenoiser`.
        null_condition, cfg_weight: For ``method="cfg"``: the condition with
            the observation part zeroed, and the guidance weight ``w``; see
            :class:`obottle_starter.inference.sda.cfg.CFGDenoiser`.
        windows: For a sequence longer than the model's window: the layout
            and per-window conditioning from
            :func:`obottle_starter.inference.sda.windows.sliding_windows`,
            used instead of the per-tile conditioning above. Unconditional
            priors only.
        window_batch: Most windows per network call (sliding windows only);
            see :class:`obottle_starter.inference.sda.windows.WindowedDenoiser`.
        channel_names: Labels for the ``channel`` coordinate.
        time_slice: Optional global frame indices, attached as ``time_i``.
        seed: Seed for the initial noise.
        verbose: Print a summary and show progress bars.

    Returns:
        Dataset with ``x_sample_members`` ``(sample, channel, time, x, y)``,
        ``mask_in`` (pixels each channel contributes to an observation),
        observation-space ``y_obs``, ``obs_term`` and ``cov_y`` on ``obs``,
        and, for an OSSE, ``x_true``, ``x_masked`` (truth on ``mask_in``) and
        ``y_noise``. With ``record_diagnostics``, ``solver_*`` variables on
        ``(member_batch, step)``.
    """
    if n_batch <= 0 or n_members <= 0:
        raise ValueError(
            f"n_members and n_batch must be positive; got {n_members}, {n_batch}"
        )
    device = next(net.parameters()).device
    obs = obs_spec.to(device)
    H, W = obs.H, obs.W
    if hasattr(obs.operator, "state_shape"):  # general operators (operators.py)
        _, T, H, W = obs.operator.state_shape
        HW = H * W
    else:
        T, HW = obs.operator.terms[0].mask.shape
    if HW != H * W:
        raise ValueError(f"Observation masks have {HW} pixels; expected {H * W}")
    C = (
        obs.truth.shape[0]
        if obs.truth is not None
        else int(getattr(net, "img_channels", 0)) or obs.mask.shape[0]
    )
    channel_names = list(channel_names or [f"ch{i}" for i in range(C)])
    if len(channel_names) != C:
        raise ValueError(f"channel_names length {len(channel_names)} != C={C}")

    schedule = VESchedule(sigma_min=sigma_min, sigma_max=sigma_max)
    if windows is not None:
        if method in ("conditional", "cfg"):
            raise ValueError(
                "sliding windows are implemented for unconditional priors; "
                f"method={method!r} needs a conditional model"
            )
        if windows.n_frames != T:
            raise ValueError(
                f"the windows cover {windows.n_frames} frames; the observations {T}"
            )
        denoiser = WindowedDenoiser(net, schedule, windows, window_batch).to(device)
    else:
        denoiser = EDMPrecondDenoiser(
            net,
            schedule,
            condition=condition,
            class_labels=class_labels,
            second_of_day=second_of_day,
            day_of_year=day_of_year,
            time_offsets=time_offsets,
        ).to(device)
    sigma_data = float(getattr(net, "sigma_data", 1.0))
    sampler_method = method
    if method in ("conditional", "cfg"):
        # Observations enter only through the network's condition input.
        sampler_method = "unconditional"
        if method == "cfg":
            if null_condition is None:
                raise ValueError("method='cfg' needs null_condition")
            null = EDMPrecondDenoiser(
                net,
                schedule,
                condition=null_condition,
                class_labels=class_labels,
                second_of_day=second_of_day,
                day_of_year=day_of_year,
                time_offsets=time_offsets,
            ).to(device)
            denoiser = CFGDenoiser(denoiser, null, cfg_weight)

    if verbose:
        sizes = ", ".join(
            f"{name}={n}" for name, n in zip(obs.operator.names, obs.operator.sizes)
        )
        print("─" * 60)
        print(
            f"run_inference: method={method}, n_members={n_members}, n_batch={n_batch}"
        )
        print(f"  C={C}, T={T}, H={H}, W={W} | observations: {sizes}")
        if windows is not None:
            print(f"  {windows.describe()}")
        print(
            f"  sigma range: [{sigma_min}, {sigma_max}] | steps={steps}, eta={eta}, "
            f"spacing={timestep_spacing}"
        )
        if method == "mmps":
            print(
                f"  mmps: solver={mmps_solver}, iterations={mmps_iterations}, "
                f"warm_start={mmps_warm_start}, preconditioner={mmps_preconditioner}, "
                f"cov_y_inflation={cov_y_inflation}"
            )
        if method == "cfg":
            print(f"  cfg: weight={cfg_weight} (1 = plain conditional model)")
        if method == "rozet":
            print(
                f"  rozet: gamma={rozet_gamma} (likelihood var = sigma_y^2 + gamma*sigma_t^2)"
            )
        print("─" * 60)

    if seed is not None:
        torch.manual_seed(int(seed))

    collected: list[Tensor] = []
    diagnostics: list[list[dict]] = []
    remaining = n_members
    while remaining > 0:
        B = min(n_batch, remaining)
        # Rebuilt per batch so warm-start state never crosses batches.
        sampler = build_sampler(
            sampler_method,
            denoiser,
            obs,
            steps=steps,
            eta=eta,
            start=start,
            stop=stop,
            timestep_spacing=timestep_spacing,
            rho=rho,
            silent=not verbose,
            mmps_solver=mmps_solver,
            mmps_iterations=mmps_iterations,
            mmps_warm_start=mmps_warm_start,
            mmps_preconditioner=mmps_preconditioner,
            cov_y_inflation=cov_y_inflation,
            record_diagnostics=record_diagnostics,
            sigma_data=sigma_data,
            rozet_gamma=rozet_gamma,
        )
        # VE initialization: x_T ~ N(0, sigma_data² + sigma_T²).
        x_T = sampler.init(
            shape=(B, C, T, HW),
            mean=0.0,
            var=sigma_data**2,
            dtype=torch.float32,
            device=device,
        )
        collected.append(sampler(x_T).detach().cpu())
        if record_diagnostics and method in ("mmps", "rozet"):
            diagnostics.append(sampler.denoiser.diagnostics)
        remaining -= B

    samples = torch.cat(collected, dim=0).reshape(n_members, C, T, H, W)
    mask_hw = obs.operator.footprint(C).reshape(C, T, H, W).cpu()

    coords = {
        "sample": np.arange(n_members),
        "channel": channel_names,
        "time": np.arange(T),
        "x": np.arange(H),
        "y": np.arange(W),
        "obs": np.arange(obs.y.numel()),
        "term": obs.operator.names,
    }
    dims = ["channel", "time", "x", "y"]
    data = {
        "x_sample_members": (["sample", *dims], samples.numpy()),
        "mask_in": (dims, mask_hw.numpy()),
        "y_obs": (["obs"], obs.y.cpu().numpy()),
        "cov_y": (["obs"], as_covariance(obs.cov_y).diagonal().cpu().numpy()),
        "obs_term": (["obs"], obs.operator.groups.cpu().numpy()),
    }
    if obs.truth is not None:
        truth_hw = obs.truth.reshape(C, T, H, W).cpu()
        data["x_true"] = (dims, truth_hw.numpy())
        data["x_masked"] = (dims, (truth_hw * mask_hw.to(truth_hw.dtype)).numpy())
    if obs.noise is not None:
        data["y_noise"] = (["obs"], obs.noise.cpu().numpy())
    if diagnostics:
        steps_recorded = min(len(d) for d in diagnostics)
        for key in diagnostics[0][0]:
            data[f"solver_{key}"] = (
                ["member_batch", "step"],
                np.array(
                    [[entry[key] for entry in d[:steps_recorded]] for d in diagnostics]
                ),
            )
    ds = xr.Dataset(data, coords=coords)
    ds.attrs.update(method=method, timestep_spacing=timestep_spacing, steps=int(steps))
    if method == "mmps":
        ds.attrs.update(
            mmps_solver=mmps_solver,
            mmps_iterations=int(mmps_iterations),
            mmps_warm_start=int(mmps_warm_start),
            mmps_preconditioner=mmps_preconditioner,
            cov_y_inflation=float(cov_y_inflation),
        )
    elif method == "rozet":
        ds.attrs.update(rozet_gamma=float(rozet_gamma))
    elif method == "cfg":
        ds.attrs.update(cfg_weight=float(cfg_weight))
    if windows is not None:
        ds.attrs.update(
            window_length=int(windows.window),
            window_step=int(windows.step),
            n_windows=int(windows.n_windows),
        )
    if time_slice is not None and len(time_slice) == T:
        ds = ds.assign_coords(time_i=("time", np.asarray(time_slice)))
    return ds
