# SPDX-License-Identifier: Apache-2.0
"""Guided DDIM sampler factory.

``VESchedule`` maps ``t ∈ [0, 1]`` to ``σ_t = σ_min^(1−t) σ_max^t``, so
azula's evenly spaced ``t`` grid is evenly spaced in ``log σ``. That puts
more steps at high σ than the Karras ρ=7 grid used by EDM samplers.
``timestep_spacing="karras"`` uses the Karras grid instead, expressed as the
matching ``t`` values.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from obottle_starter.inference.sda._azula.sample import DDIMSampler
from obottle_starter.inference.sda.defaults import DEFAULTS
from obottle_starter.inference.sda.denoiser import EDMPrecondDenoiser
from obottle_starter.inference.sda.mmps import MMPSDenoiser
from obottle_starter.inference.sda.observations import ObservationSpec
from obottle_starter.inference.sda.rozet import RozetDenoiser

TIMESTEP_SPACINGS = ("log_sigma", "karras")


class GuidedDDIMSampler(DDIMSampler):
    """DDIM with a choice of timestep spacing.

    Args:
        denoiser: Denoiser with a ``VESchedule``-like ``schedule`` that has
            ``sigma_min`` and ``sigma_max``.
        timestep_spacing: ``"log_sigma"`` (azula's default) or ``"karras"``.
        rho: Karras grid exponent.
        kwargs: Passed to :class:`DDIMSampler` (``eta``, ``start``, ``stop``,
            ``steps``, ``silent``, ...).
    """

    def __init__(
        self,
        denoiser,
        *,
        timestep_spacing: str = DEFAULTS.timestep_spacing,
        rho: float = 7.0,
        **kwargs,
    ) -> None:
        super().__init__(denoiser, **kwargs)
        if timestep_spacing not in TIMESTEP_SPACINGS:
            raise ValueError(
                f"timestep_spacing must be one of {TIMESTEP_SPACINGS}; "
                f"got {timestep_spacing!r}"
            )
        self.timestep_spacing = timestep_spacing
        self.rho = float(rho)

    @property
    def timesteps(self) -> Tensor:
        if self.timestep_spacing == "log_sigma":
            return super().timesteps
        schedule = self.denoiser.schedule
        log_min = math.log(schedule.sigma_min)
        log_range = math.log(schedule.sigma_max) - log_min
        s_hi = math.exp(log_min + self.start * log_range)
        s_lo = math.exp(log_min + self.stop * log_range)
        ramp = torch.linspace(0, 1, self.steps + 1, dtype=torch.float64)
        inv_rho = 1.0 / self.rho
        sigma = (s_hi**inv_rho + ramp * (s_lo**inv_rho - s_hi**inv_rho)) ** self.rho
        t = (torch.log(sigma) - log_min) / log_range
        return t.to(dtype=self.dtype, device=self.device)


def build_sampler(
    method: str,
    denoiser: EDMPrecondDenoiser,
    obs: ObservationSpec,
    *,
    steps: int = DEFAULTS.steps,
    eta: float = 0.0,
    start: float = 1.0,
    stop: float = 0.0,
    timestep_spacing: str = DEFAULTS.timestep_spacing,
    rho: float = 7.0,
    silent: bool = False,
    mmps_solver: str = DEFAULTS.mmps_solver,
    mmps_iterations: int = DEFAULTS.mmps_iterations,
    mmps_warm_start: bool = False,
    mmps_preconditioner: str = "none",
    cov_y_inflation: float = 0.0,
    record_diagnostics: bool = False,
    sigma_data: float = 1.0,
    rozet_gamma: float = DEFAULTS.rozet_gamma,
) -> GuidedDDIMSampler:
    """Wrap ``denoiser`` in guidance and return a configured sampler.

    Args:
        method: ``"mmps"`` for MMPS guidance, ``"rozet"`` for Rozet & Louppe
            (2023)'s fixed-variance guidance, or ``"unconditional"`` to
            sample the prior with the same sampler (observations ignored).
        denoiser: The unconditional denoiser bridge.
        obs: Observations; ``obs.operator`` is the forward operator.
        steps, eta, start, stop: DDIM settings. ``eta=0`` is deterministic.
            ``start``/``stop`` bound ``t``; lower ``start`` to begin below
            the training ``sigma_max``.
        timestep_spacing, rho: See :class:`GuidedDDIMSampler`.
        silent: Hide the progress bar.
        mmps_solver: ``"cg"``, ``"gmres"`` or ``"minres"``.
        mmps_iterations: Krylov iterations per step.
        mmps_warm_start, mmps_preconditioner, cov_y_inflation,
        record_diagnostics: See :class:`obottle_starter.inference.sda.mmps.MMPSDenoiser`.
        sigma_data: Training ``sigma_data`` (diagonal preconditioner only).
        rozet_gamma: ``γ`` for ``method="rozet"``; see
            :class:`obottle_starter.inference.sda.rozet.RozetDenoiser`.

    Returns:
        A sampler; call it on an initial noisy tensor. With guidance, the
        guided denoiser is ``sampler.denoiser``, which holds any recorded
        diagnostics.
    """
    sampler_kwargs = dict(
        eta=eta,
        start=start,
        stop=stop,
        steps=steps,
        silent=silent,
        timestep_spacing=timestep_spacing,
        rho=rho,
    )
    if method == "unconditional":
        return GuidedDDIMSampler(denoiser, **sampler_kwargs)
    if method == "mmps":
        guided = MMPSDenoiser(
            denoiser,
            y=obs.y,
            A=obs.operator,
            cov_y=obs.cov_y,
            solver=mmps_solver,
            iterations=mmps_iterations,
            warm_start=mmps_warm_start,
            preconditioner=mmps_preconditioner,
            cov_y_inflation=cov_y_inflation,
            sigma_data=sigma_data,
            record_diagnostics=record_diagnostics,
            obs_groups=obs.operator.groups,
            group_names=obs.operator.names,
        )
        return GuidedDDIMSampler(guided, **sampler_kwargs)
    if method == "rozet":
        guided = RozetDenoiser(
            denoiser,
            y=obs.y,
            A=obs.operator,
            cov_y=obs.cov_y,
            gamma=rozet_gamma,
            record_diagnostics=record_diagnostics,
            obs_groups=obs.operator.groups,
            group_names=obs.operator.names,
        )
        return GuidedDDIMSampler(guided, **sampler_kwargs)
    raise ValueError(
        f"Unknown method={method!r}; expected 'mmps', 'rozet' or 'unconditional'"
    )
