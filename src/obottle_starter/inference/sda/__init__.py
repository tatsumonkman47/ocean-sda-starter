# SPDX-License-Identifier: Apache-2.0
"""Score-based data assimilation with Moment Matching Posterior Sampling.

* :class:`EDMPrecondDenoiser`: network -> azula-style denoiser.
* :class:`MMPSDenoiser`: MMPS guidance (CG / GMRES / MINRES).
* :class:`RozetDenoiser`: Rozet & Louppe (2023) fixed-variance guidance.
* :class:`CFGDenoiser`: classifier-free guidance for conditional models.
* :class:`ObservationTerm`, :func:`build_observation`: linear observations.
* :func:`build_sampler`: guided DDIM.
* :func:`run_inference`: ensemble loop -> ``xr.Dataset`` (model units).
* :data:`DEFAULTS`: sampler defaults shared with ``configs/sda.yaml``.

``_azula`` is a vendored subset of azula (MIT).
"""

from obottle_starter.inference.sda.cfg import CFGDenoiser
from obottle_starter.inference.sda.defaults import DEFAULTS, SamplerDefaults
from obottle_starter.inference.sda.denoiser import EDMPrecondDenoiser
from obottle_starter.inference.sda.inference import run_inference
from obottle_starter.inference.sda.mmps import MMPSDenoiser
from obottle_starter.inference.sda.rozet import RozetDenoiser
from obottle_starter.inference.sda.observations import (
    ObservationOperator,
    ObservationSpec,
    ObservationTerm,
    build_observation,
    build_random_mask,
)
from obottle_starter.inference.sda.sampling import build_sampler

__all__ = [
    "DEFAULTS",
    "CFGDenoiser",
    "EDMPrecondDenoiser",
    "MMPSDenoiser",
    "ObservationOperator",
    "ObservationSpec",
    "ObservationTerm",
    "RozetDenoiser",
    "SamplerDefaults",
    "build_observation",
    "build_random_mask",
    "build_sampler",
    "run_inference",
]
