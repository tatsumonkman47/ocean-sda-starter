# SPDX-License-Identifier: Apache-2.0
"""Score-based data assimilation with Moment Matching Posterior Sampling.

* :class:`EDMPrecondDenoiser`: network -> azula-style denoiser.
* :class:`MMPSDenoiser`: MMPS guidance (CG / GMRES / MINRES).
* :class:`RozetDenoiser`: Rozet & Louppe (2023) fixed-variance guidance.
* :class:`CFGDenoiser`: classifier-free guidance for conditional models.
* :class:`ObservationTerm`, :func:`build_observation`: linear observations.
* :func:`build_sampler`: guided DDIM.
* :func:`sliding_windows`, :class:`WindowedDenoiser`: sequences longer than
  the model's window (Rozet & Louppe, 2023).
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
from obottle_starter.inference.sda.windows import (
    SlidingWindows,
    WindowedDenoiser,
    sliding_windows,
)

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
    "SlidingWindows",
    "WindowedDenoiser",
    "build_observation",
    "build_random_mask",
    "build_sampler",
    "run_inference",
    "sliding_windows",
]
