# SPDX-License-Identifier: Apache-2.0
"""Default sampler settings, shared by the Python API and ``configs/sda.yaml``.

:func:`obottle_starter.assimilate`, :func:`run_inference` and
:func:`build_sampler` take their defaults from :data:`DEFAULTS`, and a test
checks that ``configs/sda.yaml`` holds the same values, so switching between
the command line and Python never silently changes the numerical method.
The low-level guidance classes (``MMPSDenoiser``, ``RozetDenoiser``) keep
their own minimal defaults, documented there.

These are starting points, not tuned optima: check that results stop
changing when ``steps`` or ``mmps_iterations`` are increased.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SamplerDefaults:
    n_members: int = 8
    n_batch: int = 4
    steps: int = 64
    timestep_spacing: str = "log_sigma"
    mmps_solver: str = "gmres"
    mmps_iterations: int = 4
    rozet_gamma: float = 0.1
    cfg_weight: float = 1.5
    window_step: int = 1


DEFAULTS = SamplerDefaults()
