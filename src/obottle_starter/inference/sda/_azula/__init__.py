# SPDX-License-Identifier: MIT
"""Minimal vendored subset of the azula diffusion library.

Source: https://github.com/probabilists/azula, version 0.10.2
(commit 9d3edb3493083eedbfdf6d56070568b6e3b96e15).
Copyright (c) 2024 The Probabilists. MIT License; see ``LICENSE`` here.

Only the pieces ``obottle_starter.inference.sda`` needs are kept, copied verbatim apart from
dropping unused classes and inlining two small helpers:

* ``noise``: ``Schedule``, ``VESchedule``
* ``denoise``: ``Posterior``, ``DiracPosterior``, ``Denoiser``
* ``linalg``: ``cg``, ``gmres``
* ``sample``: ``Sampler``, ``DDIMSampler``

The MMPS guidance in ``obottle_starter.inference.sda.mmps`` is derived from
``azula.guidance.mmps.MMPSDenoiser`` and credits it there.
"""
