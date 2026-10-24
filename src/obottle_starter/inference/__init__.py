# SPDX-License-Identifier: Apache-2.0
"""Using a prior: score-based data assimilation (SDA).

* ``assimilate``: one call from physical-unit observations to a
  physical-unit posterior ensemble for one window.
* ``osse``: the bundled OSSE test tiles, windows and random masks.
* ``metrics``: RMSE and isotropic wavenumber spectra.
* ``sda/``: the machinery underneath: MMPS guidance, Krylov solvers, the
  guided DDIM sampler and linear observation operators.

The command-line entry point is ``scripts/run_sda.py``.
"""
