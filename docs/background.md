# Where this code comes from

## Research background

The code and methods build on:

- **NVIDIA Modulus and cBottle**, including the diffusion architecture of
  [Karras et al. (2022)](https://arxiv.org/abs/2206.00364).
- **François Rozet's work** on
  [score-based data assimilation](https://arxiv.org/abs/2306.10574) and
  [moment matching posterior sampling](https://arxiv.org/abs/2405.13712).
- **Geophysical implementations** by
  [Martin et al. (2025)](https://doi.org/10.1029/2025MS005063) and
  [Manshausen et al. (2025)](https://arxiv.org/abs/2406.16947).
- **Our earlier ocean reconstruction study**,
  [Monkman et al. (2026)](https://doi.org/10.22541/essoar.15003398/v3),
  evaluates separate, independent models for SSH and SST and explains the
  posterior-sampling algorithm for an oceanographic audience. The work
  presented in this repository builds on that study with several upgrades
  and improvements, including joint SSH/SST modeling; the joint model was
  not evaluated in the paper.

## Source projects

This repository was extracted from two research codebases. The network and
training data pipeline come from `cBottle-for-ocean-DA/ocean-examples`. The
MMPS sampler comes from `oBottle-for-SSH-decomposition/oBottle-inference`.
Its tests check MMPS against the exact posterior of a Gaussian prior. The
released checkpoints were trained in cBottle-for-ocean-DA. The OSSE tiles are
the three Atlantic test cases used there. The windowed (multi-tile) SDA used
for full 336×336 tiles is not included.

---

[Documentation index](README.md) · [Project README](../README.md)
