# SPDX-License-Identifier: Apache-2.0
"""Training a prior: datasets and the training loop.

* ``data``: ``PatchDataset`` (per-patch zarr stores) and ``SyntheticDataset``.
* ``train``: ``train(cfg)``, single GPU or DDP; see ``configs/train_base.yaml``.

The command-line entry point is ``scripts/train.py``.
"""
