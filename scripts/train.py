#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Train an EDMPrecond(SongUNet) ocean prior.

    python scripts/train.py config=configs/train_smoke_test.yaml   # 1-min code check
    python scripts/train.py config=configs/train_structured.yaml   # teaching data
    python scripts/train.py data_dir=/path/to/patches               # released joint setup
    python scripts/train.py config=configs/train_ssh_only.yaml data_dir=...
    python scripts/train.py ... start=resume                        # continue a run
    torchrun --nproc_per_node=4 scripts/train.py data_dir=...       # 4 GPUs

configs/train_base.yaml holds every option and its default; config=<file> merges
an experiment file on top, and key=value overrides either. The config is
checked before any data is read.
"""

from pathlib import Path

from obottle_starter.cli import load_config
from obottle_starter.core.config import ConfigError
from obottle_starter.training.train import train

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    cfg = load_config(str(ROOT / "configs" / "train_base.yaml"))
    try:
        train(cfg)
    except ConfigError as err:
        raise SystemExit(str(err)) from None
