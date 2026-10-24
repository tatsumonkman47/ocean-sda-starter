# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures: a tiny untrained model on CPU."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]


def tiny_config(**overrides):
    cfg = OmegaConf.merge(
        OmegaConf.load(ROOT / "configs" / "train_base.yaml"),
        OmegaConf.load(ROOT / "configs" / "train_smoke_test.yaml"),
        {"H": 16, "W": 16, "time_length": 3, "temporal_attn_resolutions": [8]},
        overrides,
    )
    return cfg


@pytest.fixture
def tiny_cfg():
    return tiny_config()


@pytest.fixture
def tiny_net(tiny_cfg):
    from obottle_starter.core.model import build_model

    torch.manual_seed(0)
    net = build_model(tiny_cfg, "cpu").eval().requires_grad_(False)
    net.use_fp16 = False
    return net
