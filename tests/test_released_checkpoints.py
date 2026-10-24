# SPDX-License-Identifier: Apache-2.0
"""The released checkpoints reproduce the original cBottle-for-ocean-DA network.

Set OBOTTLE_CHECKPOINTS to the directory holding joint_ssh_sst_T5.pt and
ssh_only_T11.pt. The reference values are a strided sample of the original
network's outputs on the inputs built here. The starter's GroupNorm uses the
population variance at inference (as torch does in training), so outputs
differ from the original by ~1e-4.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from obottle_starter.core.conditioning import sinusoidal_time_encoding
from obottle_starter.core.model import channel_names, load_checkpoint

CKPT_DIR = os.environ.get("OBOTTLE_CHECKPOINTS")
REF = np.load(Path(__file__).parent / "data" / "reference_outputs.npz")


@pytest.mark.skipif(not CKPT_DIR, reason="set OBOTTLE_CHECKPOINTS to run")
@pytest.mark.parametrize(
    "name, channels", [("joint_ssh_sst_T5", ["SSH", "SST"]), ("ssh_only_T11", ["SSH"])]
)
def test_matches_original_network(name, channels):
    net, cfg = load_checkpoint(Path(CKPT_DIR) / f"{name}.pt", "cpu")
    assert channel_names(cfg) == channels
    T, C, HW = cfg.time_length, len(channel_names(cfg)), cfg.H * cfg.W
    x = torch.randn(3, C, T, HW, generator=torch.Generator().manual_seed(0))
    sigma = torch.tensor([0.01, 1.0, 50.0])
    off = torch.arange(T, dtype=torch.float32) * 12
    enc = sinusoidal_time_encoding(off, cfg.temporal_offset_channels)
    kwargs = dict(
        condition=enc.unsqueeze(-1).expand(-1, -1, HW).expand(3, -1, -1, -1),
        class_labels=torch.zeros(3, cfg.label_dim),
        second_of_day=(off * 3600 % 86400).expand(3, T),
        day_of_year=(250 + off / 24).expand(3, T),
        time_offsets=off.expand(3, T),
    )
    with torch.no_grad():
        out = net(x * (1 + sigma.view(-1, 1, 1, 1)), sigma, force_fp32=True, **kwargs)
    got = out.out[..., ::997].numpy()
    np.testing.assert_allclose(got, REF[name], atol=1e-3, rtol=0)
