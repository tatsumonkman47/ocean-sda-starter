# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
import torch

from obottle_starter.core.conditioning import (
    frame_conditioning,
    sinusoidal_time_encoding,
)


def test_legacy_encoding():
    off = torch.tensor([0.0, 1.0, 13.0, 336.0])
    enc = sinusoidal_time_encoding(off, 8)
    freqs = torch.logspace(-1, 2, 4)
    angles = 2 * torch.pi * freqs[:, None] * off[None] / 720.0
    torch.testing.assert_close(enc, torch.cat([angles.sin(), angles.cos()]))


def test_frame_conditioning(tiny_cfg):
    times = np.array(
        ["2012-01-01T22", "2012-01-02T04", "2012-01-03T22"], dtype="datetime64[ns]"
    )
    cond = frame_conditioning(times, -90.0, tiny_cfg)
    torch.testing.assert_close(cond["time_offsets"], torch.tensor([0.0, 6.0, 48.0]))
    # 22:00 UTC at 90°W is 16:00 local solar time.
    assert cond["second_of_day"][0] == 16 * 3600
    torch.testing.assert_close(cond["day_of_year"], torch.tensor([0.0, 1.0, 2.0]))
    assert cond["condition"].shape == (32, 3, 16 * 16)
    assert cond["class_labels"].shape == (8,)
    with pytest.raises(ValueError, match="frames"):
        frame_conditioning(times[:2], 0.0, tiny_cfg)
