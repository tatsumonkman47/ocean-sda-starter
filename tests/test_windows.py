# SPDX-License-Identifier: Apache-2.0
"""Sliding windows (Rozet & Louppe, 2023): layout, per-window conditioning,
gradients, and agreement with the plain denoiser for a single window."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from obottle_starter import assimilate
from obottle_starter.inference.osse import random_masks
from obottle_starter.inference.sda._azula.noise import VESchedule
from obottle_starter.inference.sda.denoiser import EDMPrecondDenoiser
from obottle_starter.inference.sda.windows import (
    WindowedDenoiser,
    sliding_windows,
    window_layout,
)

HOUR = np.timedelta64(1, "h")
# Irregular spacing, so each window's offsets from its own first frame differ.
HOURS = np.array([0, 12, 24, 30, 48, 60, 66, 84, 96])
TIMES = np.datetime64("2012-03-01T00") + HOURS * HOUR


@pytest.mark.parametrize(
    "L, T, step",
    [
        (3, 3, 1),
        (7, 3, 1),
        (9, 5, 1),
        (9, 3, 3),
        (11, 5, 2),
        (12, 4, 1),
        (13, 5, 4),
        (10, 4, 2),
    ],
)
def test_layout_covers_every_frame_once(L, T, step):
    starts, source, local = window_layout(L, T, step)
    assert len(starts) == (L - T) // step + 1
    np.testing.assert_array_equal(starts[source] + local, np.arange(L))
    assert ((local >= 0) & (local < T)).all()
    # Frames come from consecutive windows, in order.
    assert (np.diff(source) >= 0).all() and set(source) == set(range(len(starts)))


def test_layout_is_rozet_and_louppe_for_step_1():
    starts, source, local = window_layout(9, 5, 1)  # order k = 2
    np.testing.assert_array_equal(starts, [0, 1, 2, 3, 4])
    # First k frames from the first window, last k from the last, the rest
    # from the window centred on them.
    np.testing.assert_array_equal(source, [0, 0, 0, 1, 2, 3, 4, 4, 4])
    np.testing.assert_array_equal(local, [0, 1, 2, 2, 2, 2, 2, 3, 4])


def test_layout_blocks_for_step_T():
    starts, source, local = window_layout(9, 3, 3)
    np.testing.assert_array_equal(source, [0, 0, 0, 1, 1, 1, 2, 2, 2])
    np.testing.assert_array_equal(local, [0, 1, 2] * 3)


@pytest.mark.parametrize(
    "L, T, step, match",
    [(9, 5, 3, "Use 8 or 11"), (2, 3, 1, "fewer"), (9, 3, 4, "between")],
)
def test_layout_rejects_bad_settings(L, T, step, match):
    with pytest.raises(ValueError, match=match):
        window_layout(L, T, step)


class OffsetNet(torch.nn.Module):
    """Stands in for the network: returns each frame's offset (hours) from
    its window's first frame, so the output shows where every frame came from."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))

    def forward(self, x, sigma, force_fp32=True, **kwargs):
        out = x * 0 + kwargs["time_offsets"][:, None, :, None]
        return SimpleNamespace(out=out)


def test_each_window_gets_its_own_conditioning(tiny_cfg):
    windows = sliding_windows(TIMES, 0.0, tiny_cfg, step=1)  # T = 3
    den = WindowedDenoiser(OffsetNet(), VESchedule(), windows, window_batch=4)
    out = den(torch.zeros(2, 2, 9, 256), torch.tensor(0.5)).mean
    starts, source, _ = window_layout(9, 3, 1)
    expected = HOURS - HOURS[starts[source]]
    np.testing.assert_allclose(out[0, 0, :, 0].numpy(), expected)
    assert torch.equal(out[1], out[0])  # same layout for every member


class LocalScaleNet(torch.nn.Module):
    """A linear stand-in whose output is ``(1 + local frame index) * x``."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))

    def forward(self, x, sigma, force_fp32=True, **kwargs):
        T = x.shape[2]
        return SimpleNamespace(out=x * torch.arange(1.0, T + 1)[None, None, :, None])


def test_gradients_flow_through_the_kept_frames_only(tiny_cfg):
    windows = sliding_windows(TIMES, 0.0, tiny_cfg, step=1)
    den = WindowedDenoiser(LocalScaleNet(), VESchedule(), windows)
    x = torch.randn(1, 2, 9, 256, requires_grad=True)
    (grad,) = torch.autograd.grad(den(x, torch.tensor(0.5)).mean.sum(), x)
    _, _, local = window_layout(9, 3, 1)
    # d out_j / d x_j = 1 + local[j]; other windows' frames are discarded.
    expected = torch.from_numpy(1.0 + local).float()[None, None, :, None].expand_as(x)
    torch.testing.assert_close(grad, expected)


def test_one_window_equals_the_plain_denoiser(tiny_net, tiny_cfg):
    from obottle_starter.core.conditioning import frame_conditioning

    times = TIMES[:3]
    schedule = VESchedule(sigma_min=0.0002, sigma_max=200.0)
    plain = EDMPrecondDenoiser(
        tiny_net, schedule, **frame_conditioning(times, 10.0, tiny_cfg)
    )
    windowed = WindowedDenoiser(
        tiny_net, schedule, sliding_windows(times, 10.0, tiny_cfg)
    )
    x = torch.randn(2, 2, 3, 256)
    t = torch.tensor(0.4)
    torch.testing.assert_close(windowed(x, t).mean, plain(x, t).mean)


def test_window_batch_does_not_change_the_result(tiny_net, tiny_cfg):
    windows = sliding_windows(TIMES[:7], 0.0, tiny_cfg, step=1)
    schedule = VESchedule(sigma_min=0.0002, sigma_max=200.0)
    x = torch.randn(2, 2, 7, 256)
    t = torch.tensor(0.3)
    a = WindowedDenoiser(tiny_net, schedule, windows)(x, t).mean
    b = WindowedDenoiser(tiny_net, schedule, windows, window_batch=3)(x, t).mean
    torch.testing.assert_close(a, b)


@pytest.mark.parametrize("guidance, step", [("mmps", 1), ("rozet", 1), ("rozet", 3)])
def test_assimilate_a_longer_sequence(tiny_net, tiny_cfg, guidance, step):
    rng = np.random.default_rng(0)
    L = 6
    truth = {
        "SSH": rng.normal(0, 0.05, (L, 16, 16)),
        "SST": rng.normal(18, 1, (L, 16, 16)),
    }
    masks = random_masks(["SSH", "SST"], (L, 16, 16), 0.2, seed=0)
    post = assimilate(
        tiny_net,
        tiny_cfg,
        times=TIMES[:L],
        center_lon=0.0,
        masks=masks,
        truth=truth,
        sigma_y={"SSH": 0.005, "SST": 0.1},
        guidance=guidance,
        window_step=step,
        n_members=2,
        steps=2,
        mmps_iterations=1,
        verbose=False,
        seed=0,
    )
    assert post.samples.shape == (2, 2, L, 16, 16)
    assert np.isfinite(post.samples).all()
    assert post.attrs["n_windows"] == (L - 3) // step + 1
    assert post.attrs["window_step"] == step
    np.testing.assert_array_equal(post.time.values, TIMES[:L].astype("datetime64[ns]"))
    observed = np.isfinite(post.observations.sel(channel="SSH").values)
    np.testing.assert_array_equal(observed, masks["SSH"])


def test_prior_samples_of_a_longer_sequence(tiny_net, tiny_cfg):
    post = assimilate(
        tiny_net,
        tiny_cfg,
        times=TIMES,
        center_lon=0.0,
        masks={},
        n_members=1,
        steps=2,
        verbose=False,
        seed=0,
    )
    assert post.samples.shape == (1, 2, 9, 16, 16)
    assert post.attrs["method"] == "unconditional"


def test_sliding_windows_need_an_unconditional_prior(tmp_path):
    from obottle_starter import load_checkpoint
    from obottle_starter.training.train import train

    from conftest import tiny_config

    cfg = tiny_config(
        output_dir=str(tmp_path / "run"),
        num_steps=1,
        conditioning={"mode": "observations"},
    )
    net, cfg = load_checkpoint(
        train(cfg) / "checkpoints" / "checkpoint_latest.pt", "cpu"
    )
    masks = random_masks(["SSH"], (9, 16, 16), 0.3, seed=0)
    truth = {"SSH": np.zeros((9, 16, 16)), "SST": np.full((9, 16, 16), 18.0)}
    with pytest.raises(ValueError, match="unconditional priors"):
        assimilate(
            net,
            cfg,
            times=TIMES,
            center_lon=0.0,
            masks=masks,
            truth=truth,
            sigma_y=0.01,
            guidance="conditional",
            verbose=False,
        )
