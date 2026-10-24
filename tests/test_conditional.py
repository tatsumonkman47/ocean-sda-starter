# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for conditional models (M_3 / M_4): they build, train, load
from an unconditional checkpoint, and sample. Not a test of skill."""

import numpy as np
import pytest
import torch

from obottle_starter import assimilate, load_checkpoint
from obottle_starter.core.model import build_model, load_weights_padded
from obottle_starter.core.obs_conditioning import observation_channels
from obottle_starter.inference.osse import random_masks
from obottle_starter.training.train import train

from conftest import tiny_config

TIMES = np.datetime64("2012-03-01T00") + np.arange(3) * np.timedelta64(12, "h")


def conditional_config(**overrides):
    cond = {"mode": "observations", "observe": ["SSH", "SST"], "mask_channel": True}
    cond.update(overrides.pop("conditioning", {}))
    return tiny_config(conditioning=cond, **overrides)


@pytest.mark.parametrize(
    "observe, mask_channel, n", [(["SSH", "SST"], True, 4), (["SST"], False, 1)]
)
def test_input_channels(observe, mask_channel, n):
    cfg = conditional_config(
        conditioning={"observe": observe, "mask_channel": mask_channel}
    )
    assert observation_channels(cfg) == n
    build_model(cfg)  # input layer sized for the extra channels


def test_init_from_unconditional_reproduces_it(tmp_path):
    torch.manual_seed(0)
    base_cfg = tiny_config()
    base = build_model(base_cfg).eval()
    cfg = conditional_config()
    net = build_model(cfg).eval()
    padded = load_weights_padded(net, base.state_dict(), cfg, base_cfg)
    assert padded == ["model.enc.16x16_conv_embed.module.weight"]
    assert padded  # the first convolution gained input channels
    x = torch.randn(1, 2, 3, 256)
    sigma = torch.tensor([1.0])
    time_cond = torch.randn(1, 32, 3, 256)
    obs_cond = torch.randn(1, observation_channels(cfg), 3, 256)
    cal = dict(
        second_of_day=torch.zeros(1, 3),
        day_of_year=torch.zeros(1, 3),
        time_offsets=torch.tensor([[0.0, 12.0, 24.0]]),
    )
    with torch.no_grad():
        a = base(x, sigma, condition=time_cond, **cal).out
        b = net(x, sigma, condition=torch.cat([obs_cond, time_cond], 1), **cal).out
    torch.testing.assert_close(a, b)


@pytest.mark.parametrize("masks", ["bernoulli", "stripes"])
def test_train_conditional(tmp_path, masks):
    cfg = conditional_config(
        output_dir=str(tmp_path / masks),
        num_steps=2,
        conditioning={
            "training_observations": {"masks": masks, "null_probability": 0.5}
        },
    )
    out = train(cfg)
    net, loaded = load_checkpoint(out / "checkpoints" / "checkpoint_latest.pt", "cpu")
    assert loaded.conditioning.mode == "observations"


def test_assimilate_conditional_and_cfg(tmp_path):
    cfg = conditional_config(
        output_dir=str(tmp_path / "run"),
        num_steps=1,
        conditioning={"training_observations": {"null_probability": 0.5}},
    )
    net, cfg = load_checkpoint(
        train(cfg) / "checkpoints" / "checkpoint_latest.pt", "cpu"
    )
    rng = np.random.default_rng(0)
    truth = {
        "SSH": rng.normal(0, 0.05, (3, 16, 16)),
        "SST": rng.normal(18, 1, (3, 16, 16)),
    }
    masks = random_masks(["SSH", "SST"], (3, 16, 16), 0.3, seed=0)
    common = dict(
        times=TIMES,
        center_lon=0.0,
        masks=masks,
        truth=truth,
        sigma_y={"SSH": 0.005, "SST": 0.1},
        n_members=2,
        steps=3,
        verbose=False,
        seed=0,
    )
    plain = assimilate(net, cfg, guidance="conditional", **common)
    same = assimilate(net, cfg, guidance="cfg", cfg_weight=1.0, **common)
    np.testing.assert_allclose(plain.samples, same.samples, rtol=1e-5, atol=1e-6)
    pushed = assimilate(net, cfg, guidance="cfg", cfg_weight=2.0, **common)
    hybrid = assimilate(net, cfg, guidance="mmps", mmps_iterations=1, **common)
    for ds in (pushed, hybrid):
        assert np.isfinite(ds.samples).all()
    assert pushed.attrs["cfg_weight"] == 2.0 and plain.attrs["conditional_model"] == 1


def test_conditional_guidance_needs_conditional_model(tiny_net, tiny_cfg):
    masks = random_masks(["SSH"], (3, 16, 16), 0.3, seed=0)
    truth = {"SSH": np.zeros((3, 16, 16)), "SST": np.full((3, 16, 16), 18.0)}
    with pytest.raises(ValueError, match="conditional model"):
        assimilate(
            tiny_net,
            tiny_cfg,
            times=TIMES,
            center_lon=0.0,
            masks=masks,
            truth=truth,
            guidance="conditional",
            verbose=False,
        )


def test_cfg_needs_a_trained_null_branch(tmp_path):
    cfg = conditional_config(output_dir=str(tmp_path / "run"), num_steps=1)
    net, cfg = load_checkpoint(
        train(cfg) / "checkpoints" / "checkpoint_latest.pt", "cpu"
    )
    masks = random_masks(["SSH"], (3, 16, 16), 0.3, seed=0)
    truth = {"SSH": np.zeros((3, 16, 16)), "SST": np.full((3, 16, 16), 18.0)}
    with pytest.raises(ValueError, match="null_probability"):
        assimilate(
            net,
            cfg,
            times=TIMES,
            center_lon=0.0,
            masks=masks,
            truth=truth,
            sigma_y=0.01,
            guidance="cfg",
            verbose=False,
        )


def test_obs_condition_ignores_unobserved_nans():
    from obottle_starter.core.obs_conditioning import build_obs_condition

    cfg = conditional_config()
    values = torch.randn(2, 3, 256)
    masks = {"SSH": torch.rand(3, 256) < 0.3, "SST": torch.rand(3, 256) < 0.3}
    clean = build_obs_condition(cfg, values, masks)
    dirty = values.clone()
    dirty[0][~masks["SSH"]] = float("nan")
    dirty[1][~masks["SST"]] = float("inf")
    out = build_obs_condition(cfg, dirty, masks)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, clean)
    dirty[0][masks["SSH"].nonzero()[0, 0], masks["SSH"].nonzero()[0, 1]] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        build_obs_condition(cfg, dirty, masks)


def test_init_from_rejects_different_variables():
    base_cfg = tiny_config()
    base = build_model(base_cfg)
    cfg = conditional_config()
    cfg.variables = [cfg.variables[1], cfg.variables[0]]  # SST, SSH: reordered
    with pytest.raises(ValueError, match="variables"):
        load_weights_padded(build_model(cfg), base.state_dict(), cfg, base_cfg)
