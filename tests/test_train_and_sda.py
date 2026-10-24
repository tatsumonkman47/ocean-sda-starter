# SPDX-License-Identifier: Apache-2.0
"""End-to-end on a tiny untrained model: train, export, load, assimilate."""

import numpy as np
import pytest

from obottle_starter import (
    assimilate,
    channel_names,
    export_checkpoint,
    load_checkpoint,
)
from obottle_starter.inference.osse import random_masks
from obottle_starter.training.train import train

from conftest import tiny_config

TIMES = np.datetime64("2012-03-01T00") + np.arange(3) * np.timedelta64(12, "h")


def test_train_resume_export(tmp_path):
    cfg = tiny_config(
        output_dir=str(tmp_path / "run"), num_steps=3, checkpoint_every=2, log_every=1
    )
    out = train(cfg)
    ckpts = sorted(p.name for p in (out / "checkpoints").glob("*.pt"))
    assert ckpts == [
        "checkpoint_latest.pt",
        "checkpoint_step2.pt",
        "checkpoint_step3.pt",
    ]
    cfg.num_steps = 4
    cfg.start = "resume"
    train(cfg)  # resumes from checkpoint_latest.pt
    export_checkpoint(
        out / "checkpoints" / "checkpoint_latest.pt",
        tmp_path / "e.pt",
        extra={"description": "tiny"},
    )
    net, loaded = load_checkpoint(tmp_path / "e.pt", "cpu")
    assert channel_names(loaded) == ["SSH", "SST"] and loaded.time_length == 3
    assert loaded.description == "tiny"


@pytest.mark.parametrize("observe", [["SSH", "SST"], ["SSH"], []])
def test_assimilate(tiny_net, tiny_cfg, observe):
    rng = np.random.default_rng(0)
    truth = {
        "SSH": rng.normal(0, 0.05, (3, 16, 16)),
        "SST": rng.normal(18, 1, (3, 16, 16)),
    }
    masks = random_masks(observe, (3, 16, 16), 0.2, seed=0)
    ds = assimilate(
        tiny_net,
        tiny_cfg,
        times=TIMES,
        center_lon=-40.0,
        masks=masks,
        truth=truth,
        sigma_y={"SSH": 0.005, "SST": 0.1},
        n_members=3,
        n_batch=2,
        steps=4,
        mmps_iterations=2,
        verbose=False,
        seed=0,
    )
    assert ds.samples.shape == (3, 2, 3, 16, 16)
    assert np.isfinite(ds.samples).all()
    np.testing.assert_allclose(ds.truth.sel(channel="SST"), truth["SST"], rtol=1e-5)
    assert ds.attrs["method"] == ("mmps" if observe else "unconditional")
    for c in ("SSH", "SST"):
        assert bool(ds.mask.sel(channel=c).any()) == (c in observe)
    assert ds.attrs["normalization_source"] == "checkpoint"
    assert "SSH: x_model = x / 0.051 m" in ds.attrs["normalization_formulas"]


def test_assimilate_real_observations(tiny_net, tiny_cfg):
    rng = np.random.default_rng(1)
    y = {"SSH": rng.normal(0, 0.05, (3, 16, 16))}
    masks = random_masks(["SSH"], (3, 16, 16), 0.3, seed=1)
    ds = assimilate(
        tiny_net,
        tiny_cfg,
        times=TIMES,
        center_lon=0.0,
        masks=masks,
        y=y,
        sigma_y=0.01,
        n_members=2,
        steps=3,
        verbose=False,
    )
    assert "truth" not in ds and ds.samples.shape[0] == 2


def test_assimilate_rozet(tiny_net, tiny_cfg):
    rng = np.random.default_rng(2)
    truth = {
        "SSH": rng.normal(0, 0.05, (3, 16, 16)),
        "SST": rng.normal(18, 1, (3, 16, 16)),
    }
    masks = random_masks(["SSH", "SST"], (3, 16, 16), 0.2, seed=2)
    ds = assimilate(
        tiny_net,
        tiny_cfg,
        times=TIMES,
        center_lon=0.0,
        masks=masks,
        truth=truth,
        sigma_y={"SSH": 0.005, "SST": 0.1},
        guidance="rozet",
        rozet_gamma=0.1,
        record_diagnostics=True,
        n_members=2,
        steps=3,
        verbose=False,
    )
    assert ds.attrs["method"] == "rozet" and ds.attrs["rozet_gamma"] == 0.1
    assert np.isfinite(ds.samples).all() and "solver_misfit_rms_SSH" in ds
    with pytest.raises(ValueError, match="guidance"):
        assimilate(
            tiny_net,
            tiny_cfg,
            times=TIMES,
            center_lon=0.0,
            masks=masks,
            truth=truth,
            guidance="dps",
            verbose=False,
        )


def test_log_normal_noise_schedule(tmp_path):
    cfg = tiny_config(
        output_dir=str(tmp_path / "ln"),
        num_steps=2,
        noise_distribution="log_normal",
        p_mean=0.0,
        p_std=1.0,
    )
    train(cfg)
    assert (tmp_path / "ln" / "checkpoints" / "checkpoint_latest.pt").exists()


def test_prior_without_truth(tiny_net, tiny_cfg):
    ds = assimilate(
        tiny_net,
        tiny_cfg,
        times=TIMES,
        center_lon=0.0,
        masks={},
        n_members=2,
        steps=3,
        verbose=False,
    )
    assert ds.attrs["method"] == "unconditional" and "truth" not in ds


def test_validation_pass_and_structured_data(tmp_path, capsys):
    cfg = tiny_config(
        output_dir=str(tmp_path / "val"),
        data_source="structured",
        num_steps=2,
        validation={"every": 1, "n_samples": 3, "sigmas": [0.1, 10.0]},
    )
    out = train(cfg)
    rows = (out / "validation.csv").read_text().strip().splitlines()
    assert len(rows) == 4  # 2 steps x 2 noise levels
    step, sigma, ssh, sst = rows[0].split(",")
    assert (step, sigma) == ("1", "0.1") and float(ssh) > 0 and float(sst) > 0
    assert "sigma 10: SSH" in capsys.readouterr().out


def test_validation_loss_is_repeatable(tiny_net, tiny_cfg):
    from omegaconf import OmegaConf

    from obottle_starter.training.data import validation_samples
    from obottle_starter.training.validation import denoising_loss_by_sigma

    cfg = OmegaConf.merge(tiny_cfg, {"validation": {"every": 1, "n_samples": 2}})
    samples = validation_samples(cfg)
    a = denoising_loss_by_sigma(tiny_net, samples, [0.5], cfg)
    b = denoising_loss_by_sigma(tiny_net, samples, [0.5], cfg)
    assert a == b and len(a[0.5]) == 2
