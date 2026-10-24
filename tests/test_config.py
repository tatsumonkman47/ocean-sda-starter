# SPDX-License-Identifier: Apache-2.0
"""Config checks, CLI loading, run start modes and shared defaults."""

from dataclasses import asdict

import pytest
import torch
from omegaconf import OmegaConf

from obottle_starter.cli import load_config
from obottle_starter.core.config import (
    ConfigError,
    describe_architecture,
    run_differences,
    validate_sda,
    validate_training,
)
from obottle_starter.core.model import build_model, load_checkpoint
from obottle_starter.core.nn.loss import EDMLoss
from obottle_starter.inference.sda.defaults import DEFAULTS
from obottle_starter.training.train import train

from conftest import ROOT, tiny_config

BASE = str(ROOT / "configs" / "train_base.yaml")
SDA = str(ROOT / "configs" / "sda.yaml")


def test_presets_are_valid():
    for name in ("smoke_test", "structured", "ssh_only", "conditional"):
        cfg = load_config(BASE, [f"config={ROOT}/configs/train_{name}.yaml"])
        if cfg.data_source == "patches":
            cfg.data_dir = "/data"
        validate_training(cfg)
    validate_training(OmegaConf.merge(load_config(BASE, []), {"data_dir": "/data"}))


def test_cli_rejects_typos_and_overrides_list_items():
    with pytest.raises(SystemExit, match="unknown option lrr.*did you mean lr"):
        load_config(BASE, ["lrr=0.1"])
    with pytest.raises(SystemExit, match="conditioning.maks"):
        load_config(BASE, ["conditioning.maks=true"])
    cfg = load_config(
        BASE,
        [
            "variables.0.normalization.scale=0.05",
            "conditioning.training_observations.coverage={SSH: 0.2}",
        ],
    )
    assert cfg.variables[0].normalization.scale == 0.05
    assert cfg.variables[1].normalization.scale == 5.5
    assert cfg.conditioning.training_observations.coverage.SSH == 0.2


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"W": 64}, "square levels"),
        ({"H": 18, "W": 18, "channel_mult": [1, 2, 2]}, "not divisible by 2"),
        ({"attn_resolutions": [4]}, r"attn_resolutions \[4\] match no level"),
        ({"temporal_attn_resolutions": [4]}, "temporal_attn_resolutions"),
        ({"temporal_offset_channels": 31}, "even"),
        ({"sigma_min": 5.0, "sigma_max": 1.0}, "sigma_min < sigma_max"),
        (
            {"per_frame_adaptive_scale": True, "use_per_frame_adagn": False},
            "use_per_frame_adagn",
        ),
        ({"data_source": "patches"}, "needs data_dir"),
        ({"data_source": "netcdf"}, "data_source"),
        ({"start": "resume", "init_from": "x.pt"}, "init_from"),
        ({"ema_decay": 1.0}, "ema_decay"),
        ({"conditioning": {"mode": "observations", "observe": ["SSHA"]}}, "SSHA"),
        (
            {
                "conditioning": {
                    "mode": "observations",
                    "training_observations": {"coverage": 1.5},
                }
            },
            "coverage",
        ),
        ({"validation": {"every": 10, "sigmas": [0.0]}}, "sigmas"),
    ],
)
def test_invalid_training_configs(overrides, match):
    with pytest.raises(ConfigError, match=match):
        validate_training(tiny_config(**overrides))


def test_errors_are_collected():
    with pytest.raises(ConfigError) as err:
        validate_training(tiny_config(W=8, lr=-1.0, batch_size=0))
    message = str(err.value)
    assert "square" in message and "lr must" in message and "batch_size" in message


def test_validation_hours_must_not_overlap_training():
    cfg = tiny_config(
        data_source="patches",
        data_dir="/data",
        time_range_start=0,
        time_range_end=1000,
        validation={"every": 10, "time_range_start": 1100, "time_range_end": 1800},
    )
    with pytest.raises(ConfigError, match="overlap"):
        validate_training(cfg)  # gap of 100 h < 336 h maximum span
    cfg.validation.time_range_start = 1400
    validate_training(cfg)
    cfg.validation.time_range_end = 1600  # 200 h cannot hold a 336 h window
    with pytest.raises(ConfigError, match="too short"):
        validate_training(cfg)


def test_build_model_rejects_unusable_grids():
    with pytest.raises(ValueError, match="square"):
        build_model(tiny_config(W=32))


def test_architecture_report(tiny_cfg):
    text = describe_architecture(tiny_cfg)
    assert "level 0: 16x16, 32 channels" in text
    assert "level 1: 8x8, 64 channels, 2 blocks, temporal attention" in text


def test_per_frame_adaptive_scale_is_exposed():
    plain = build_model(tiny_config())
    scaled = build_model(tiny_config(per_frame_adaptive_scale=True))
    a, b = plain.state_dict(), scaled.state_dict()
    key = "model.enc.16x16_block0.affine.weight"
    assert b[key].shape[0] == 2 * a[key].shape[0]  # scale and shift
    x = torch.randn(1, 2, 3, 256)
    out = scaled(
        x,
        torch.tensor([1.0]),
        condition=torch.zeros(1, 32, 3, 256),
        time_offsets=torch.tensor([[0.0, 6.0, 30.0]]),
        second_of_day=torch.zeros(1, 3),
        day_of_year=torch.zeros(1, 3),
    ).out
    assert out.shape == x.shape and torch.isfinite(out).all()


def test_loss_requires_finite_targets():
    loss = EDMLoss(distribution="log_uniform", sigma_data=1.0)

    class Out:
        def __init__(self, x):
            self.out, self.logits = x, None

    images = torch.randn(2, 1, 3, 16)
    assert torch.isfinite(loss(lambda x, s: Out(0 * x), images).total).all()
    images[0, 0, 1, 4] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        loss(lambda x, s: Out(x), images)


def test_sda_yaml_matches_python_defaults():
    cfg = OmegaConf.load(SDA)
    expected = asdict(DEFAULTS)
    got = {
        "n_members": cfg.sampler.n_members,
        "n_batch": cfg.sampler.n_batch,
        "steps": cfg.sampler.steps,
        "timestep_spacing": cfg.sampler.timestep_spacing,
        "mmps_solver": cfg.mmps.solver,
        "mmps_iterations": cfg.mmps.iterations,
        "rozet_gamma": cfg.rozet.gamma,
        "cfg_weight": cfg.classifier_free.weight,
    }
    assert got == expected


def test_validate_sda(tiny_cfg):
    cfg = load_config(SDA, ["ckpt=a", "tile=b", "output=c"])
    validate_sda(cfg, tiny_cfg)
    with pytest.raises(ConfigError, match="conditional model"):
        validate_sda(OmegaConf.merge(cfg, {"guidance": {"method": "cfg"}}), tiny_cfg)
    with pytest.raises(ConfigError, match="sigma_y.SST"):
        validate_sda(
            OmegaConf.merge(cfg, {"observations": {"sigma_y": {"SST": 0.0}}}), tiny_cfg
        )
    with pytest.raises(ConfigError, match="observations.mode"):
        validate_sda(OmegaConf.merge(cfg, {"observations": {"mode": "real"}}), tiny_cfg)


# ── How a run starts ──────────────────────────────────────────────────────


def _run(tmp_path, **overrides):
    settings = dict(output_dir=str(tmp_path / "run"), num_steps=2, checkpoint_every=2)
    return tiny_config(**{**settings, **overrides})


def test_fresh_refuses_an_existing_run(tmp_path):
    train(_run(tmp_path))
    with pytest.raises(ValueError, match="already holds a run"):
        train(_run(tmp_path))


def test_resume_rejects_changes_that_alter_the_run(tmp_path):
    train(_run(tmp_path))
    with pytest.raises(ValueError, match="model_channels"):
        train(_run(tmp_path, start="resume", num_steps=3, model_channels=16))
    reordered = _run(tmp_path, start="resume", num_steps=3)
    reordered.variables = [reordered.variables[1], reordered.variables[0]]
    with pytest.raises(ValueError, match="variables.0.name"):
        train(reordered)
    with pytest.raises(ValueError, match="no checkpoint"):
        train(tiny_config(output_dir=str(tmp_path / "none"), start="resume"))


def test_resume_allows_runtime_changes_and_records_them(tmp_path):
    out = train(_run(tmp_path))
    original = (out / "config.yaml").read_text()
    train(_run(tmp_path, start="resume", num_steps=4, log_every=1))
    assert (out / "config.yaml").read_text() == original
    changes = (out / "config_changes.yaml").read_text()
    assert "num_steps: 2 -> 4" in changes
    assert torch.load(out / "checkpoints" / "checkpoint_latest.pt")["step"] == 4


def test_auto_starts_fresh_then_resumes(tmp_path):
    train(_run(tmp_path, start="auto"))
    train(_run(tmp_path, start="auto", num_steps=3))
    state = torch.load(tmp_path / "run" / "checkpoints" / "checkpoint_latest.pt")
    assert state["step"] == 3


def test_checkpoint_records_climatology_checksum(tmp_path):
    out = train(_run(tmp_path))
    _, cfg = load_checkpoint(out / "checkpoints" / "checkpoint_latest.pt", "cpu")
    assert len(cfg.variables[1].normalization.climatology.sha256) == 64


def test_run_differences_ignores_runtime_keys(tiny_cfg):
    other = OmegaConf.merge(tiny_cfg, {"num_steps": 99, "lr": 0.1})
    assert run_differences(other, tiny_cfg) == ["lr: 0.0002 -> 0.1"]
