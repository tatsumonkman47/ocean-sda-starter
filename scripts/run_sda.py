#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Assimilate synthetic observations from an OSSE tile and save the ensemble.

    python scripts/run_sda.py ckpt=checkpoints/joint_ssh_sst_T5.pt \
        tile=osse_tiles/MidAtlantic_35N_50W.nc output=mid_atlantic.nc

The truth comes from the tile's simulation. With observations.mode=instrument
(default) the coverage is real SWOT+nadir and ACSPO footprints, but the
observed values are still taken from the simulated truth: this is an OSSE,
not an assimilation of measured data. The model's variables and frame count
decide what is read. Defaults are in configs/sda.yaml. The output netCDF
holds ``samples``, ``truth``, ``mask`` and ``observations`` in physical
units on (sample, channel, time, row, col), plus latitude and longitude.
"""

from pathlib import Path

import torch
from omegaconf import OmegaConf

from obottle_starter import assimilate, channel_names, load_checkpoint
from obottle_starter.cli import load_config
from obottle_starter.core.config import ConfigError, validate_sda
from obottle_starter.core.normalization import Normalizer
from obottle_starter.inference.metrics import summarize
from obottle_starter.inference.osse import load_window, random_masks

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    cfg = load_config(str(ROOT / "configs" / "sda.yaml"))
    device = cfg.device if torch.cuda.is_available() else "cpu"
    net, model_cfg = load_checkpoint(cfg.ckpt, device)
    try:
        validate_sda(cfg, model_cfg)
    except ConfigError as err:
        raise SystemExit(str(err)) from None
    names = channel_names(model_cfg)
    print(
        f"model: channels {names}, {model_cfg.time_length} frames, "
        f"{model_cfg.H}x{model_cfg.W}, device {device}"
    )

    win = load_window(
        cfg.tile,
        names,
        model_cfg.time_length,
        t_start=cfg.window.t_start,
        stride=cfg.window.stride,
        size=model_cfg.H,
        origin=cfg.window.origin,
    )
    obs_cfg = cfg.observations
    observe = list(obs_cfg.observe or names)
    if obs_cfg.mode == "instrument":
        print("observations: real instrument coverage, values from simulated truth")
        masks = {c: win.masks[c] for c in observe}
    elif obs_cfg.mode == "random":
        shape = win.truth[names[0]].shape
        masks = random_masks(
            observe, shape, dict(obs_cfg.random_coverage), seed=obs_cfg.seed
        )
    elif obs_cfg.mode == "none":
        masks = {}
    for c, m in masks.items():
        print(f"  {c}: {int(m.sum())} observed pixels ({100 * m.mean():.1f}%)")

    normalizer = Normalizer.from_settings(
        model_cfg, OmegaConf.to_container(cfg.normalization)
    )
    sigma_y = {c: obs_cfg.sigma_y[c] for c in masks}
    print(normalizer.describe(sigma_y))

    ds = assimilate(
        net,
        model_cfg,
        times=win.times,
        center_lon=win.center_lon,
        masks=masks,
        truth=win.truth,
        sigma_y=sigma_y,
        normalizer=normalizer,
        instrument_noise=bool(obs_cfg.instrument_noise),
        obs_seed=obs_cfg.seed,
        n_members=cfg.sampler.n_members,
        n_batch=cfg.sampler.n_batch,
        steps=cfg.sampler.steps,
        timestep_spacing=cfg.sampler.timestep_spacing,
        seed=cfg.sampler.seed,
        guidance=cfg.guidance.method,
        record_diagnostics=cfg.guidance.record_diagnostics,
        mmps_solver=cfg.mmps.solver,
        mmps_iterations=cfg.mmps.iterations,
        rozet_gamma=cfg.rozet.gamma,
        cfg_weight=cfg.classifier_free.weight,
    )
    ds = ds.assign_coords(
        latitude=(("row", "col"), win.latitude),
        longitude=(("row", "col"), win.longitude),
        tile_frame=("time", win.frames),
    )
    ds.attrs.update(
        tile=str(cfg.tile),
        window_origin=list(win.origin),
        sda_config=OmegaConf.to_yaml(cfg),
    )

    print(summarize(ds, sigma_y))
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(cfg.output)
    print(f"wrote {cfg.output}")


if __name__ == "__main__":
    main()
