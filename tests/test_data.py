# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
import xarray as xr
from omegaconf import OmegaConf

from obottle_starter.training.data import (
    PatchDataset,
    StructuredDataset,
    SyntheticDataset,
    structured_fields,
)
from obottle_starter.core.normalization import Normalizer

from conftest import tiny_config


def _variables(cfg, fields):
    """The tiny config's variables, read from ``fields``."""
    specs = OmegaConf.to_container(cfg.variables)[: len(fields)]
    for spec, field in zip(specs, fields):
        spec["field"] = field
    return specs


def _write_patches(root, fields, n=16, hours=80, nan_at=None):
    rng = np.random.default_rng(0)
    times = np.datetime64("2011-09-13T00", "ns") + np.arange(hours) * np.timedelta64(
        1, "h"
    )
    lat, lon = np.meshgrid(
        np.linspace(30, 31, n), np.linspace(-150, -149, n), indexing="ij"
    )
    for field, var, scale in fields:
        for pid in ("000", "001"):
            data = scale * rng.standard_normal((hours, n, n)).astype(np.float32)
            if nan_at is not None:
                data[nan_at] = np.nan
            ds = xr.Dataset(
                {var: (("time", "x", "y"), data)},
                coords={
                    "time": times,
                    "latitude": (("x", "y"), lat),
                    "longitude": (("x", "y"), lon),
                },
            )
            ds.to_zarr(root / field / f"{pid}.zarr", zarr_format=2)


def test_patch_dataset(tmp_path):
    fields = [("ssh_field", "SSH", 0.1), ("sst_field", "SST", 20.0)]
    _write_patches(tmp_path, fields)
    cfg = tiny_config(
        data_source="patches",
        data_dir=str(tmp_path),
        time_range_start=0,
        nonuniform_max_span_hours=40,
    )
    cfg.variables = _variables(cfg, ["ssh_field", "sst_field"])
    ds = PatchDataset(cfg)
    assert ds.patch_ids == ["000", "001"]
    sample = ds[0]
    assert sample["target"].shape == (2, 3, 256)
    offsets = sample["time_offsets"]
    assert offsets[0] == 0 and (offsets.diff() >= 1).all() and offsets[-1] <= 40

    # The target is the normalized store content at the drawn frames.
    ds = PatchDataset(OmegaConf.merge(cfg, {"patch_ids": ["000"]}))
    sample = ds[0]
    offsets = sample["time_offsets"].numpy().astype(int)
    ssh = xr.open_zarr(tmp_path / "ssh_field" / "000.zarr").SSH.values
    sst = xr.open_zarr(tmp_path / "sst_field" / "000.zarr").SST
    first = sample["target"][0, 0].numpy() * 0.051
    start = [
        t for t in range(len(ssh)) if np.allclose(ssh[t].ravel(), first, atol=1e-6)
    ]
    assert len(start) == 1
    frames = start[0] + offsets
    norm = Normalizer.from_config(cfg)
    times = sst.time.values[frames]
    expect = norm.normalize("SST", sst.values[frames], times).reshape(3, -1)
    np.testing.assert_allclose(sample["target"][1].numpy(), expect, atol=1e-5)


def test_uniform_stride(tmp_path):
    _write_patches(tmp_path, [("ssh_field", "SSH", 0.1)])
    cfg = tiny_config(
        data_source="patches",
        data_dir=str(tmp_path),
        time_range_start=0,
        time_sampling_mode="uniform_stride",
        llc_dtime_choices=[12],
    )
    cfg.variables = _variables(cfg, ["ssh_field"])
    sample = PatchDataset(cfg)[0]
    assert sample["time_offsets"].tolist() == [0.0, 12.0, 24.0]


def test_missing_values_are_rejected(tmp_path):
    _write_patches(tmp_path, [("ssh_field", "SSH", 0.1)], nan_at=(slice(None), 3, 4))
    cfg = tiny_config(
        data_source="patches",
        data_dir=str(tmp_path),
        time_range_start=0,
        nonuniform_max_span_hours=40,
    )
    cfg.variables = _variables(cfg, ["ssh_field"])
    with pytest.raises(ValueError, match="NaN"):
        PatchDataset(cfg)[0]


def test_structured_fields_are_what_they_claim():
    rng = np.random.default_rng(0)
    offsets = np.array([0, 24, 48])
    f = structured_fields(
        (32, 32), offsets, 2, rng, correlation=0.8, speed_cells_per_day=4.0
    )
    assert f.shape == (2, 3, 32, 32)
    # Many draws: unit variance and the requested correlation.
    draws = np.stack(
        [
            structured_fields((32, 32), [0], 2, np.random.default_rng(i))
            for i in range(200)
        ]
    )
    assert abs(draws.var() - 1) < 0.1
    a, b = draws[:, 0].ravel(), draws[:, 1].ravel()
    assert abs(np.corrcoef(a, b)[0, 1] - 0.8) < 0.05
    # Drift is an exact shift: 4 cells/day for 24 h moves features 4 cells.
    shifted = [
        np.roll(f[0, 0], (dy, dx), axis=(0, 1))
        for dy in range(-5, 6)
        for dx in range(-5, 6)
    ]
    best = max(np.corrcoef(s.ravel(), f[0, 1].ravel())[0, 1] for s in shifted)
    assert best > 0.95


def test_structured_dataset(tiny_cfg):
    cfg = OmegaConf.merge(tiny_cfg, {"data_source": "structured"})
    ds = StructuredDataset(cfg)
    a, b = ds[5], ds[5]
    assert a["target"].shape == (2, 3, 256)
    np.testing.assert_array_equal(a["target"], b["target"])  # reproducible
    val = StructuredDataset(cfg, split=1)[5]
    assert not np.allclose(val["target"], a["target"])  # held-out stream


def test_synthetic(tiny_cfg):
    sample = SyntheticDataset(tiny_cfg)[3]
    assert sample["target"].shape == (2, 3, 256)
    assert "condition" in sample and "class_labels" in sample
