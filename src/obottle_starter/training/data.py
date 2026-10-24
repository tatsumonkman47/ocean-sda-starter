# SPDX-License-Identifier: Apache-2.0
"""Training datasets.

``data_source`` in the training config picks one:

* ``patches`` (:class:`PatchDataset`): the per-patch layout the released
  models were trained on::

      {data_dir}/{field}/{patch_id}.zarr

  one directory per variable (its ``field``) holding one zarr store per
  patch. Each store has a data variable on ``(time, row, col)`` (the
  variable's ``source_variable``, or the store's only one), an hourly
  ``time`` coordinate, and 2-D ``latitude`` / ``longitude``. Rows run south
  to north and columns west to east (the original stores label these dims
  ``x`` and ``y``; only the order matters). Every field must have the same
  patch ids and times. Values must be finite: the loss has no missing-data
  handling.
* ``structured`` (:class:`StructuredDataset`): smooth random fields that
  drift across the patch, with a known correlation between variables. For
  teaching and for checks that need spatial structure, without data files.
* ``synthetic`` (:class:`SyntheticDataset`): white noise, only to check that
  the code runs.

Each sample is a window of ``cfg.time_length`` frames. Frame offsets follow
``cfg.time_sampling_mode`` (:func:`draw_offsets`):

* ``uniform_stride``: equally spaced by a stride drawn from
  ``cfg.llc_dtime_choices`` (hours);
* ``nonuniform_offsets``: a maximum span drawn uniformly from
  ``[min*(T-1), nonuniform_max_span_hours]``, then ``T-1`` distinct
  offsets on the ``nonuniform_minimum_offset_hours`` grid inside it. The
  last frame lands anywhere inside the drawn span, so the actual
  first-to-last span is usually shorter than the drawn one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import xarray as xr

from obottle_starter.core.conditioning import frame_conditioning
from obottle_starter.core.config import num_channels, variables
from obottle_starter.core.normalization import Normalizer


def _drop_none(sample: dict) -> dict:
    # default_collate cannot batch None.
    return {k: v for k, v in sample.items() if v is not None}


def draw_offsets(cfg, rng: np.random.Generator) -> np.ndarray:
    """Frame offsets in hours from frame 0, shape ``(T,)`` (see module doc)."""
    T = int(cfg.time_length)
    if T == 1:
        return np.zeros(1, dtype=int)
    if cfg.get("time_sampling_mode", "uniform_stride") == "uniform_stride":
        stride = int(rng.choice(list(cfg.llc_dtime_choices)))
        return stride * np.arange(T)
    step = int(cfg.nonuniform_minimum_offset_hours)
    max_span = int(cfg.nonuniform_max_span_hours)
    max_span -= max_span % step
    spans = np.arange(step * (T - 1), max_span + 1, step)
    span = int(rng.choice(spans))
    grid = np.arange(step, span + 1, step)
    picks = rng.choice(grid, size=T - 1, replace=False)
    return np.concatenate([[0], np.sort(picks)])


class PatchDataset(torch.utils.data.Dataset):
    """Random windows from per-patch zarr stores, normalized as in training.

    Args:
        cfg: Training config (see ``configs/train_base.yaml``).
        normalizer: Defaults to ``Normalizer.from_config(cfg)``.
        length: Nominal epoch length; windows are drawn at random anyway.
        time_range: ``(start, end)`` hourly indices to draw frames from;
            defaults to ``cfg.time_range_start`` / ``time_range_end``.
        seed: Fixed seed for the window draws (validation sets). ``None``
            seeds from torch, so each DataLoader worker differs.
    """

    def __init__(
        self,
        cfg,
        normalizer: Normalizer | None = None,
        length=100_000,
        time_range: tuple[int, int | None] | None = None,
        seed: int | None = None,
    ):
        self.cfg = cfg
        self.normalizer = normalizer or Normalizer.from_config(cfg)
        self.root = Path(cfg.data_dir)
        self.specs = variables(cfg)
        self.fields = [s["field"] for s in self.specs]
        if not all(self.fields):
            raise ValueError("every variable needs a 'field' to read patches")
        if cfg.get("patch_ids"):
            self.patch_ids = [str(p) for p in cfg.patch_ids]
        else:
            self.patch_ids = sorted(
                p.stem for p in (self.root / self.fields[0]).glob("*.zarr")
            )
        if not self.patch_ids:
            raise FileNotFoundError(f"No patches in {self.root / self.fields[0]}")
        self.length = int(length)
        if time_range is None:
            time_range = (cfg.get("time_range_start", 0), cfg.get("time_range_end"))
        self.time_range = time_range
        self.seed = seed
        self._stores: dict[tuple[str, str], xr.Dataset] = {}
        self._rng: np.random.Generator | None = None

    def __len__(self) -> int:
        return self.length

    def _rng_(self) -> np.random.Generator:
        # Created lazily so each DataLoader worker gets its own torch seed.
        if self._rng is None:
            seed = self.seed if self.seed is not None else torch.initial_seed()
            self._rng = np.random.default_rng(seed % 2**32)
        return self._rng

    def _store(self, field: str, pid: str) -> xr.Dataset:
        key = (field, pid)
        if key not in self._stores:
            self._stores[key] = xr.open_zarr(
                self.root / field / f"{pid}.zarr", consolidated=None, chunks=None
            )
        return self._stores[key]

    def _variable(self, c: int, pid: str) -> xr.DataArray:
        ds = self._store(self.fields[c], pid)
        name = self.specs[c]["source_variable"]
        if name is None:
            if len(ds.data_vars) != 1:
                raise ValueError(
                    f"{self.fields[c]}/{pid}.zarr has data variables "
                    f"{list(ds.data_vars)}; set source_variable for "
                    f"{self.specs[c]['name']}"
                )
            name = next(iter(ds.data_vars))
        return ds[name]

    def __getitem__(self, idx: int) -> dict:
        cfg, rng = self.cfg, self._rng_()
        pid = self.patch_ids[int(rng.integers(len(self.patch_ids)))]
        offsets = draw_offsets(cfg, rng)

        first = self._store(self.fields[0], pid)
        n_time = first.sizes["time"]
        lo = int(self.time_range[0] or 0)
        end = self.time_range[1]
        hi = min(int(end if end is not None else n_time), n_time) - int(offsets[-1])
        if hi <= lo:
            raise ValueError(f"Patch {pid} is too short for a {offsets[-1]} h window")
        frames = int(rng.integers(lo, hi)) + offsets
        times = first.time.values[frames]

        channels = []
        for c in range(len(self.fields)):
            var = self._variable(c, pid)
            if c and not np.array_equal(var.time.values[frames], times):
                raise ValueError(
                    f"{self.fields[c]}/{pid}.zarr has different times from "
                    f"{self.fields[0]}; every field must share the same times"
                )
            values = var.isel(time=frames).values  # reads only the touched chunks
            if not np.isfinite(values).all():
                raise ValueError(
                    f"{self.fields[c]}/{pid}.zarr has NaN/inf at frames "
                    f"{frames.tolist()}. Training expects complete, finite "
                    "fields (docs/02_training.md, 'Missing data')"
                )
            channels.append(self.normalizer.normalize(c, values, times))
        target = torch.from_numpy(np.stack(channels)).reshape(
            len(self.fields), cfg.time_length, cfg.H * cfg.W
        )
        center_lon = float(np.nanmean(first.longitude.values))
        sample = frame_conditioning(times, center_lon, cfg)
        sample["target"] = target
        return _drop_none(sample)


def structured_fields(
    shape: tuple[int, int],
    offsets_hours,
    n_channels: int,
    rng: np.random.Generator,
    *,
    spectral_slope: float = 3.0,
    correlation: float = 0.8,
    speed_cells_per_day: float = 2.0,
) -> np.ndarray:
    """Smooth random fields that drift, ``(n_channels, T, H, W)``, unit variance.

    Each channel is a Gaussian random field with power spectrum
    ``(k² + k0²)^(-spectral_slope/2)`` (``k0`` = 1/16 cycles per cell), so
    larger slopes are smoother. All channels drift together at
    ``speed_cells_per_day`` in one random direction, exactly (a periodic
    shift), so frame ``t`` is frame 0 moved by ``speed * t``. Channel 0 is
    the base field; channel ``c > 0`` is
    ``correlation * base + sqrt(1 - correlation²) * independent``, so every
    later channel has correlation ``correlation`` with the first at every
    scale. Fields are periodic, unlike the network's zero padding.
    """
    H, W = shape
    ky = np.fft.fftfreq(H)[:, None]
    kx = np.fft.fftfreq(W)[None, :]
    k0 = 1.0 / 16.0
    amp = (kx**2 + ky**2 + k0**2) ** (-spectral_slope / 4.0)
    amp /= np.sqrt(np.mean(amp**2))  # unit variance after ifft2(fft2(noise)*amp)
    angle = rng.uniform(0, 2 * np.pi)
    speed = speed_cells_per_day / 24.0  # cells per hour
    offsets = np.asarray(offsets_hours, dtype=float)
    dy = speed * np.sin(angle) * offsets
    dx = speed * np.cos(angle) * offsets
    phase = np.exp(
        -2j * np.pi * (ky[None] * dy[:, None, None] + kx[None] * dx[:, None, None])
    )

    def field():
        coeffs = np.fft.fft2(rng.standard_normal((H, W))) * amp
        return np.fft.ifft2(coeffs[None] * phase).real

    base = field()
    out = [base]
    rho = float(correlation)
    for _ in range(1, n_channels):
        out.append(rho * base + np.sqrt(1.0 - rho**2) * field())
    return np.stack(out).astype(np.float32)


class StructuredDataset(torch.utils.data.Dataset):
    """Drifting smooth random fields with a known cross-variable correlation.

    Targets are :func:`structured_fields` in model units, so a variable's
    physical value is ``scale * x + offset`` for its normalization. Settings
    come from ``cfg.structured``. Sample ``idx`` is the same on every call;
    ``split`` 1 (validation) draws different samples from split 0.
    """

    def __init__(self, cfg, length: int = 100_000, split: int = 0):
        self.cfg = cfg
        self.length = int(length)
        self.split = int(split)
        self.settings = dict(cfg.get("structured") or {})

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict:
        cfg = self.cfg
        rng = np.random.default_rng([int(cfg.seed), self.split, int(idx)])
        offsets = draw_offsets(cfg, rng)
        start = np.datetime64("2012-01-01T00") + np.timedelta64(
            int(rng.integers(0, 365 * 24)), "h"
        )
        times = start + offsets.astype("timedelta64[h]")
        fields = structured_fields(
            (cfg.H, cfg.W), offsets, num_channels(cfg), rng, **self.settings
        )
        sample = frame_conditioning(times, 0.0, cfg)
        sample["target"] = torch.from_numpy(fields).reshape(
            num_channels(cfg), cfg.time_length, cfg.H * cfg.W
        )
        return _drop_none(sample)


class SyntheticDataset(torch.utils.data.Dataset):
    """White-noise targets with random frame times, for smoke tests."""

    def __init__(self, cfg, length: int = 1000, split: int = 0):
        self.cfg = cfg
        self.length = int(length)
        self.split = int(split)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict:
        cfg = self.cfg
        gen = torch.Generator().manual_seed(int(idx) + self.split * 1_000_003)
        hours = torch.randperm(72, generator=gen)[: cfg.time_length - 1] + 1
        offsets = np.concatenate([[0], np.sort(hours.numpy())])
        times = np.datetime64("2012-01-01T00") + offsets.astype("timedelta64[h]")
        sample = frame_conditioning(times, 0.0, cfg)
        sample["target"] = torch.randn(
            num_channels(cfg), cfg.time_length, cfg.H * cfg.W, generator=gen
        )
        return _drop_none(sample)


def build_dataset(cfg) -> torch.utils.data.Dataset:
    """The training dataset ``cfg.data_source`` selects."""
    source = cfg.get("data_source")
    if source == "patches":
        return PatchDataset(cfg)
    if source == "structured":
        return StructuredDataset(cfg)
    if source == "synthetic":
        return SyntheticDataset(cfg)
    raise ValueError(
        f"data_source must be patches, structured or synthetic; got {source!r}"
    )


def validation_samples(cfg) -> list[dict]:
    """A fixed list of ``cfg.validation.n_samples`` held-out samples.

    Drawn once with a fixed seed, so every validation pass sees the same
    windows. For ``patches`` they come from the held-out hours
    ``validation.time_range_start`` to ``time_range_end``; for the generated
    sources, from a separate stream of samples.
    """
    val = cfg.validation
    n = int(val.n_samples)
    source = cfg.get("data_source")
    if source == "patches":
        ds = PatchDataset(
            cfg,
            time_range=(val.time_range_start, val.time_range_end),
            seed=int(cfg.seed) + 12345,
        )
    elif source == "structured":
        ds = StructuredDataset(cfg, split=1)
    else:
        ds = SyntheticDataset(cfg, length=n, split=1)
    return [ds[i] for i in range(n)]
