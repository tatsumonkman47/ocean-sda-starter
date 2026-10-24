# SPDX-License-Identifier: Apache-2.0
"""OSSE test tiles: llc4320 truth plus real-data observation coverage.

Each tile is one netCDF file (see ``scripts/export_osse_tiles.py``) on
``(time, row, col)``: rows run south to north and columns west to east, on a
4 km local azimuthal-equidistant grid around ``center_lat``/``center_lon``.

* ``SSH`` (m): llc4320 sea-surface height with scales longer than 300 km
  removed, the field the SSH channel was trained on.
* ``SST`` (°C): llc4320 sea-surface temperature.
* ``ssh_mask``, ``sst_mask`` (0/1): where real instruments saw the ocean.
  SSH coverage is SWOT KaRIn swaths plus Copernicus nadir altimeters, SST
  coverage is ACSPO L3S. They are donor footprints from 2023-2025 mapped
  onto the 2011 model frames (``ssh_mask_time`` / ``sst_mask_time`` give the
  donor times); the observed values always come from the truth.

:func:`load_window` cuts a model-sized window out of a tile.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

MASK_VARS = {"SSH": "ssh_mask", "SST": "sst_mask"}


@dataclass
class Window:
    """A window of an OSSE tile, ready for SDA.

    Attributes:
        truth: ``{channel: (T, H, W)}`` truth in physical units.
        masks: ``{channel: (T, H, W) bool}`` real-instrument coverage.
        times: ``(T,)`` frame times.
        latitude, longitude: ``(H, W)`` cell centres.
        center_lon: Window-centre longitude (for local solar time).
        frames: Tile frame index of each window frame.
        origin: ``(row, col)`` of the window's first cell in the tile.
    """

    truth: dict[str, np.ndarray]
    masks: dict[str, np.ndarray]
    times: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    center_lon: float
    frames: np.ndarray
    origin: tuple[int, int]


def open_tile(path: str | Path) -> xr.Dataset:
    """Open an exported OSSE tile."""
    return xr.open_dataset(path)


def load_window(
    tile: xr.Dataset | str | Path,
    channels: Sequence[str],
    n_frames: int,
    *,
    t_start: int = 0,
    stride: int = 1,
    size: int = 128,
    origin: Sequence[int] | None = None,
) -> Window:
    """Cut ``n_frames`` frames of a ``size x size`` window out of a tile.

    Args:
        tile: Tile dataset or path.
        channels: Channels to return, e.g. ``["SSH", "SST"]``.
        n_frames: Number of frames (the model's ``time_length``).
        t_start: First tile frame.
        stride: Take every ``stride``-th tile frame (tiles are 12-hourly).
        size: Window size in cells (the model's ``H`` and ``W``).
        origin: ``(row, col)`` of the window's first cell; centred if ``None``.
    """
    ds = open_tile(tile) if not isinstance(tile, xr.Dataset) else tile
    n_row, n_col = ds.sizes["row"], ds.sizes["col"]
    if origin is None:
        origin = ((n_row - size) // 2, (n_col - size) // 2)
    r0, c0 = (int(v) for v in origin)
    if not (0 <= r0 <= n_row - size and 0 <= c0 <= n_col - size):
        raise ValueError(f"origin {origin} puts the window outside the tile")
    frames = t_start + stride * np.arange(n_frames)
    if frames[-1] >= ds.sizes["time"]:
        raise ValueError(
            f"Frames {frames.tolist()} exceed the tile's {ds.sizes['time']} frames"
        )
    sel = dict(time=frames, row=slice(r0, r0 + size), col=slice(c0, c0 + size))
    win = ds.isel(sel)
    truth = {c: win[c].values.astype(np.float32) for c in channels}
    masks = {c: win[MASK_VARS[c]].values > 0 for c in channels}
    return Window(
        truth=truth,
        masks=masks,
        times=win.time.values,
        latitude=win.latitude.values,
        longitude=win.longitude.values,
        center_lon=float(win.longitude.values[size // 2, size // 2]),
        frames=frames,
        origin=(r0, c0),
    )


def random_masks(
    channels: Sequence[str],
    shape: tuple[int, int, int],
    coverage,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Independent Bernoulli masks; ``coverage`` is a float or ``{channel: float}``."""
    rng = np.random.default_rng(seed)
    out = {}
    for c in channels:
        p = coverage[c] if isinstance(coverage, dict) else coverage
        out[c] = rng.random(shape) < float(p)
    return out
