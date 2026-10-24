# SPDX-License-Identifier: Apache-2.0
"""Per-channel normalization between physical units and model units.

The network is trained, and SDA is solved, in *model units*. The high-level
interfaces (:func:`obottle_starter.assimilate`, the scripts) take and return
*physical units* (m, °C) and convert with a :class:`Normalizer`; the
low-level SDA classes in :mod:`obottle_starter.inference.sda` work in
whatever units their inputs declare, normally model units.

Each channel's transform is affine,
``x_model = (x - climatology(t) - offset) / scale``, set by the
``normalization`` entry of its variable (:mod:`obottle_starter.core.config`)::

    normalization:
      scale: 5.5               # physical units per model unit
      offset: 0.0              # constant removed (after the climatology)
      climatology:             # optional time-dependent offset
        path: null             # .nc with a daily 1-D time series; null = data/SST_NP_daily_climatology.nc
        variable: SST          # data variable in that file
        lookup: day_of_year    # day_of_year | elapsed_days (see Climatology)
        out_of_range: error    # elapsed_days only: error | clip
        sha256: null           # filled in at training; checked when loaded

Where the numbers come from
---------------------------
Training writes its config into every checkpoint, so by default inference
uses exactly the normalization the model was trained with
(:meth:`Normalizer.from_config`). ``configs/sda.yaml`` can adjust the
climatology lookup or, for experiments, replace the normalization
(:meth:`Normalizer.from_settings`); any difference from training is reported.

Why SDA works in model units
----------------------------
MMPS solves a linear system with a few Krylov (GMRES/CG) iterations per
step. The exact solution does not depend on units, but a truncated solve
reduces the largest residuals first. In physical units an SST residual of
several °C dwarfs an SSH residual of centimetres, and with 4 iterations the
SSH observations are effectively ignored. In model units both channels have
residuals of order one, so :meth:`Normalizer.observation_terms` and
:meth:`Normalizer.sigma_to_model` express observations and their errors
there.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from obottle_starter.core.config import (
    channel_names,
    validate_variables,
    variables,
)

SOURCES = ("from_checkpoint", "config")


# src/obottle_starter/core/normalization.py -> repository root
REPO_DATA = Path(__file__).resolve().parents[3] / "data"


def default_climatology_path() -> str:
    """The SST climatology in the repository's ``data/`` directory.

    Found relative to this file, so it works from a clone or an editable
    install (``pip install -e .``). Elsewhere, set the climatology ``path``.
    """
    return str(REPO_DATA / "SST_NP_daily_climatology.nc")


def file_sha256(path: str | Path) -> str:
    """SHA-256 of a file, to record which climatology a model used."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Climatology:
    """A daily climatology, looked up at arbitrary times.

    Values are piecewise constant over each UTC day; there is no
    interpolation between days. Two lookup rules:

    * ``elapsed_days``: whole days since the file's first day. This is the
      rule the released models were trained with; it only makes sense for
      dates inside the file. ``out_of_range="error"`` (default) rejects
      other dates; ``"clip"`` uses the first or last day instead, which is
      what the original training code did silently.
    * ``day_of_year``: the file's value for the same calendar day
      (month and day), so any year works. A calendar day that appears more
      than once in the file (e.g. a 377-day record) uses the mean of its
      values. Feb 29 uses the file's Feb 29 if it has one, otherwise the
      mean of Feb 28 and Mar 1. Every other calendar day must be present.

    Args:
        path: netCDF file with a 1-D daily ``time`` series; ``None`` = the
            repository's SST climatology.
        variable: Data variable in the file.
        lookup: ``"elapsed_days"`` or ``"day_of_year"``.
        out_of_range: ``"error"`` or ``"clip"`` (``elapsed_days`` only).
        sha256: Expected checksum; checked when the file is first read.
    """

    def __init__(
        self,
        path: str | None = None,
        variable: str = "SST",
        lookup: str = "day_of_year",
        out_of_range: str = "error",
        sha256: str | None = None,
    ) -> None:
        if lookup not in ("elapsed_days", "day_of_year"):
            raise ValueError("climatology lookup must be elapsed_days or day_of_year")
        if out_of_range not in ("error", "clip"):
            raise ValueError("climatology out_of_range must be error or clip")
        self.path = path or default_climatology_path()
        self.variable = variable
        self.lookup = lookup
        self.out_of_range = out_of_range
        self.sha256 = sha256
        self._times: np.ndarray | None = None
        self._values: np.ndarray | None = None
        self._by_day: np.ndarray | None = None

    def _load(self) -> None:
        if self._values is not None:
            return
        if not Path(self.path).exists():
            raise FileNotFoundError(
                f"climatology not found at {self.path}; set the variable's "
                "normalization.climatology.path (training config, or sda.yaml "
                "normalization.climatology.path)"
            )
        if self.sha256 and file_sha256(self.path) != self.sha256:
            raise ValueError(
                f"climatology {self.path} is not the file this model was trained "
                f"with (sha256 differs from {self.sha256}); point "
                "climatology.path at the original file"
            )
        with xr.open_dataset(self.path) as ds:
            da = ds[self.variable].load()
        self._times = da.time.values.astype("datetime64[ns]")
        self._values = da.values.astype(np.float32)
        if self.lookup == "day_of_year":
            self._by_day = self._calendar_table()

    def _calendar_table(self) -> np.ndarray:
        month, day = _month_day(self._times)
        table = np.full((13, 32), np.nan)
        for m, d in set(zip(month.tolist(), day.tolist())):
            table[m, d] = self._values[(month == m) & (day == d)].mean()
        if np.isnan(table[2, 29]):
            table[2, 29] = 0.5 * (table[2, 28] + table[3, 1])
        days = np.arange("2001-01-01", "2002-01-01", dtype="datetime64[D]")
        missing = [
            str(t)[5:]
            for t, m, d in zip(days, *_month_day(days))
            if np.isnan(table[m, d])
        ]
        if missing:
            raise ValueError(
                f"day_of_year lookup needs every calendar day in {self.path}; "
                f"missing {missing[:5]}{'...' if len(missing) > 5 else ''}"
            )
        return table.astype(np.float32)

    def __call__(self, times) -> np.ndarray:
        """Climatology value at each of ``times``, shape ``(T,)``."""
        self._load()
        times = np.atleast_1d(np.asarray(times, dtype="datetime64[ns]"))
        if self.lookup == "day_of_year":
            month, day = _month_day(times)
            return self._by_day[month, day]
        days = ((times - self._times[0]) // np.timedelta64(1, "D")).astype(int)
        n = len(self._values)
        outside = (days < 0) | (days >= n)
        if outside.any() and self.out_of_range == "error":
            first = str(self._times[0])[:10]
            last = str(self._times[-1])[:10]
            raise ValueError(
                f"times {[str(t)[:16] for t in times[outside][:3]]} are outside "
                f"the climatology record {first} to {last} ({self.path}). This "
                "model's normalization looks days up by position in that record "
                "(lookup: elapsed_days). Choose deliberately: "
                "climatology.lookup=day_of_year (same calendar day, any year) or "
                "climatology.out_of_range=clip (first/last day, as the original "
                "training code did silently)"
            )
        return self._values[np.clip(days, 0, n - 1)]

    def settings(self) -> dict:
        return {
            "path": self.path,
            "variable": self.variable,
            "lookup": self.lookup,
            "out_of_range": self.out_of_range,
            "sha256": self.sha256,
        }

    def describe(self) -> str:
        rule = self.lookup
        if self.lookup == "elapsed_days":
            rule += f", out of range: {self.out_of_range}"
        return f"daily {self.variable} from {self.path} ({rule})"


def _month_day(times) -> tuple[np.ndarray, np.ndarray]:
    d = np.asarray(times).astype("datetime64[D]")
    months = d.astype("datetime64[M]")
    month = months.astype(int) % 12 + 1
    day = (d - months.astype("datetime64[D]")).astype(int) + 1
    return month, day


class Normalizer:
    """Affine per-channel transforms between physical and model units.

    Args:
        specs: One variable spec per channel, in model order (the
            ``variables`` list of a config, see
            :mod:`obottle_starter.core.config`).
        source: Where these settings came from, for :meth:`describe`.
    """

    def __init__(self, specs: Sequence[Mapping], source: str = "") -> None:
        validate_variables(specs)
        self.specs = [dict(s) for s in variables({"variables": list(specs)})]
        self.channels = [s["name"] for s in self.specs]
        self.source = source
        self._climatologies = {
            s["name"]: Climatology(**s["normalization"]["climatology"])
            for s in self.specs
            if s["normalization"]["climatology"] is not None
        }

    # ── Construction ──────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, cfg, climatology: Mapping | None = None) -> Normalizer:
        """The normalization a training config (or checkpoint) specifies.

        ``climatology`` overrides entries of every climatology (``path``,
        ``lookup``, ``out_of_range``); ``None`` values keep the config's.
        """
        specs = variables(cfg)
        overrides = {k: v for k, v in (climatology or {}).items() if v is not None}
        for s in specs:
            clim = s["normalization"]["climatology"]
            if clim is not None:
                # A new path keeps the recorded sha256, so a moved copy of
                # the training file passes and a different file is rejected.
                clim.update(overrides)
        return cls(specs, source="checkpoint")

    @classmethod
    def from_settings(cls, cfg, settings: Mapping | None) -> Normalizer:
        """Normalization chosen by the ``normalization`` section of sda.yaml.

        ``source: from_checkpoint`` (default) uses the checkpoint's own
        normalization, with any ``climatology`` adjustments. ``source:
        config`` uses ``settings["variables"]`` instead. Either way,
        differences from training are printed as a warning.
        """
        settings = dict(settings or {})
        source = settings.get("source", "from_checkpoint")
        if source not in SOURCES:
            raise ValueError(f"normalization.source must be one of {SOURCES}")
        trained = cls.from_config(cfg)
        if source == "from_checkpoint":
            norm = cls.from_config(cfg, settings.get("climatology"))
            if norm.differences(trained):
                norm.source = "checkpoint, climatology adjusted by the sda config"
        else:
            specs = settings.get("variables")
            if not specs:
                raise ValueError("normalization.source=config needs variables")
            norm = cls(list(specs), source="sda config (overrides the checkpoint)")
            if norm.channels != channel_names(cfg):
                raise ValueError(
                    f"normalization.variables {norm.channels} must match the "
                    f"model's variables {channel_names(cfg)}, in order"
                )
        diffs = norm.differences(trained)
        if diffs:
            print(
                "WARNING: inference normalization differs from the one this "
                "model was trained with:\n  " + "\n  ".join(diffs)
            )
        return norm

    # ── The transform ─────────────────────────────────────────────────────

    def _index(self, channel: str | int) -> int:
        return channel if isinstance(channel, int) else self.channels.index(channel)

    def scale(self, channel: str | int) -> float:
        """Physical units per model unit."""
        return float(self.specs[self._index(channel)]["normalization"]["scale"])

    def units(self, channel: str | int) -> str:
        """Physical units of a channel."""
        return self.specs[self._index(channel)]["units"]

    def climatology(self, channel: str | int) -> Climatology | None:
        """The channel's climatology, or ``None``."""
        return self._climatologies.get(self.channels[self._index(channel)])

    def offset(self, channel: str | int, times) -> np.ndarray:
        """Per-frame offset subtracted before scaling, shape ``(T,)``."""
        c = self._index(channel)
        constant = np.float32(self.specs[c]["normalization"]["offset"])
        clim = self.climatology(c)
        if clim is not None:
            return clim(times) + constant
        return np.full(len(np.atleast_1d(times)), constant, dtype=np.float32)

    def normalize(self, channel: str | int, values, times) -> np.ndarray:
        """Physical ``(T, ...)`` values to model units."""
        values = np.asarray(values, dtype=np.float32)
        offset = self.offset(channel, times).reshape(-1, *[1] * (values.ndim - 1))
        return (values - offset) / np.float32(self.scale(channel))

    def denormalize(self, channel: str | int, values, times) -> np.ndarray:
        """Model-unit ``(..., T, H, W)`` values to physical units."""
        values = np.asarray(values, dtype=np.float32)
        offset = self.offset(channel, times).reshape(-1, 1, 1)
        return values * np.float32(self.scale(channel)) + offset

    def to_physical(self, x, times) -> np.ndarray:
        """``(..., C, T, H, W)`` model-unit array to physical units."""
        x = np.asarray(x, dtype=np.float32)
        out = np.empty_like(x)
        for c in range(len(self.channels)):
            out[..., c, :, :, :] = self.denormalize(c, x[..., c, :, :, :], times)
        return out

    # ── Observations ──────────────────────────────────────────────────────

    def sigma_to_model(self, channel: str | int, sigma) -> float:
        """An error std (or noise std) in physical units, in model units."""
        return float(sigma) / self.scale(channel)

    def observation_terms(self, masks: Mapping[str, object]) -> list:
        """One model-unit observation term per observed channel.

        Args:
            masks: ``{channel name: (T, H, W) or (T, H*W) bool mask}``.
                Channels without an entry, or with an empty mask, are
                unobserved.

        Returns:
            Unit-weight terms in model order: an observation is the
            normalized channel value at a masked pixel. Normalize observed
            values with :meth:`normalize` and errors with
            :meth:`sigma_to_model` to match. See the module docstring for
            why these are not physical-unit terms.
        """
        # Imported here so core/ does not depend on inference/ at import time.
        from obottle_starter.inference.sda.observations import ObservationTerm

        terms = []
        for name in self.channels:
            if name not in masks or masks[name] is None:
                continue
            mask = torch.as_tensor(np.asarray(masks[name]), dtype=torch.bool)
            if bool(mask.any()):
                terms.append(ObservationTerm(name, mask, {self._index(name): 1.0}))
        return terms

    # ── Reporting ─────────────────────────────────────────────────────────

    def formula(self, channel: str | int) -> str:
        """``x_model = ...`` for one channel, with the numbers filled in."""
        c = self._index(channel)
        scale = f"{self.scale(c):g} {self.units(c)}".rstrip()
        offset = float(self.specs[c]["normalization"]["offset"])
        shift = " - clim(t)" if self.climatology(c) is not None else ""
        if offset:
            shift += f" - {offset:g}"
        if shift:
            return f"x_model = (x{shift}) / {scale}"
        return f"x_model = x / {scale}"

    def describe(
        self, sigma_y: Mapping[str, float] | None = None, *, sda: bool = True
    ) -> str:
        """A readable summary; with ``sigma_y``, also its model-unit values.

        ``sda=False`` leaves out the note on SDA (for training logs).
        """
        lines = [f"normalization (from {self.source or 'arguments'}):"]
        for c, name in enumerate(self.channels):
            line = f"  {name}: {self.formula(c)}"
            if sigma_y and name in sigma_y:
                line += (
                    f"\n       sigma_y {sigma_y[name]:g} {self.units(c)}"
                    f" -> {self.sigma_to_model(c, sigma_y[name]):.4g} model units"
                )
            lines.append(line)
            if self.specs[c]["preprocessing"]:
                lines.append(f"       preprocessing: {self.specs[c]['preprocessing']}")
        for name, clim in self._climatologies.items():
            lines.append(f"  clim(t) for {name} = {clim.describe()}")
        if sda:
            lines.append("  SDA observations and sigma_y are solved in model units")
        return "\n".join(lines)

    def as_attrs(self) -> dict[str, str]:
        """Flat string attributes recording this normalization."""
        specs = [dict(s) for s in self.specs]
        for s in specs:
            if s["name"] in self._climatologies:
                s["normalization"] = dict(
                    s["normalization"],
                    climatology=self._climatologies[s["name"]].settings(),
                )
        return {
            "normalization_source": self.source,
            "normalization_formulas": "; ".join(
                f"{n}: {self.formula(c)}" for c, n in enumerate(self.channels)
            ),
            "normalization_variables": json.dumps(specs),
        }

    def differences(self, other: Normalizer) -> list[str]:
        """Human-readable differences from ``other`` (empty if equivalent)."""
        diffs = []
        for c, name in enumerate(self.channels):
            if self.formula(c) != other.formula(c):
                diffs.append(
                    f"{name}: {self.formula(c)}  (trained: {other.formula(c)})"
                )
            mine, theirs = self.climatology(c), other.climatology(c)
            if mine is not None and theirs is not None:
                for key in ("variable", "lookup", "out_of_range"):
                    if getattr(mine, key) != getattr(theirs, key):
                        diffs.append(
                            f"{name} climatology {key}: {getattr(mine, key)} "
                            f"(trained: {getattr(theirs, key)})"
                        )
                if mine.path != theirs.path and not theirs.sha256:
                    diffs.append(
                        f"{name} climatology file {mine.path} (trained: "
                        f"{theirs.path}; no checksum recorded to confirm they match)"
                    )
        return diffs
