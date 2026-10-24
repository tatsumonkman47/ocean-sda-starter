#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Copy a few patches (and optionally a time range) of the per-patch archive.

    python scripts/maintainer/make_training_subset.py SRC_DIR DST_DIR \
        --fields FULL_PACIFIC_zarr_llc4320_SSH_minus_300km_filtered_fullt \
        --patches 000 050 100 --hours 0 2000

Writes DST_DIR/{field}/{patch}.zarr as float32, in the same layout
PatchDataset reads.
"""

import argparse
from pathlib import Path

import numpy as np
import xarray as xr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src")
    parser.add_argument("dst")
    parser.add_argument("--fields", nargs="+", required=True)
    parser.add_argument("--patches", nargs="+", required=True)
    parser.add_argument("--hours", nargs=2, type=int, default=None)
    args = parser.parse_args()

    for field in args.fields:
        for pid in args.patches:
            ds = xr.open_zarr(Path(args.src) / field / f"{pid}.zarr", consolidated=None)
            if args.hours:
                ds = ds.isel(time=slice(*args.hours))
            for name in ds.data_vars:
                ds[name] = ds[name].astype(np.float32)
            ds = ds.chunk({"time": 100})
            for name in ds.variables:
                ds[name].encoding = {}
            dst = Path(args.dst) / field / f"{pid}.zarr"
            dst.parent.mkdir(parents=True, exist_ok=True)
            ds.to_zarr(dst, mode="w", zarr_format=2, consolidated=True)
            print(f"{dst}: {dict(ds.sizes)}")


if __name__ == "__main__":
    main()
