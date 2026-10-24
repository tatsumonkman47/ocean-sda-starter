#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Pack cBottle-for-ocean-DA OSSE tiles into one netCDF per region.

    python scripts/maintainer/export_osse_tiles.py SRC_DIR DST.nc

SRC_DIR holds SSH_llc4320.nc, SST_llc4320.nc, SSH_mask_SWOT_CMEMS.nc and
SST_mask_ACSPO.nc (and optionally provenance.json), all on
(time, y, x) with y south->north. The output is on (time, row, col) with the
masks as 0/1 bytes and the donor observation times kept as coordinates.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src")
    parser.add_argument("dst")
    parser.add_argument("--center-lat", type=float)
    parser.add_argument("--center-lon", type=float)
    args = parser.parse_args()
    src = Path(args.src)

    def first(name):
        ds = xr.open_dataset(src / name)
        return ds[next(iter(ds.data_vars))], ds

    ssh, ssh_ds = first("SSH_llc4320.nc")
    sst, _ = first("SST_llc4320.nc")
    ssh_mask, _ = first("SSH_mask_SWOT_CMEMS.nc")
    sst_mask, _ = first("SST_mask_ACSPO.nc")
    lat, lon = ssh_ds.latitude.values, ssh_ds.longitude.values
    if not (np.diff(lat[:, 0]) > 0).all():
        raise ValueError("expected rows to run south to north")

    dims = ("time", "row", "col")
    out = xr.Dataset(
        {
            "SSH": (
                dims,
                ssh.values.astype(np.float32),
                {"units": "m", "long_name": "llc4320 SSH, >300 km scales removed"},
            ),
            "SST": (
                dims,
                sst.values.astype(np.float32),
                {"units": "degC", "long_name": "llc4320 SST"},
            ),
            "ssh_mask": (
                dims,
                (np.nan_to_num(ssh_mask.values) > 0).astype(np.uint8),
                {"long_name": "SWOT KaRIn + Copernicus nadir coverage"},
            ),
            "sst_mask": (
                dims,
                (np.nan_to_num(sst_mask.values) > 0).astype(np.uint8),
                {"long_name": "ACSPO L3S SST coverage"},
            ),
        },
        coords={
            "time": ssh.time.values,
            "latitude": (("row", "col"), lat.astype(np.float32)),
            "longitude": (("row", "col"), lon.astype(np.float32)),
            "x_m": ("col", ssh_ds.x.values.astype(np.float32)),
            "y_m": ("row", ssh_ds.y.values.astype(np.float32)),
            "ssh_mask_time": ("time", ssh_mask.time.values),
            "sst_mask_time": ("time", sst_mask.time.values),
        },
    )
    for name in ("SSH", "SST"):
        if np.isnan(out[name].values).any():
            raise ValueError(f"{name} has NaNs; the starter assumes ocean-only tiles")
    prov = src / "provenance.json"
    meta = json.loads(prov.read_text()) if prov.exists() else {}
    out.attrs.update(
        center_lat=args.center_lat
        if args.center_lat is not None
        else meta.get("center_lat"),
        center_lon=args.center_lon
        if args.center_lon is not None
        else meta.get("center_lon"),
        grid="4 km local azimuthal equidistant; rows south->north, cols west->east",
        note=(
            "Masks are real-instrument footprints from 2023-2025 (see *_mask_time) "
            "mapped onto the 2011 model frames; observed values come from the truth."
        ),
        source=str(src),
        provenance=json.dumps(meta),
    )
    enc = {v: {"zlib": True, "complevel": 4} for v in out.data_vars}
    Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(args.dst, encoding=enc)
    print(
        f"{args.dst}: {dict(out.sizes)}, SSH coverage {out.ssh_mask.mean().item():.3f}, "
        f"SST coverage {out.sst_mask.mean().item():.3f}"
    )


if __name__ == "__main__":
    main()
