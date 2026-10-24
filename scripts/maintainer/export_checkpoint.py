#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Strip a training checkpoint to config + EMA weights for sharing.

    python scripts/maintainer/export_checkpoint.py SRC.pt DST.pt \
        --channels SSH SST --description "joint SSH+SST, 5 frames"

Paths in the config that only exist on the training machine (data_dir,
resume_from, ...) are removed, and the config is saved in the current
format (a variables list). The weights are copied unchanged.
"""

import argparse

from obottle_starter.core.model import (
    channel_names,
    export_checkpoint,
    load_checkpoint,
)

LOCAL_KEYS = (
    "data_dir",
    "patch_coords_path",
    "resume_from",
    "output_dir",
    "conflict_free_lib_path",
    "wandb_id",
    "wandb_name",
    "wandb_project",
    "wandb_enabled",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src")
    parser.add_argument("dst")
    parser.add_argument(
        "--channels", nargs="+", help="names for an older flat config's channels"
    )
    parser.add_argument("--description", default="")
    args = parser.parse_args()

    saved = export_checkpoint(
        args.src,
        args.dst,
        extra={
            "description": args.description,
            "exported_from": args.src,
            **({"channels": args.channels} if args.channels else {}),
        },
        drop=LOCAL_KEYS,
    )
    net, cfg = load_checkpoint(args.dst, "cpu")  # must rebuild strictly
    n = sum(p.numel() for p in net.parameters())
    print(
        f"{args.dst}: step {saved['step']}, {n:,} parameters, "
        f"channels {channel_names(cfg)}, {cfg.time_length} frames"
    )


if __name__ == "__main__":
    main()
