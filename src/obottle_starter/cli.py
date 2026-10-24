# SPDX-License-Identifier: Apache-2.0
"""Command-line config loading shared by the scripts."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from omegaconf import DictConfig, OmegaConf

from obottle_starter.core.config import ConfigError, check_unknown_keys


def load_config(default_yaml: str, argv: Sequence[str] | None = None) -> DictConfig:
    """Merge ``default_yaml``, optional ``config=other.yaml`` files, then overrides.

    Overrides use dot syntax: ``key=value``, ``a.b=1``, ``observe='[SSH]'``,
    and list items by index, ``variables.0.normalization.scale=0.05``.
    Options that ``default_yaml`` does not define are rejected, so a typo
    fails instead of being ignored.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    base = OmegaConf.load(default_yaml)
    cfg = base.copy()
    for item in [a for a in argv if a.startswith("config=")]:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(item.split("=", 1)[1]))
    for item in [a for a in argv if not a.startswith("config=")]:
        if "=" not in item:
            raise SystemExit(f"Overrides look like key=value; got {item!r}")
        key, value = item.split("=", 1)
        # Parse the value as YAML (numbers, lists, null, ...).
        parsed = OmegaConf.from_dotlist([f"value={value}"]).value
        OmegaConf.update(cfg, key, parsed, merge=True)
    try:
        check_unknown_keys(cfg, base, what=f"options (defaults: {default_yaml})")
    except ConfigError as err:
        raise SystemExit(str(err)) from None
    missing = sorted(OmegaConf.missing_keys(cfg))
    if missing:
        raise SystemExit(f"Missing required options: {', '.join(missing)}")
    return cfg
