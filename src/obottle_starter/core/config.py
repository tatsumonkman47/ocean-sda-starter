# SPDX-License-Identifier: Apache-2.0
"""Config checks and the variable list, shared by training and inference.

A config describes its model variables in one list (``configs/train_base.yaml``)::

    variables:
      - name: SSH
        field: FULL_PACIFIC_zarr_llc4320_SSH_minus_300km_filtered_fullt
        units: m
        preprocessing: llc4320 SSH, scales longer than 300 km removed
        normalization: {scale: 0.051, offset: 0.0, climatology: null}

The channel count and order come from this list. Checkpoints trained in
cBottle-for-ocean-DA (including the released ones) store the older flat keys
(``outfields``, ``channels``, ``out_transform_list``, ``llc_standards``,
``climatology_path``); :func:`variables` and :func:`upgrade` translate those,
so nothing else has to know about them.

The checks here run before any data is read or any network is built, and
say which option to change:

* :func:`validate_model`: grid, levels and attention settings.
* :func:`validate_training`: everything ``scripts/train.py`` reads.
* :func:`validate_sda`: everything ``scripts/run_sda.py`` reads, and whether
  it fits the checkpoint.
* :func:`check_unknown_keys`: misspelled options.
* :func:`run_differences`: what changed between a run and its checkpoint.
"""

from __future__ import annotations

import difflib
import math
from collections.abc import Mapping

from omegaconf import DictConfig, ListConfig, OmegaConf

CONFIG_VERSION = 2

DATA_SOURCES = ("patches", "synthetic", "structured")
START_MODES = ("fresh", "resume", "auto")
CLIMATOLOGY_LOOKUPS = ("elapsed_days", "day_of_year")
OUT_OF_RANGE = ("error", "clip")
NOISE_DISTRIBUTIONS = ("log_uniform", "log_normal", "power")
TIME_SAMPLING_MODES = ("nonuniform_offsets", "uniform_stride")
GUIDANCE_METHODS = ("mmps", "rozet", "conditional", "cfg")
SOLVERS = ("gmres", "cg", "minres")
TIMESTEP_SPACINGS = ("log_sigma", "karras")
OBSERVATION_MODES = ("instrument", "random", "none")

VARIABLE_KEYS = (
    "name",
    "field",
    "source_variable",
    "units",
    "preprocessing",
    "normalization",
)
NORMALIZATION_KEYS = ("scale", "offset", "climatology")
CLIMATOLOGY_KEYS = ("path", "variable", "lookup", "out_of_range", "sha256")

# Dicts under these keys map channel names to values, so their keys are not
# options and are not checked by check_unknown_keys.
PER_CHANNEL_KEYS = (
    "coverage",
    "stripe_width",
    "stripes_per_frame",
    "frame_probability",
    "noise_std",
    "random_coverage",
    "sigma_y",
)

# Options a resumed run may change. Everything else defines what the run
# learns (data, variables, normalization, network, noise, conditioning,
# optimizer), so changing it mid-run makes the result hard to interpret.
RUNTIME_KEYS = (
    "experiment_name",
    "output_dir",
    "data_dir",
    "num_workers",
    "num_steps",
    "log_every",
    "checkpoint_every",
    "start",
    "resume_from",
    "init_from",
    "validation",
)

# Settings that fix the network's parameter shapes and meaning.
ARCHITECTURE_KEYS = (
    "H",
    "W",
    "time_length",
    "model_channels",
    "channel_mult",
    "num_blocks",
    "attn_resolutions",
    "temporal_attn_resolutions",
    "decoder_start_temporal_attn",
    "upsample_temporal_attn",
    "use_per_frame_adagn",
    "per_frame_adaptive_scale",
    "calendar_embed_channels",
    "temporal_offset_channels",
    "label_dim",
    "span_label_channels",
    "mixing_type",
    "sigma_data",
)

# Legacy normalization transforms -> (scale key, offset key, units).
_LEGACY_TRANSFORMS = {
    "std_ssh_norm": ("std_ssh", None, "m"),
    "std_sst_norm": ("std_sst", None, "degC"),
    "std_global_mean_ssh_norm": ("std_ssh", "mean_ssh", "m"),
    "std_global_mean_sst_norm": ("std_sst", "mean_sst", "degC"),
    "std_seasonal_mean_sst_norm": ("std_sst", "extra_mean_tuning", "degC"),
    "std_uv_norm": ("std_uv", None, "m/s"),
}
_LEGACY_KEYS = (
    "channels",
    "outfields",
    "infields",
    "num_channels",
    "out_transform_list",
    "in_transform_list",
    "in_mask_list",
    "out_mask_list",
    "llc_standards",
    "climatology_path",
)


def _plain(value):
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


# ── Variables ─────────────────────────────────────────────────────────────


def _legacy_variables(cfg) -> list[dict]:
    """The variable list implied by the older flat keys."""
    transforms = list(cfg.get("out_transform_list") or [])
    standards = dict(_plain(cfg.get("llc_standards")) or {})
    fields = list(cfg.get("outfields") or [None] * len(transforms))
    names = list(cfg.get("channels") or [])
    if not names:
        for field in fields:
            upper = str(field).upper()
            names.append(next((s for s in ("SSH", "SST") if s in upper), str(field)))
    if not (len(transforms) == len(fields) == len(names)) or not transforms:
        raise ValueError(
            "config has neither a 'variables' list nor matching outfields / "
            "channels / out_transform_list"
        )
    out = []
    for name, field, transform in zip(names, fields, transforms):
        if transform not in _LEGACY_TRANSFORMS:
            raise ValueError(f"unknown legacy normalization transform {transform!r}")
        scale_key, offset_key, units = _LEGACY_TRANSFORMS[transform]
        climatology = None
        if transform == "std_seasonal_mean_sst_norm":
            # The rule the released models were trained with: whole days
            # since the file's first day (see Normalizer).
            climatology = {
                "path": cfg.get("climatology_path"),
                "variable": "SST",
                "lookup": "elapsed_days",
                "out_of_range": "error",
                "sha256": None,
            }
        out.append(
            {
                "name": str(name),
                "field": field,
                "source_variable": None,
                "units": units,
                "preprocessing": (
                    "scales longer than 300 km removed"
                    if field and "300km" in str(field)
                    else ""
                ),
                "normalization": {
                    "scale": float(standards[scale_key]),
                    "offset": float(standards.get(offset_key, 0.0))
                    if offset_key
                    else 0.0,
                    "climatology": climatology,
                },
            }
        )
    return out


def _fill_variable(spec) -> dict:
    spec = dict(_plain(spec))
    norm = dict(spec.get("normalization") or {})
    clim = norm.get("climatology")
    if clim is not None:
        clim = {
            "path": clim.get("path"),
            "variable": clim.get("variable", "SST"),
            "lookup": clim.get("lookup", "day_of_year"),
            "out_of_range": clim.get("out_of_range", "error"),
            "sha256": clim.get("sha256"),
        }
    return {
        "name": str(spec.get("name")),
        "field": spec.get("field"),
        "source_variable": spec.get("source_variable"),
        "units": str(spec.get("units") or ""),
        "preprocessing": str(spec.get("preprocessing") or ""),
        "normalization": {
            "scale": norm.get("scale"),
            "offset": norm.get("offset", 0.0),
            "climatology": clim,
        },
    }


def variables(cfg) -> list[dict]:
    """The config's variables as plain dicts with every key filled in.

    Works on current configs (a ``variables`` list) and on older checkpoint
    configs (flat lists, translated).
    """
    if cfg.get("variables") is not None:
        return [_fill_variable(v) for v in cfg.get("variables")]
    return _legacy_variables(cfg)


def channel_names(cfg) -> list[str]:
    """Variable names (``SSH``, ``SST``, ...) in model channel order."""
    return [v["name"] for v in variables(cfg)]


def num_channels(cfg) -> int:
    """Number of model variables."""
    return len(variables(cfg))


def upgrade(cfg) -> DictConfig:
    """A copy of ``cfg`` in the current format (``variables`` list).

    Older checkpoint configs keep their other keys; only the flat
    normalization and channel keys are replaced.
    """
    cfg = OmegaConf.create(_plain(cfg))
    if cfg.get("variables") is None:
        cfg.variables = _legacy_variables(cfg)
        for key in _LEGACY_KEYS:
            cfg.pop(key, None)
    cfg.config_version = CONFIG_VERSION
    return cfg


# ── Error collection ──────────────────────────────────────────────────────


class ConfigError(ValueError):
    """One or more invalid options; the message lists them all."""


class _Errors:
    def __init__(self) -> None:
        self.items: list[str] = []

    def add(self, message: str) -> None:
        self.items.append(message)

    def check(self, ok: bool, message: str) -> None:
        if not ok:
            self.add(message)

    def raise_if_any(self, what: str) -> None:
        if self.items:
            lines = "\n  - ".join(self.items)
            raise ConfigError(f"Invalid {what}:\n  - {lines}")


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _in_unit_interval(x) -> bool:
    return _is_number(x) and 0.0 <= x <= 1.0


# ── Unknown keys ──────────────────────────────────────────────────────────


def check_unknown_keys(cfg, base, what: str = "config") -> None:
    """Fail on keys of ``cfg`` that ``base`` (the defaults file) lacks.

    Catches misspelled options before they are silently ignored. Dicts under
    :data:`PER_CHANNEL_KEYS` hold channel names and are not checked; lists
    (such as ``variables``) are checked by the validators.
    """
    errors = _Errors()

    def walk(node, ref, prefix):
        known = list(ref.keys())  # `in ref` is False for required (???) keys
        for key in node:
            path = f"{prefix}{key}"
            if key not in known:
                close = difflib.get_close_matches(str(key), [str(k) for k in known], 1)
                hint = f" (did you mean {prefix}{close[0]}?)" if close else ""
                errors.add(f"unknown option {path}{hint}")
                continue
            if OmegaConf.is_missing(ref, key) or OmegaConf.is_missing(node, key):
                continue
            child, ref_child = node[key], ref[key]
            if (
                isinstance(child, DictConfig)
                and isinstance(ref_child, DictConfig)
                and key not in PER_CHANNEL_KEYS
            ):
                walk(child, ref_child, f"{path}.")

    walk(cfg, base, "")
    errors.raise_if_any(what)


# ── Variables and normalization ───────────────────────────────────────────


def _check_variables(specs, errors: _Errors, *, need_field: bool) -> None:
    if not specs:
        errors.add("variables: list at least one variable")
        return
    names = []
    for i, spec in enumerate(specs):
        where = f"variables[{i}]"
        if not isinstance(spec, (Mapping, DictConfig)):
            errors.add(f"{where}: expected a mapping with name, units, normalization")
            continue
        for key in spec:
            errors.check(
                key in VARIABLE_KEYS,
                f"{where}: unknown key {key!r} (allowed: {', '.join(VARIABLE_KEYS)})",
            )
        name = spec.get("name")
        errors.check(
            isinstance(name, str) and name != "", f"{where}: needs a string 'name'"
        )
        names.append(name)
        if need_field:
            errors.check(
                bool(spec.get("field")),
                f"{where} ({name}): 'field' (its directory under data_dir) is "
                "needed when data_source=patches",
            )
        norm = spec.get("normalization")
        if norm is None:
            errors.add(f"{where} ({name}): needs normalization: {{scale: ...}}")
            continue
        for key in norm:
            errors.check(
                key in NORMALIZATION_KEYS,
                f"{where}.normalization: unknown key {key!r} "
                f"(allowed: {', '.join(NORMALIZATION_KEYS)})",
            )
        scale = norm.get("scale")
        errors.check(
            _is_number(scale) and scale > 0,
            f"{where} ({name}): normalization.scale must be a positive number "
            f"(physical units per model unit); got {scale!r}",
        )
        offset = norm.get("offset", 0.0)
        errors.check(
            _is_number(offset),
            f"{where} ({name}): normalization.offset must be a finite number; "
            f"got {offset!r}",
        )
        clim = norm.get("climatology")
        if clim is not None:
            for key in clim:
                errors.check(
                    key in CLIMATOLOGY_KEYS,
                    f"{where}.normalization.climatology: unknown key {key!r} "
                    f"(allowed: {', '.join(CLIMATOLOGY_KEYS)})",
                )
            lookup = clim.get("lookup", "day_of_year")
            errors.check(
                lookup in CLIMATOLOGY_LOOKUPS,
                f"{where} ({name}): climatology.lookup must be one of "
                f"{CLIMATOLOGY_LOOKUPS}; got {lookup!r}",
            )
            oor = clim.get("out_of_range", "error")
            errors.check(
                oor in OUT_OF_RANGE,
                f"{where} ({name}): climatology.out_of_range must be one of "
                f"{OUT_OF_RANGE}; got {oor!r}",
            )
    dupes = sorted({n for n in names if names.count(n) > 1 and n})
    errors.check(not dupes, f"variables: duplicate names {dupes}")


def validate_variables(specs, *, need_field: bool = False) -> None:
    """Check a ``variables`` list on its own (e.g. sda.yaml's override)."""
    errors = _Errors()
    _check_variables(specs, errors, need_field=need_field)
    errors.raise_if_any("variables")


# ── Network ───────────────────────────────────────────────────────────────


def level_resolutions(cfg) -> list[int]:
    """Grid size at each U-Net level, finest first (e.g. 128, 64, 32, 16)."""
    return [int(cfg.H) >> level for level in range(len(cfg.channel_mult))]


def _check_model(cfg, errors: _Errors) -> None:
    H, W = cfg.get("H"), cfg.get("W")
    if not (_is_int(H) and _is_int(W) and H > 0 and W > 0):
        errors.add(f"H and W must be positive integers; got H={H!r}, W={W!r}")
        return
    errors.check(
        H == W,
        f"H={H} and W={W} differ, but this SongUNet builds square levels; use "
        "square patches (rectangular support would need (height, width) "
        "tracked through every level)",
    )
    mult = list(cfg.get("channel_mult") or [])
    errors.check(
        bool(mult) and all(_is_int(m) and m > 0 for m in mult),
        f"channel_mult must be a non-empty list of positive integers; got {mult}",
    )
    if not mult:
        return
    n_down = len(mult) - 1
    errors.check(
        H % (2**n_down) == 0,
        f"H={H} is not divisible by 2^{n_down}: channel_mult has {len(mult)} "
        f"levels, so the grid is halved {n_down} times. Use a patch size that "
        f"is a multiple of {2**n_down} or fewer levels",
    )
    available = level_resolutions(cfg)
    attn = list(cfg.get("attn_resolutions") or [])
    bad = [r for r in attn if r not in available]
    errors.check(
        not bad,
        f"attn_resolutions {bad} match no level; with H={H} and "
        f"{len(mult)} levels the grid sizes are {available}",
    )
    if int(cfg.get("time_length", 1)) > 1:
        tattn = list(cfg.get("temporal_attn_resolutions") or [])
        bad = [r for r in tattn if r not in available]
        errors.check(
            not bad,
            f"temporal_attn_resolutions {bad} match no level; the grid sizes "
            f"are {available}",
        )
    for key in ("time_length", "model_channels", "num_blocks"):
        value = cfg.get(key)
        errors.check(_is_int(value) and value >= 1, f"{key} must be an integer >= 1")
    for key in ("temporal_offset_channels", "span_label_channels"):
        value = cfg.get(key, 0)
        errors.check(
            _is_int(value) and value >= 0 and value % 2 == 0,
            f"{key} must be an even integer >= 0 (sin/cos pairs); got {value!r}",
        )
    for key in ("calendar_embed_channels", "label_dim"):
        value = cfg.get(key, 0)
        errors.check(_is_int(value) and value >= 0, f"{key} must be an integer >= 0")
    errors.check(
        cfg.get("mixing_type", "spatial") == "spatial",
        "mixing_type must be 'spatial' (the only layout these models use)",
    )
    errors.check(
        not cfg.get("per_frame_adaptive_scale", False)
        or cfg.get("use_per_frame_adagn", False),
        "per_frame_adaptive_scale needs use_per_frame_adagn: true",
    )
    errors.check(
        not cfg.get("condition_channels", 0),
        "condition_channels is a legacy key; keep it 0 and set up observation "
        "inputs with the conditioning section",
    )
    sd, lo, hi = cfg.get("sigma_data"), cfg.get("sigma_min"), cfg.get("sigma_max")
    errors.check(_is_number(sd) and sd > 0, "sigma_data must be positive")
    errors.check(
        _is_number(lo) and _is_number(hi) and 0 < lo < hi,
        f"need 0 < sigma_min < sigma_max; got {lo!r}, {hi!r}",
    )


def validate_model(cfg) -> None:
    """Check that the network settings describe a network that can be built."""
    errors = _Errors()
    _check_model(cfg, errors)
    errors.raise_if_any("network settings")


def describe_architecture(cfg) -> str:
    """The levels, widths and attention placement the config builds."""
    temporal = int(cfg.time_length) > 1
    tattn = set(cfg.get("temporal_attn_resolutions") or []) if temporal else set()
    attn = set(cfg.get("attn_resolutions") or [])
    lines = [f"network: {cfg.H}x{cfg.W} grid, {len(cfg.channel_mult)} levels"]
    for level, (res, mult) in enumerate(zip(level_resolutions(cfg), cfg.channel_mult)):
        extras = []
        if res in attn:
            extras.append("spatial attention")
        if res in tattn:
            extras.append("temporal attention")
        tail = f", {' + '.join(extras)}" if extras else ""
        lines.append(
            f"  level {level}: {res}x{res}, {cfg.model_channels * mult} channels, "
            f"{cfg.num_blocks} blocks{tail}"
        )
    bottom = level_resolutions(cfg)[-1]
    extra = (
        " + temporal attention"
        if temporal and cfg.get("decoder_start_temporal_attn")
        else ""
    )
    lines.append(f"  bottom ({bottom}x{bottom}): spatial attention (always){extra}")
    if temporal and cfg.get("upsample_temporal_attn"):
        lines.append("  decoder upsampling blocks: temporal attention")
    return "\n".join(lines)


# ── Training ──────────────────────────────────────────────────────────────


def _check_per_channel(section, key, names, errors, *, check, message) -> list[str]:
    """Check a value that is a number or a {channel: number} map."""
    value = section.get(key)
    if value is None:
        return []
    if isinstance(value, (Mapping, DictConfig)):
        items = list(value.items())
        unknown = [str(k) for k in value if k not in names]
    else:
        items, unknown = [("all", value)], []
    for name, v in items:
        errors.check(check(v), f"{message} ({key}.{name}={v!r})")
    return unknown


def validate_training(cfg) -> list[str]:
    """Check a training config; raise :class:`ConfigError` listing every problem.

    Returns warnings (strings) for suspicious but valid settings.
    """
    errors, warnings = _Errors(), []
    source = cfg.get("data_source")
    errors.check(
        source in DATA_SOURCES,
        f"data_source must be one of {DATA_SOURCES}; got {source!r}",
    )
    raw_vars = cfg.get("variables")
    if raw_vars is None:
        errors.add("variables: missing (see configs/train_base.yaml)")
    else:
        _check_variables(raw_vars, errors, need_field=source == "patches")
    if source == "patches":
        errors.check(
            bool(cfg.get("data_dir")),
            "data_source=patches needs data_dir; for a code check without data "
            "use data_source=synthetic (or configs/train_smoke_test.yaml)",
        )
    _check_model(cfg, errors)

    lo, hi = cfg.get("time_range_start", 0), cfg.get("time_range_end")
    range_ok = _is_int(lo) and lo >= 0 and (hi is None or (_is_int(hi) and hi > lo))
    errors.check(
        range_ok,
        f"need 0 <= time_range_start < time_range_end; got {lo!r}, {hi!r}",
    )
    span_ok = _span_settings_ok(cfg)
    if source == "patches" and range_ok and span_ok and hi is not None:
        errors.check(
            hi - lo > _max_span(cfg),
            f"training hours [{lo}, {hi}) are too short for the longest window "
            f"span ({_max_span(cfg)} h); widen them or shorten the span",
        )
    mode = cfg.get("time_sampling_mode")
    errors.check(
        mode in TIME_SAMPLING_MODES,
        f"time_sampling_mode must be one of {TIME_SAMPLING_MODES}; got {mode!r}",
    )
    T = cfg.get("time_length", 1)
    if mode == "nonuniform_offsets" and _is_int(T):
        step = cfg.get("nonuniform_minimum_offset_hours")
        span = cfg.get("nonuniform_max_span_hours")
        ok = _is_number(step) and step > 0 and _is_number(span)
        errors.check(
            ok and span >= step * (T - 1),
            "nonuniform_max_span_hours must be at least "
            "nonuniform_minimum_offset_hours * (time_length - 1)",
        )
    if mode == "uniform_stride":
        choices = list(cfg.get("llc_dtime_choices") or [])
        errors.check(
            bool(choices) and all(_is_int(c) and c > 0 for c in choices),
            "llc_dtime_choices must list positive integer strides (hours)",
        )

    dist = cfg.get("noise_distribution", "log_uniform")
    errors.check(
        dist in NOISE_DISTRIBUTIONS,
        f"noise_distribution must be one of {NOISE_DISTRIBUTIONS}; got {dist!r}",
    )
    if dist == "log_normal":
        errors.check(
            _is_number(cfg.get("p_std")) and cfg.p_std > 0, "p_std must be positive"
        )

    for key in ("batch_size", "num_steps", "log_every", "checkpoint_every"):
        v = cfg.get(key)
        errors.check(_is_int(v) and v >= 1, f"{key} must be an integer >= 1")
    errors.check(
        _is_int(cfg.get("num_workers")) and cfg.num_workers >= 0,
        "num_workers must be an integer >= 0",
    )
    errors.check(
        _is_int(cfg.get("lr_rampup_steps")) and cfg.lr_rampup_steps >= 0,
        "lr_rampup_steps must be an integer >= 0",
    )
    errors.check(_is_number(cfg.get("lr")) and cfg.lr > 0, "lr must be positive")
    errors.check(
        _is_number(cfg.get("lr_min")) and cfg.lr_min >= 0, "lr_min must be >= 0"
    )
    ema = cfg.get("ema_decay")
    errors.check(
        _is_number(ema) and 0 <= ema < 1, f"ema_decay must be in [0, 1); got {ema!r}"
    )
    errors.check(
        _in_unit_interval(cfg.get("label_dropout", 0.0)),
        "label_dropout must be in [0, 1]",
    )
    dropout = cfg.get("dropout", 0.0)
    errors.check(_is_number(dropout) and 0 <= dropout < 1, "dropout must be in [0, 1)")

    start = cfg.get("start", "fresh")
    errors.check(
        start in START_MODES, f"start must be one of {START_MODES}; got {start!r}"
    )
    errors.check(
        not (start == "resume" and cfg.get("init_from")),
        "init_from starts a new run from other weights, so it cannot be "
        "combined with start=resume; use start=fresh (new run) or drop init_from",
    )
    errors.check(
        not (start == "fresh" and cfg.get("resume_from")),
        "resume_from is set but start=fresh; use start=resume to continue that "
        "checkpoint, or init_from to start a new run from its weights",
    )

    names = [v.get("name") for v in raw_vars or []]
    cond = cfg.get("conditioning") or {}
    if (cond.get("mode") or "none") not in ("none", "observations"):
        errors.add(
            "conditioning.mode must be 'none' or 'observations'; "
            f"got {cond.get('mode')!r}"
        )
    elif (cond.get("mode") or "none") == "observations":
        observe = list(cond.get("observe") or names)
        unknown = [c for c in observe if c not in names]
        errors.check(
            not unknown,
            f"conditioning.observe has {unknown}, which are not variables {names}",
        )
        obs = cond.get("training_observations") or {}
        errors.check(
            obs.get("masks", "bernoulli") in ("bernoulli", "stripes"),
            "conditioning.training_observations.masks must be bernoulli or stripes",
        )
        stray = []
        for key, check, message in (
            ("coverage", _in_unit_interval, "coverage must be in [0, 1]"),
            ("frame_probability", _in_unit_interval, "must be in [0, 1]"),
            ("noise_std", lambda v: _is_number(v) and v >= 0, "must be >= 0"),
            ("stripe_width", lambda v: _is_number(v) and v > 0, "must be > 0"),
            ("stripes_per_frame", lambda v: _is_int(v) and v >= 1, "must be >= 1"),
        ):
            stray += _check_per_channel(
                obs, key, observe, errors, check=check, message=message
            )
        if stray:
            warnings.append(
                "conditioning.training_observations has entries for "
                f"{sorted(set(stray))}, which are not observed channels; ignored"
            )
        errors.check(
            _in_unit_interval(obs.get("null_probability", 0.0)),
            "conditioning.training_observations.null_probability must be in [0, 1]",
        )

    val = cfg.get("validation") or {}
    every = val.get("every", 0)
    errors.check(_is_int(every) and every >= 0, "validation.every must be >= 0")
    if every:
        errors.check(
            _is_int(val.get("n_samples")) and val.n_samples >= 1,
            "validation.n_samples must be >= 1",
        )
        sigmas = list(val.get("sigmas") or [])
        errors.check(
            bool(sigmas) and all(_is_number(s) and s > 0 for s in sigmas),
            "validation.sigmas must list positive noise levels",
        )
        if source == "patches":
            vlo, vhi = val.get("time_range_start"), val.get("time_range_end")
            if vlo is None or vhi is None:
                errors.add(
                    "validation with data_source=patches needs "
                    "validation.time_range_start and time_range_end (held-out "
                    "hourly indices)"
                )
            elif _is_int(lo) and _is_int(vlo) and _is_int(vhi) and span_ok:
                buffer = _max_span(cfg)
                errors.check(
                    vhi - vlo > buffer,
                    f"validation hours [{vlo}, {vhi}) are too short for the "
                    f"longest window span ({buffer} h)",
                )
                train_hi = hi if hi is not None else float("inf")
                overlaps = vlo < train_hi + buffer and lo < vhi + buffer
                errors.check(
                    not overlaps,
                    f"validation hours [{vlo}, {vhi}) overlap or touch training "
                    f"hours [{lo}, {hi}) within one window span ({buffer} h); "
                    "windows from the two would share frames. Leave a gap of at "
                    "least the maximum window span",
                )

    errors.raise_if_any("training config")
    return warnings


def _span_settings_ok(cfg) -> bool:
    T = cfg.get("time_length")
    if not _is_int(T):
        return False
    if cfg.get("time_sampling_mode") == "uniform_stride":
        choices = list(cfg.get("llc_dtime_choices") or [])
        return bool(choices) and all(_is_int(c) and c > 0 for c in choices)
    return _is_number(cfg.get("nonuniform_max_span_hours"))


def _max_span(cfg) -> int:
    """Longest first-to-last window span the sampler can draw (hours)."""
    T = int(cfg.time_length)
    if cfg.get("time_sampling_mode") == "uniform_stride":
        return max(int(c) for c in cfg.llc_dtime_choices) * (T - 1)
    return int(cfg.nonuniform_max_span_hours)


# ── Inference ─────────────────────────────────────────────────────────────


def validate_sda(cfg, model_cfg) -> None:
    """Check an SDA config (``configs/sda.yaml`` layout) against a checkpoint."""
    errors = _Errors()
    names = channel_names(model_cfg)
    smp = cfg.get("sampler") or {}
    for key in ("n_members", "n_batch", "steps"):
        v = smp.get(key)
        errors.check(_is_int(v) and v >= 1, f"sampler.{key} must be an integer >= 1")
    errors.check(
        smp.get("timestep_spacing") in TIMESTEP_SPACINGS,
        f"sampler.timestep_spacing must be one of {TIMESTEP_SPACINGS}",
    )
    method = (cfg.get("guidance") or {}).get("method")
    errors.check(
        method in GUIDANCE_METHODS,
        f"guidance.method must be one of {GUIDANCE_METHODS}; got {method!r}",
    )
    mmps = cfg.get("mmps") or {}
    errors.check(mmps.get("solver") in SOLVERS, f"mmps.solver must be one of {SOLVERS}")
    errors.check(
        _is_int(mmps.get("iterations")) and mmps.iterations >= 1,
        "mmps.iterations must be an integer >= 1",
    )
    gamma = (cfg.get("rozet") or {}).get("gamma")
    errors.check(_is_number(gamma) and gamma >= 0, "rozet.gamma must be >= 0")
    weight = (cfg.get("classifier_free") or {}).get("weight")
    errors.check(_is_number(weight), "classifier_free.weight must be a number")

    obs = cfg.get("observations") or {}
    mode = obs.get("mode")
    errors.check(
        mode in OBSERVATION_MODES,
        f"observations.mode must be one of {OBSERVATION_MODES}; got {mode!r}",
    )
    observe = list(obs.get("observe") or names) if mode != "none" else []
    unknown = [c for c in observe if c not in names]
    errors.check(
        not unknown, f"observations.observe has {unknown}; the model has {names}"
    )
    sigma_y = obs.get("sigma_y") or {}
    for c in observe:
        v = sigma_y.get(c) if isinstance(sigma_y, (Mapping, DictConfig)) else sigma_y
        errors.check(
            _is_number(v) and v > 0,
            f"observations.sigma_y.{c} must be a positive error std in physical "
            f"units; got {v!r}",
        )
    if mode == "random":
        cov = obs.get("random_coverage") or {}
        for c in observe:
            v = cov.get(c) if isinstance(cov, (Mapping, DictConfig)) else cov
            errors.check(
                _in_unit_interval(v),
                f"observations.random_coverage.{c} must be in [0, 1]",
            )

    cond = model_cfg.get("conditioning") or {}
    conditional = (cond.get("mode") or "none") != "none"
    if method in ("conditional", "cfg"):
        errors.check(
            conditional,
            f"guidance.method={method} needs a conditional model; this "
            "checkpoint is an unconditional prior (use mmps or rozet)",
        )
    if method == "cfg" and conditional:
        errors.check(
            trained_null_probability(model_cfg) > 0,
            "guidance.method=cfg needs a model trained with "
            "conditioning.training_observations.null_probability > 0; this one "
            "never saw the null condition, so CFG's unconditional branch is "
            "untrained",
        )
    errors.raise_if_any("SDA config")


def trained_null_probability(model_cfg) -> float:
    """``null_probability`` a conditional checkpoint was trained with."""
    cond = model_cfg.get("conditioning") or {}
    obs = cond.get("training_observations") or {}
    return float(obs.get("null_probability", 0.0) or 0.0)


# ── Comparing runs ────────────────────────────────────────────────────────


def _flatten(node, prefix="") -> dict:
    out = {}
    if isinstance(node, Mapping):
        for key, value in node.items():
            out.update(_flatten(value, f"{prefix}{key}."))
    elif isinstance(node, list) and node and isinstance(node[0], Mapping):
        out[f"{prefix}length"] = len(node)
        for i, value in enumerate(node):
            out.update(_flatten(value, f"{prefix}{i}."))
    else:
        out[prefix[:-1]] = node
    return out


def run_differences(new, old, *, ignore=RUNTIME_KEYS) -> list[str]:
    """Settings that differ between two configs, except ``ignore`` keys.

    Both are compared in the current format with variables resolved, so a
    legacy checkpoint config compares fairly with a new one.
    """
    a, b = _plain(upgrade(new)), _plain(upgrade(old))
    a["variables"] = variables(a)
    b["variables"] = variables(b)
    for d in (a, b):
        for v in d["variables"]:
            # The checksum is filled in at the start of a run.
            clim = v["normalization"]["climatology"]
            if clim is not None:
                clim.pop("sha256", None)
    fa, fb = _flatten(a), _flatten(b)
    diffs = []
    for key in sorted(set(fa) | set(fb)):
        if key.split(".")[0] in ignore or key == "config_version":
            continue
        if fa.get(key) != fb.get(key):
            diffs.append(f"{key}: {fb.get(key)!r} -> {fa.get(key)!r}")
    return diffs
