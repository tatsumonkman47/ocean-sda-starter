# Units and scaling

The main Python functions use **physical units** (m, °C); training and SDA use
**model units**, scaled for the network. `Normalizer` converts each variable
using:

```text
x_model = (x - climatology(t) - offset) / scale
```

Use the [inference guide](inference.md) for reconstruction commands and the
[training guide](training.md) when preparing a new model. The
[download guide](downloads.md#variables-units-and-supported-dates) lists the
released models' units and supported dates.

## Settings saved with the model

Normalization settings are stored in the checkpoint's `variables` list.
For example:

```yaml
variables:
  - name: SST
    field: FULL_PACIFIC_zarr_llc4320_SST_tiles_4km_unfiltered_fullt
    units: degC
    preprocessing: llc4320 SST, unfiltered
    normalization:
      scale: 5.5
      offset: 0.0
      climatology:              # optional time-dependent offset
        path: null              # null = data/SST_NP_daily_climatology.nc
        variable: SST
        lookup: elapsed_days    # the released rule; day_of_year for new models
        out_of_range: error
        sha256: null            # recorded when training starts, checked on load
```

The released checkpoints store an older flat format (`out_transform_list`,
`llc_standards`, `climatology_path`), which is translated to this list when
they are loaded. The `normalization` section of `configs/sda.yaml` selects
what inference uses:

```yaml
normalization:
  source: from_checkpoint   # default: exactly what the model was trained with
  climatology: {path: null, lookup: null, out_of_range: null}   # adjust; null = keep
  variables: null           # source: config uses this list instead (experiments only)
```

Overrides that differ from the checkpoint trigger a warning. Use the
training normalization for consistent inference.

<details>
<summary>Example normalization report</summary>

`run_sda.py` prints the normalization at startup, for example:

```
normalization (from checkpoint):
  SSH: x_model = x / 0.051 m  [std_ssh_norm]
       sigma_y 0.005 m -> 0.09804 model units
  SST: x_model = (x - clim(t)) / 5.5 degC
       sigma_y 0.1 degC -> 0.01818 model units
       preprocessing: llc4320 SST, unfiltered
  clim(t) for SST = daily SST from .../SST_NP_daily_climatology.nc (elapsed_days, out of range: error)
  SDA observations and sigma_y are solved in model units
```

The output netCDF stores the same information in its `normalization_*`
attributes. Training prints its normalization at startup
and saves it in `config.yaml` and in every checkpoint.

</details>

## Why model units matter

MMPS uses a limited number of Krylov
iterations, which prioritize the largest residuals. In physical units,
SST residuals can dominate SSH residuals. On the example window with four
iterations, converting to model units reduced ensemble-mean SSH RMSE from
0.073 m to 0.028 m. `assimilate` handles this conversion for observations,
`sigma_y`, and instrument noise. Custom observation operators must use
consistent units too.

## Adding a variable

Define its field directory, units, preprocessing,
and normalization in the training config's `variables` list. Affine
normalization, with or without a climatology, requires no code changes.
Inference reads these settings from the checkpoint.

## Other observation types

`assimilate` covers per-channel masks. Underneath, `obottle_starter.inference.sda`
takes any linear observation operator:

```python
from obottle_starter.inference.sda import ObservationTerm, build_observation, run_inference
# y = sum_c w_c * L_c(x_c) + offset, sampled on a (T, H*W) mask, in model units
term = ObservationTerm("SSH_smoothed", mask, weights={0: 1.0}, transforms={0: smooth})
```

`L_c` can be any linear map, such as a smoothing filter or a footprint
average. Keep `y` and `sigma_y` in model units, as explained above. See
`src/obottle_starter/inference/sda/observations.py`.

For off-grid points, cell averages, footprints, time averages, combinations
of variables, or correlated errors, use `assimilate_general`. Like
`assimilate`, it accepts physical units:

```python
import torch
from obottle_starter.inference.general import assimilate_general
from obottle_starter.inference.sda.operators import block_average

shape = (2, cfg.time_length, cfg.H, cfg.W)                     # (C, T, H, W)
coarse = block_average(shape, 0, 16)                            # 64 km means of SSH
post = assimilate_general(net, cfg, coarse, torch.full((coarse.n_obs,), 0.003**2),
                          times=win.times, center_lon=win.center_lon, truth=win.truth)
```

`examples/03_general_observations.ipynb` covers three cases: coarse-to-fine
reconstruction, along-track points with correlated errors, and SST
footprints averaged over two frames.

---

[Documentation index](../README.md) · [Project README](../../README.md)
