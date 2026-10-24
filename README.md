# obottle-starter

Diffusion priors for sea-surface height (SSH) and temperature (SST), and
score-based data assimilation (SDA) that conditions them on sparse, noisy
observations. This is a small, self-contained starting point. It can:

* **run SDA with pretrained priors** on bundled OSSE (observing system
  simulation experiment) tiles: llc4320 truth, observed where real SWOT,
  nadir-altimeter and ACSPO instruments saw the ocean (the observed values
  come from the simulation);
* **train your own `SongUNet` prior** on 128×128 llc4320 patches, on your
  own data in the same layout, or on generated structured fields that need
  no data files.

The network is cBottle's EDM-preconditioned `SongUNet` (NVIDIA), adapted for
ocean patches with temporal attention over irregularly spaced frames. The
SDA uses Moment Matching Posterior Sampling (MMPS; Rozet et al., 2024) on a
DDIM sampler, built on a vendored subset of
[azula](https://github.com/probabilists/azula).

**New to diffusion models or SDA?** [`docs/`](docs/README.md) explains the
network, training, data loading, inference methods and possible extensions,
at the level of a first-year graduate student.

## Install

Python ≥ 3.10 and a CUDA build of PyTorch.

```bash
git clone <this repo> obottle-starter && cd obottle-starter
pip install -e '.[examples,dev]'
python -m pytest              # ~3 min on CPU
```

Install in editable mode (`-e`), as above, and keep the repository in
place. The code finds `data/` relative to the repository.

## Data bundle

The checkpoints and test data are shared separately. Put them next to the
repository, or anywhere you like and pass the paths:

```
checkpoints/
  joint_ssh_sst_T5.pt      joint SSH + SST prior, 5 frames      (95 MB)
  ssh_only_T11.pt          SSH-only prior, 11 frames            (95 MB)
osse_tiles/
  MidAtlantic_35N_50W.nc   336x336 cells (4 km), 24 frames, 12-hourly (16 MB each)
  NorthAtlantic_48N_30W.nc
  SubtropAtlantic_26N_44W.nc
training_patches/          optional: 8 North Pacific SSH patches x 2000 hours (850 MB)
```

With the bundle in place, `python -m pytest` also checks that both
checkpoints reproduce the original training-code network
(`OBOTTLE_CHECKPOINTS=checkpoints python -m pytest`).

### The pretrained models

| | `joint_ssh_sst_T5` | `ssh_only_T11` |
|---|---|---|
| channels | SSH, SST | SSH |
| frames per window | 5 | 11 |
| patch | 128 × 128 at 4 km (512 km) | same |
| training data | llc4320 North Pacific, 415 patches, Sep 2011 – Sep 2012, hourly | same |
| frame spacing in training | random, 1 h grid, total span ≤ 336 h | same |
| steps | 315k (batch 130) | 195k (batch 60) |
| parameters | 23.7 M | 23.7 M |

**SSH** is llc4320 sea-surface height with scales longer than 300 km
removed, in metres, normalized by 0.051 m. **SST** is in °C. A North Pacific
basin-mean daily climatology is subtracted, then the result is divided by
5.5 °C. Normalization is applied for you: `assimilate`, the scripts and the
OSSE helpers take and return physical units. (The low-level classes in
`obottle_starter.inference.sda` work in the units of their inputs, normally
model units.) See [Units and normalization](#units-and-normalization).

The released models look the SST climatology up by days since its first
day (2011-09-13), so they only accept dates inside its record, 2011-09-13
to 2012-09-23; other dates are an error unless you choose another rule
(`normalization.climatology.lookup=day_of_year`, or `out_of_range=clip`).

The OSSE tiles are in the Atlantic, so they also test transfer from the
Pacific training region.

## Run SDA

Command line (defaults in `configs/sda.yaml`):

```bash
python scripts/run_sda.py ckpt=checkpoints/joint_ssh_sst_T5.pt \
    tile=osse_tiles/MidAtlantic_35N_50W.nc output=out/mid_atlantic.nc

# variations
... observations.observe='[SSH]'                 # observe SSH only
... observations.mode=random                     # Bernoulli masks instead of instrument coverage
... observations.mode=none                       # prior samples
... observations.sigma_y.SSH=0.01 mmps.iterations=8 sampler.steps=128
... guidance.method=rozet rozet.gamma=0.01         # Rozet & Louppe (2023) heuristic
... window.t_start=6 window.origin='[50,180]'     # another window of the tile
```

The output netCDF holds `samples` (sample, channel, time, row, col),
`truth`, `mask` and `observations` (the values used, NaN elsewhere), in
physical units, with latitude and longitude. The run prints diagnostics:
errors at observed and unobserved pixels, spread against error, interval
coverage and the observation misfit (`inference/metrics.py`). Options are
checked before anything runs; a misspelled option is an error.

Python:

```python
import obottle_starter as obs
from obottle_starter.inference.osse import load_window

net, cfg = obs.load_checkpoint("checkpoints/joint_ssh_sst_T5.pt", "cuda")
names = obs.channel_names(cfg)                       # ['SSH', 'SST']
win = load_window("osse_tiles/MidAtlantic_35N_50W.nc", names, cfg.time_length)

post = obs.assimilate(
    net, cfg, times=win.times, center_lon=win.center_lon,
    truth=win.truth,                  # {channel: (T, H, W)} in m / degC
    masks=win.masks,                  # {channel: (T, H, W) bool}
    sigma_y={"SSH": 0.005, "SST": 0.1},
    n_members=8, steps=64, mmps_iterations=4,
)
post.samples.mean("sample")           # xarray, physical units
```

For real observations, pass `y={channel: (T, H, W) values}` instead of
`truth`. Only values on `masks` are used. `examples/01_prior_and_sda.ipynb`
walks through prior samples, posteriors, errors and spectra.

### Knobs that matter

| option | meaning |
|---|---|
| `sigma_y` | observation-error std the guidance assumes (m, °C); required. Smaller fits the data harder. |
| `instrument_noise` | noise added to the synthetic observations (default: `sigma_y`) |
| `steps` | DDIM steps from σ = 200 down to 2e-4 |
| `mmps_iterations` | Krylov iterations per step, one network backward each. Cost grows roughly linearly. |
| `mmps_solver` | `gmres` (default), `cg`, `minres` |
| `guidance`, `rozet_gamma` | `mmps` (default) or `rozet`: a fixed uncertainty `γ σ_t²` instead of the network's, no linear solve, one backward pass per step |
| `guidance="conditional"` / `"cfg"`, `cfg_weight` | for conditional models only (experimental; `docs/06_conditional_models.md`) |
| `n_members`, `n_batch` | ensemble size; members per GPU batch |

The Python defaults equal `configs/sda.yaml` (`inference/sda/defaults.py`;
a test keeps them identical). They are starting points: check that results
stop changing when you increase `steps` or `mmps_iterations`.

Measured on one L40S with 64 steps and 4 iterations, 4 members per batch:
2.2 s per step for the joint 5-frame model (about 2.5 min per batch) and
5.2 s per step for the SSH-only 11-frame model.

## Units and normalization

The network is trained, and SDA is solved, in **model units**. You work in
**physical units** (m, °C). A `Normalizer` converts between them with one
affine map per channel, `x_model = (x - offset(t)) / scale`.

**The numbers come from the checkpoint.** Training saves its config into
every checkpoint, and each entry of its `variables` list says how that
variable is normalized, `x_model = (x - climatology(t) - offset) / scale`:

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

Any difference from the checkpoint is printed as a warning. A model only
gives sensible results with the normalization it was trained with.

**It is always printed and recorded.** `run_sda.py` starts with, for example:

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

**Why the solve uses model units.** At every diffusion step, MMPS solves a
linear system with a few Krylov (GMRES/CG) iterations. The exact answer
does not depend on units, but a truncated solve reduces the largest
residuals first. In physical units, an SST residual of several °C dwarfs an
SSH residual of centimetres. With the joint model and 4 iterations, the SSH
observations were then effectively ignored: ensemble-mean SSH RMSE was
0.073 m with physical-unit terms and 0.028 m in model units, on the same
window. So `assimilate` converts observations, `sigma_y` and instrument
noise to model units (`Normalizer.observation_terms`,
`Normalizer.sigma_to_model`). If you build your own observation operator,
do the same.

**Developing your own model.** To add or change a variable, edit the
`variables` list in your training config: its directory under `data_dir`,
units, a note on preprocessing (filtering changes what the variable *is*,
not just its units), and its normalization. No code changes are needed for
an affine normalization, with or without a climatology. Inference picks it
up from the checkpoint.

### Beyond the defaults

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

## Train

```bash
python scripts/train.py config=configs/train_smoke_test.yaml       # 1-min code check (white noise)
python scripts/train.py config=configs/train_structured.yaml       # small model, generated fields
python scripts/train.py data_dir=/path/to/patches                  # joint SSH+SST, as released
python scripts/train.py config=configs/train_ssh_only.yaml data_dir=training_patches \
    time_range_start=0 time_range_end=2000                         # the bundled subset
python scripts/train.py config=configs/train_conditional.yaml data_dir=... \
    init_from=checkpoints/joint_ssh_sst_T5.pt                      # conditional model (experimental)
python scripts/train.py ... start=resume num_steps=400000          # continue a run
torchrun --nproc_per_node=4 scripts/train.py data_dir=...          # multi-GPU (batch_size per GPU)
```

**How the configs combine.** `configs/train_base.yaml` lists every option
with its default and a description of what it does. `scripts/train.py`
always loads it first. It then merges an optional experiment file
(`config=configs/train_ssh_only.yaml`), which lists only the options it
changes, and finally any `key=value` overrides. To make your own
experiment, copy one of the small experiment files. Options the base file
does not define are rejected, and the whole config is checked (grid,
attention levels, variables, noise range, probabilities, start mode) before
any data is read, with messages that say what to change. The fully merged
config is saved as `runs/<experiment_name>/config.yaml` and inside every
checkpoint. Runs write
`runs/<experiment_name>/checkpoints/checkpoint_{latest,stepN}.pt`.

**Starting and resuming are explicit** (`start`): `fresh` (default) refuses
an `output_dir` that already holds a run; `resume` continues it and refuses
changes to anything but runtime settings such as `num_steps`; `auto`
resumes if possible, for requeued jobs. `init_from` starts a new run from
another checkpoint's weights. See `docs/02_training.md`. A trained
checkpoint works directly with `run_sda.py` and `assimilate`.
`examples/02_train_small.ipynb` trains a small model and samples it.

Training throughput on one L40S at the released model size: about 9.5
samples/s for 5 frames (batch 32) and 4.3 samples/s for 11 frames (batch 8).
The released joint model saw about 41 M samples (315k steps at batch 130),
which would take weeks on one L40S. To explore, use smaller models
(`model_channels`, `time_length`) or multiple GPUs.

### Data layout

```
{data_dir}/{field}/{patch_id}.zarr     one store per variable (its `field`) and patch
```

Each store has a data variable on `(time, row, col)` (rows south→north,
columns west→east; the variable's `source_variable`, or the store's only
one), an hourly `time` coordinate, and 2-D `latitude` / `longitude`. Every
field must have the same patch ids and times, and values must be finite
(training stops at the first NaN). Each sample is a random window of
`time_length` frames from a random patch. The frame offsets follow
`time_sampling_mode`. To add a variable, add a field directory and an entry
in `variables`.

## Layout

```
src/obottle_starter/
  core/                shared by training and inference
    nn/                  SongUNet + EDMPrecond (from NVIDIA cBottle)
    config.py            variables list, config checks, legacy translation
    model.py             build_model, load_checkpoint, export_checkpoint
    obs_conditioning.py  observations as network inputs (conditional models)
    normalization.py     physical units <-> model units
    conditioning.py      frame times -> the network's time inputs
  training/            developing a prior
    data.py              PatchDataset (per-patch zarr), StructuredDataset, SyntheticDataset
    train.py             training loop (single GPU or DDP), start modes
    validation.py        held-out denoising loss by variable and noise level
    obs_simulation.py    synthetic observations for conditional models
  inference/           using a prior: SDA
    assimilate.py        one call from observations to a posterior ensemble
    osse.py              OSSE tiles, windows, random masks
    metrics.py           errors, spread/skill, ranks, coverage, spectra, coherence
    sda/                 MMPS, Rozet and CFG guidance, Krylov solvers, DDIM, observation operators
  cli.py               config loading for the scripts
data/                  SST climatology used by the released models' normalization
configs/               train_base.yaml (every training option) + experiment presets; sda.yaml
scripts/               train.py, run_sda.py; maintainer/ builds the data bundle
docs/                  background: network, training, data, SDA, extensions, conditional models, literature
examples/              notebooks
tests/
```

If you are developing a model, work in `core/` and `training/`. If you are
assimilating data, work in `inference/`. `core/` never imports from the
other two. The main functions are also available from the top level
(`obottle_starter.load_checkpoint`, `obottle_starter.assimilate`, ...).

## Provenance

This repository was extracted from two research codebases. The network and
training data pipeline come from `cBottle-for-ocean-DA/ocean-examples`. The
MMPS sampler comes from `oBottle-for-SSH-decomposition/oBottle-inference`.
Its tests check MMPS against the exact posterior of a Gaussian prior. The
released checkpoints were trained in cBottle-for-ocean-DA. The OSSE tiles are
the three Atlantic test cases used there. The windowed (multi-tile) SDA used
for full 336×336 tiles is not included.

## Licenses

Apache-2.0 (`LICENSE.txt`), except the vendored azula subset
(`src/obottle_starter/inference/sda/_azula/`, MIT). See `THIRD_PARTY_LICENSES.md`.
