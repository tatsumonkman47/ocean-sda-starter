# ocean-sda-starter

> **Work in progress — use with caution.** This repository, including its
> documentation, was organized and adapted from existing research code with
> the help of AI coding agents (Claude and Codex). Code and explanations may
> contain errors or incomplete validation; review and test them for your
> application. Contributions, corrections, and feedback are welcome.

**A tutorial and reusable foundation for diffusion-based field reconstruction.**
This repository provides training, data-loading, and posterior-sampling
workflows that you can adapt to your own variables, datasets, and scientific
applications.

The included joint sea-surface height (SSH) and temperature (SST) model
serves primarily as a **worked demonstration** of those workflows. The
repository is intended to support efforts beyond that particular model,
including new oceanographic and other geophysical reconstruction problems.

I built it with help from Claude and Codex to make it easier to learn these
methods, run an example, and develop your own experiments from a common
starting point.

**Start here:** [Getting started](#getting-started) · [Install](#install) ·
[Download models and data](#download-models-and-example-data)

**Explore:** [Reconstruct fields](#reconstruct-fields-from-observations) ·
[Train a model](#train-a-model) · [Documentation](docs/README.md) ·
[Planned additions](#planned-additions)

## Getting started

To begin, start with a quick check of the training pipeline, then train a small model
or try pretrained SDA. The training examples need no data download;
pretrained SDA requires the
[models and example data](#download-models-and-example-data) and a GPU.

| Task | Command or notebook | Expected result |
|---|---|---|
| Install and test | Follow [Install](#install), then run `python -m pytest` | Tests pass; two checkpoint tests are skipped unless enabled |
| Check the training pipeline | `python scripts/train.py config=configs/train_smoke_test.yaml` | Writes `runs/smoke_test/checkpoints/checkpoint_latest.pt` in about a minute on CPU. This uses white noise, so loss need not decrease. |
| Train a small model | `examples/02_train_small.ipynb` | Training and validation losses decrease; generated samples resemble the structured training fields. Runs on CPU or GPU. |
| Run pretrained SDA | `examples/01_prior_and_sda.ipynb` or [the command-line example](#reconstruct-fields-from-observations) | Reconstructs fields from sparse observations and evaluates errors. Allow about 10 minutes on a GPU. |

### Choose an extension

| Task | Start here | Requirements |
|---|---|---|
| Conditional models and classifier-free guidance (CFG) | `examples/04_conditional_and_cfg.ipynb`, [Conditional models](docs/06_conditional_models.md) | GPU (~20 min); no download needed |
| Tracks, averages, footprints, or correlated observation errors | `examples/03_general_observations.ipynb` | Downloaded models and example data; GPU |
| Your own data or variables | [Training data format](#training-data-format), [Units and scaling](#units-and-scaling), [Data loading](docs/03_data_loading.md) | Your data and GPU |
| Network or training changes | [Documentation](docs/README.md), pages 1, 2, and 5 | GPU |

## Install

Requires Python ≥ 3.10 and PyTorch ≥ 2.1. Install in a fresh environment:

```bash
git clone https://github.com/tatsumonkman47/ocean-sda-starter.git && cd ocean-sda-starter
python -m venv .venv && source .venv/bin/activate
pip install -e '.[examples,dev]'
python -m pytest              # ~2 min on CPU
```

Import the package as `obottle_starter`. Keep the repository in place after
installation: the code locates `data/` relative to it.

### Tested environment

**Python 3.13, PyTorch 2.14.1, and NVIDIA L40S**
(2026-10-26). To use the dependency versions tested with all four notebooks:

```bash
pip install -r requirements-tested.txt && pip install -e '.[examples,dev]'
```

Its PyTorch wheel bundles CUDA 13.0 and needs a recent NVIDIA
driver. If `torch.cuda.is_available()` is `False` on your GPU machine,
install the PyTorch build for your CUDA version from
[pytorch.org](https://pytorch.org/get-started/locally/) and then the
package. The tests run on CPU.

## Download models and example data

The saved models (checkpoints) and simulated observation datasets (OSSE
tiles) are too large for the repository. They are
attached to the
[`data-v1` release](https://github.com/tatsumonkman47/ocean-sda-starter/releases/tag/data-v1).
From the repository root:

```bash
URL=https://github.com/tatsumonkman47/ocean-sda-starter/releases/download/data-v1
mkdir -p checkpoints osse_tiles
for f in joint_ssh_sst_T5.pt ssh_only_T11.pt; do curl -L -o checkpoints/$f $URL/$f; done
for f in MidAtlantic_35N_50W NorthAtlantic_48N_30W SubtropAtlantic_26N_44W; do
    curl -L -o osse_tiles/$f.nc $URL/$f.nc
done
# optional, 850 MB unpacked: real llc4320 SSH patches for training
curl -L -O $URL/training_patches.zip && unzip training_patches.zip && rm training_patches.zip
```

### Where to put the files

Place the downloaded directories in the repository root:

```
ocean-sda-starter/
  checkpoints/
    joint_ssh_sst_T5.pt      joint SSH + SST model, 5 frames      (95 MB)
    ssh_only_T11.pt          SSH-only model, 11 frames            (95 MB)
  osse_tiles/
    MidAtlantic_35N_50W.nc   336x336 cells (4 km), 24 frames, 12-hourly (16 MB each)
    NorthAtlantic_48N_30W.nc
    SubtropAtlantic_26N_44W.nc
  training_patches/          optional: 8 North Pacific SSH patches x 2000 hours (850 MB)
  src/ configs/ examples/ ...
```

The notebooks use `BUNDLE = Path("..")` to find these directories from
`examples/`. Git ignores all three directories. To store the downloads
elsewhere, create symbolic links in the repository root (for example,
`ln -s /path/to/bundle/checkpoints .`), or update `BUNDLE` in the notebooks
and pass full paths to the scripts.

### Verify the downloaded models

**These tests must be enabled explicitly.** To check that both checkpoints
reproduce the original training-code network, run:

```bash
OBOTTLE_CHECKPOINTS=checkpoints python -m pytest
```

Without this environment variable, the two checkpoint tests are skipped,
even if the models have been downloaded.

### Available models

| | `joint_ssh_sst_T5` | `ssh_only_T11` |
|---|---|---|
| channels | SSH, SST | SSH |
| frames per window | 5 | 11 |
| patch | 128 × 128 at 4 km (512 km) | same |
| training data | llc4320 North Pacific, 415 patches, Sep 2011 – Sep 2012, hourly | same |
| frame spacing in training | random, 1 h grid, total span ≤ 336 h | same |
| steps | 315k (batch 130) | 195k (batch 60) |
| parameters | 23.7 M | 23.7 M |

### Variables, units, and supported dates

- **SSH:** llc4320 sea-surface height with scales longer than 300 km removed,
  in metres, normalized by 0.051 m.
- **SST:** temperature in °C, with a North Pacific basin-mean daily climatology
  subtracted and the result divided by 5.5 °C.

**Unit conversion is automatic:** `assimilate`, the scripts, and the OSSE
helpers take and return physical units. The low-level classes in
`obottle_starter.inference.sda` use the units of their inputs, normally model
units. See [Units and scaling](#units-and-scaling).

**Supported dates:** the released SST normalization looks up the climatology
by elapsed days from 2011-09-13. Dates outside **2011-09-13 through 2012-09-23**
raise an error unless you set `normalization.climatology.lookup=day_of_year`
or `normalization.climatology.out_of_range=clip`.

The OSSE tiles are in the Atlantic, so they also test transfer from the
Pacific training region.

## What you can do

### Reconstruct fields with a pretrained model

**Score-based data assimilation (SDA)** steers a pretrained diffusion model
using observations at sampling time. Here, we reconstruct joint SSH/SST
fields from synthetic observations derived from the MITgcm llc4320 simulation.

Diffusion posterior sampling methods modify the generated trajectory using
the **likelihood score**, so fields satisfy soft constraints such as agreement
with noisy observations. This follows from a Bayesian decomposition of the
conditional diffusion process and uses automatic differentiation through the
network. See [Chung et al. (2023)](https://arxiv.org/abs/2209.14687),
[Rozet and Louppe (2023)](https://arxiv.org/abs/2306.10574), and
[Rozet et al. (2024)](https://arxiv.org/abs/2405.13712).

For a detailed introduction aimed at fellow newcomers, see
[my preprint](https://doi.org/10.22541/essoar.15003398/v3). That study evaluates
**two independent models: one for SSH and one for SST**. This repository
builds on that work with several upgrades, including a joint SSH/SST model.

### Train a new model

The training example uses **128 × 128 llc4320 patches**, with a data loader
and configuration that can be adapted to other geophysical settings.

- **Try the workflow:** train on generated synthetic fields without downloading data.
- **Use your own data:** replace the input datasets and configure their units and normalization.
- **Develop a scientific application:** pay particular attention to data loading,
  normalization, and evaluation to use your GPUs efficiently and represent your fields accurately.

### Model and sampling method

- **Network:** NVIDIA cBottle's EDM-preconditioned `SongUNet`, with temporal
  attention added for irregularly spaced ocean time series.
- **Reconstruction:** Moment Matching Posterior Sampling
  (MMPS; [Rozet et al., 2024](https://arxiv.org/abs/2405.13712)), with a DDIM
  sampler and a vendored subset of [azula](https://github.com/probabilists/azula).

## Intro to diffusion models and score-based data assimilation

Below are some cartoon illustrations of both "unconditional" (here referring to generation without posterior-steering)
and SDA-steered diffusion model generation.
In both cases we begin with a Gaussian random field and are targeting a "truth" field with a SSH-like spectrum.

### Training: learn to remove noise

Add Gaussian noise at a random level $\sigma$ to a clean
patch $x_0$. Train the denoiser $D_\theta$ to recover the original patch.
Repeating this across noise levels teaches the model the distribution of
ocean fields. See [Training](docs/02_training.md).

<p align="center">
  <img src="examples/figures/a_training.png" alt="Training: a patch is noised, the denoiser estimates the clean patch, and the loss compares the estimate with the original">
  <br>
  <em>Figure 1. Training the network to recover a clean field from a noisy input.</em>
</p>

### Sampling: generate a field

Start from Gaussian noise and repeatedly denoise while lowering
$\sigma$. At each step, the model predicts a clean field $\hat{x}_0$
(bottom), and the sampler updates the noisy state $x_\tau$ (top).

<p align="center">
  <img src="examples/figures/b_unconditional.gif" alt="Unconditional denoising: the noisy state and the denoised estimate as the noise level falls">
  <br>
  <em>Figure 2. Generating a field without observations. The noisy state (top) and predicted clean field (bottom) evolve as noise decreases.</em>
</p>

<details><summary>Still frames</summary>

<p align="center">
  <img src="examples/figures/b_unconditional_strip.png" alt="Unconditional denoising, six still frames">
</p>

</details>

### Data assimilation: incorporate observations

SDA combines a trained model (the "prior") with
observations to generate reconstructions. Each step compares the model's
prediction with the observations and uses the difference to guide the next
update.

**Accounting for uncertainty.** MMPS estimates uncertainty in the denoised
prediction from the network's derivatives. This makes the observation update
more expensive than using a prescribed variance, but allows it to depend on
the model's prediction. The alternative `rozet` guidance uses
[Rozet and Louppe (2023)](https://arxiv.org/abs/2306.10574)'s variance heuristic.

**Reconstructing gaps and uncertainty.** The resulting samples reconstruct unobserved regions by sampling from the model's learned prior.
Different initial noise produces a set of possible
reconstructions (an ensemble). Because observations enter at sampling time, the same model can support different
observation patterns, instruments, and error levels. See
[Inference and SDA](docs/04_inference_sda.md) for the equations and methods,
and [the literature guide](docs/07_literature.md#applications-to-geophysical-data-assimilation)
for ocean applications.

<p align="center">
  <img src="examples/figures/c_sda_loop.png" alt="SDA loop: from pure noise, denoise, compare with observations, apply the likelihood score, step, and repeat until the posterior sample">
  <br>
  <em>Figure 3. The reconstruction cycle: denoise, compare with observations, update, and repeat.</em>
</p>

<p align="center">
  <img src="examples/figures/c_sda.gif" alt="SDA-steered denoising on two crossing SWOT passes">
  <br>
  <em>Figure 4. Reconstructing a field from two crossing SWOT passes. Observations guide the model as noise decreases.</em>
</p>

<details><summary>Still frames</summary>

<p align="center">
  <img src="examples/figures/c_sda_strip.png" alt="SDA-steered denoising, six still frames with the observations and the truth">
</p>

</details>

### Comparing posterior-sampling methods

DPS neglects uncertainty in the denoised estimate; Rozet and Louppe's
heuristic prescribes it; **MMPS estimates it from the denoiser**. These choices
can produce different sampling trajectories, even with the same initial
noise and observation. See [the method comparison](docs/04_inference_sda.md#three-ways-to-handle-the-uncertainty)
for the equations and limitations.

<details>
<summary>Figure 5 — DPS, SDA guidance, and MMPS in a one-dimensional example</summary>

<p align="center">
  <img src="docs/figures/mmps_vs_SDA_vs_DPS.png" alt="One-dimensional example comparing unconditional, DPS, SDA heuristic, and MMPS trajectories from the same noisy state toward an observation">
  <br>
  <em>Figure 5. Read from right to left, from the noisy state to the reconstruction. The curves illustrate one observation and one choice of settings, not a general ranking of the methods. The SDA curve uses gamma = 1; this repository defaults to 0.1.</em>
</p>

</details>

### Longer time series with sliding windows

**A model trained on short sequences can reconstruct a longer period.**
Following [Rozet and Louppe (2023)](https://arxiv.org/abs/2306.10574), we run
one diffusion process over the full sequence. At each step, overlapping
short windows are denoised and their central predictions assembled into a
full-sequence estimate, with special handling at the endpoints.

This reuses the trained model without averaging independently generated
sequences. The current implementation supports unconditional priors;
accuracy depends on how well local windows represent the longer sequence.
See [Sliding windows](docs/08_windowed_sda.md) for commands, memory costs,
and validation checks. This extends the reconstruction in **time**;
spatial tiling is a separate method.

<details>
<summary>Figure 6 — How overlapping temporal windows form a long-sequence prediction</summary>

<p align="center">
  <a href="docs/figures/rozet_sliding_window.pdf"><img src="docs/figures/rozet_sliding_window.png" alt="Sliding-window construction: physical time runs vertically, while successive denoising steps move horizontally from noise to reconstructed fields"></a>
  <br>
  <em>Figure 6. Physical time runs vertically; denoising progresses horizontally. Overlapping windows provide local predictions at each step of the full-sequence sampling process. Click the figure for the original PDF.</em>
</p>

</details>

## Reconstruct fields from observations

### Command line

Run from the repository root. Defaults are in `configs/sda.yaml`:

```bash
python scripts/run_sda.py ckpt=checkpoints/joint_ssh_sst_T5.pt \
    tile=osse_tiles/MidAtlantic_35N_50W.nc output=out/mid_atlantic.nc

# Append these overrides to the command above (replace the leading ...).
... observations.observe='[SSH]'                 # observe SSH only
... observations.mode=random                     # Bernoulli masks instead of instrument coverage
... observations.mode=none                       # prior samples
... observations.sigma_y.SSH=0.01 mmps.iterations=8 sampler.steps=128
... guidance.method=rozet rozet.gamma=0.1          # Rozet & Louppe (2023) heuristic
... window.t_start=6 window.origin='[50,180]'     # another window of the tile
```

**Output and diagnostics.** The output netCDF contains `samples` with dimensions
`(sample, channel, time, row, col)`, plus `truth`, `mask`, and `observations`
(NaN outside the mask). Fields use physical units and include latitude and
longitude. The script reports reconstruction errors, ensemble spread,
interval coverage, and observation misfit. Invalid options are rejected
before inference starts.

### Python

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

**Using real observations:** pass `y={channel: (T, H, W) values}` instead of
`truth`. Only values on `masks` are used. `examples/01_prior_and_sda.ipynb`
walks through generated fields, observation-based reconstructions, errors,
and spectra.

### Settings

| Option | Meaning |
|---|---|
| `sigma_y` | Required observation-error standard deviation (m, °C). Smaller values enforce closer agreement with observations. |
| `instrument_noise` | noise added to the synthetic observations (default: `sigma_y`) |
| `steps` | DDIM steps from σ = 200 down to 2e-4 |
| `mmps_iterations` | Krylov iterations per step, each requiring a network backward pass. Cost grows roughly linearly. |
| `mmps_solver` | `gmres` (default), `cg`, `minres` |
| `guidance`, `rozet_gamma` | `mmps` (default) or `rozet`: a fixed uncertainty `γ σ_t²` instead of the network's, no linear solve, one backward pass per step |
| `guidance="conditional"` / `"cfg"`, `cfg_weight` | for conditional models only (experimental; `docs/06_conditional_models.md`) |
| `n_members`, `n_batch` | ensemble size; members per GPU batch |

Python defaults match `configs/sda.yaml`. Check result stability by increasing
`steps` or `mmps_iterations` until results stabilize.

### Expected runtime

Measured on one L40S with 64 steps and 4 iterations, 4 members per batch:
2.2 s per step for the joint 5-frame model (about 2.5 min per batch) and
5.2 s per step for the SSH-only 11-frame model.

## Units and scaling

The main Python functions use **physical units** (m, °C); training and SDA use
**model units**, scaled for the network. `Normalizer` converts each variable
using:

```text
x_model = (x - climatology(t) - offset) / scale
```

### Settings saved with the model

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

### Why model units matter

MMPS uses a limited number of Krylov
iterations, which prioritize the largest residuals. In physical units,
SST residuals can dominate SSH residuals. On the example window with four
iterations, converting to model units reduced ensemble-mean SSH RMSE from
0.073 m to 0.028 m. `assimilate` handles this conversion for observations,
`sigma_y`, and instrument noise. Custom observation operators must use
consistent units too.

### Adding a variable

Define its field directory, units, preprocessing,
and normalization in the training config's `variables` list. Affine
normalization, with or without a climatology, requires no code changes.
Inference reads these settings from the checkpoint.

### Other observation types

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

## Train a model

### Example commands

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

### Configure a training run

Training configuration is loaded in this order, with later values taking
precedence:

1. `configs/train_base.yaml`: all options, defaults, and descriptions.
2. An optional experiment file, such as `config=configs/train_ssh_only.yaml`.
3. Command-line `key=value` overrides.

Copy an experiment file to create your own configuration. Unknown options
and invalid settings are rejected before data loading. Each run saves its
merged config to `runs/<experiment_name>/config.yaml` and in every
checkpoint. Checkpoints are written to
`runs/<experiment_name>/checkpoints/checkpoint_{latest,stepN}.pt`.

### Start, resume, or initialize from a checkpoint

Use `start` to control run behavior:

- `fresh` (default): starts a new run; rejects an existing run directory.
- `resume`: continues a run; permits changes only to runtime settings such
  as `num_steps`.
- `auto`: resumes if possible, for requeued jobs.

Use `init_from` to start a new run from another checkpoint's weights.
Trained checkpoints work directly with `run_sda.py` and `assimilate`.
See [Training](docs/02_training.md) for details, notebook 02 for a small
training example, and notebook 04 for conditional training and CFG.

### Training cost

Training throughput on one L40S at the released model size: about 9.5
samples/s for 5 frames (batch 32) and 4.3 samples/s for 11 frames (batch 8).
The released joint model saw about 41 M samples (315k steps at batch 130),
which would take weeks on one L40S. To explore, use smaller models
(`model_channels`, `time_length`) or multiple GPUs.

### Training data format

```
{data_dir}/{field}/{patch_id}.zarr     one store per variable (its `field`) and patch
```

Each Zarr store must contain:

- A data variable with dimensions `(time, row, col)`, with rows ordered
  south to north and columns west to east. Select it with `source_variable`,
  or provide a store containing only one data variable.
- An hourly `time` coordinate and 2-D `latitude` and `longitude` coordinates.
- Finite values; training stops at the first NaN.

All fields must share the same patch IDs and times. Training samples random
patches and windows of `time_length` frames, with frame offsets controlled
by `time_sampling_mode`. To add a variable, add its field directory and an
entry in `variables`.

## Repository structure

<details>
<summary>Expand the directory and module reference</summary>

```
src/obottle_starter/
  core/                shared by training and inference
    nn/                  SongUNet + EDMPrecond (from NVIDIA cBottle)
    config.py            variables list, config checks, legacy translation
    model.py             build_model, load_checkpoint, export_checkpoint
    obs_conditioning.py  observations as network inputs (conditional models)
    normalization.py     physical units <-> model units
    conditioning.py      frame times -> the network's time inputs
  training/            training a model
    data.py              PatchDataset (per-patch zarr), StructuredDataset, SyntheticDataset
    train.py             training loop (single GPU or DDP), start modes
    validation.py        held-out denoising loss by variable and noise level
    obs_simulation.py    synthetic observations for conditional models
  inference/           reconstructing fields with SDA
    assimilate.py        one call from observations to a posterior ensemble
    osse.py              OSSE tiles, windows, random masks
    metrics.py           errors, spread/skill, ranks, coverage, spectra, coherence
    general.py           assimilate_general: general linear observations, physical units (opt-in)
    sda/                 MMPS, Rozet and CFG guidance, Krylov solvers, DDIM, observation operators
                         (masks: observations.py; points, averages, footprints, covariances: operators.py)
  cli.py               config loading for the scripts
data/                  SST climatology used by the released models' normalization
configs/               train_base.yaml (every training option) + experiment presets; sda.yaml
scripts/               train.py, run_sda.py; maintainer/ prepares model and data downloads
docs/                  background: network, training, data, SDA, extensions, conditional models, literature
examples/              notebooks 01-04 (pretrained SDA, small training, general observations,
                         conditional models and CFG)
  figures/             README figures and the script that draws them
tests/
```

</details>

Model development belongs in `core/` and `training/`; assimilation belongs
in `inference/`. `core/` does not import either module. Main functions such
as `load_checkpoint` and `assimilate` are also available from
`obottle_starter`.

## Planned additions

I intend to update this repository as my projects progress. Planned additions include:

- **Larger-domain reconstruction:** SDA-compatible
  [MultiDiffusion](https://arxiv.org/abs/2302.08113) and Score-based MultiDiffusion
  (manuscript in preparation).
- **EDM2:** the architecture and training strategy of
  [Karras et al. (2024)](https://arxiv.org/abs/2312.02696).
- **Conditioning and guidance:** more robust support for classifier-free
  guidance ([Ho and Salimans, 2022](https://arxiv.org/abs/2207.12598)),
  AutoGuidance ([Karras et al., 2024](https://arxiv.org/abs/2406.02507)), and
  methods such as [Ahn et al. (2026)](https://arxiv.org/abs/2610.14136).
  Thanks to Jacob Spier for suggesting the latter.

## Where this code comes from

### Research background

The code and methods build on:

- **NVIDIA Modulus and cBottle**, including the diffusion architecture of
  [Karras et al. (2022)](https://arxiv.org/abs/2206.00364).
- **François Rozet's work** on
  [score-based data assimilation](https://arxiv.org/abs/2306.10574) and
  [moment matching posterior sampling](https://arxiv.org/abs/2405.13712).
- **Geophysical implementations** by
  [Martin et al. (2025)](https://doi.org/10.1029/2025MS005063) and
  [Manshausen et al. (2025)](https://arxiv.org/abs/2406.16947).
- **Our earlier ocean reconstruction study**,
  [Monkman et al. (2026)](https://doi.org/10.22541/essoar.15003398/v3),
  evaluates separate, independent models for SSH and SST and explains the
  posterior-sampling algorithm for an oceanographic audience. The work
  presented in this repository builds on that study with several upgrades
  and improvements, including joint SSH/SST modeling; the joint model was
  not evaluated in the paper.

### Source projects

This repository was extracted from two research codebases. The network and
training data pipeline come from `cBottle-for-ocean-DA/ocean-examples`. The
MMPS sampler comes from `oBottle-for-SSH-decomposition/oBottle-inference`.
Its tests check MMPS against the exact posterior of a Gaussian prior. The
released checkpoints were trained in cBottle-for-ocean-DA. The OSSE tiles are
the three Atlantic test cases used there. The windowed (multi-tile) SDA used
for full 336×336 tiles is not included.

## Contributors

- **Tatsu Monkman** — research, model development, and repository direction.
- **Claude (Anthropic)** — AI coding contributor, assisting with code and documentation.
- **Codex (OpenAI)** — AI coding contributor, assisting with code and documentation.

Contributions are welcome, including bug reports, documentation corrections,
validation studies, and code improvements. Please open an issue or pull request.

## License

Apache-2.0 (`LICENSE.txt`), except the vendored azula subset
(`src/obottle_starter/inference/sda/_azula/`, MIT). See `THIRD_PARTY_LICENSES.md`.
