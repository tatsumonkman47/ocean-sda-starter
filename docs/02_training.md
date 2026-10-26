# 2. Training

Training uses complete fields as targets and adds noise to them on the fly.
You do not need paired satellite observations to train the released type of
prior. This page explains the objective, the settings worth changing, and
how to judge whether a model is learning useful ocean statistics.

For a first experiment, [notebook 02](../examples/02_train_small.ipynb) trains
a small model on generated fields. The commands for real data are in the
[training guide](quickstart/training.md).

## Training objective

Each training step takes a batch of clean samples `x` (in model units),
draws a noise level σ for each sample, adds noise `n ~ N(0, σ²)`, and trains
the denoiser to recover `x`:

```
loss = λ(σ) · ‖ D(x + n, σ) − x ‖²,      λ(σ) = (σ² + σ_data²) / (σ · σ_data)²
```

<p align="center">
  <img src="../examples/figures/a_training.png" alt="Training diagram: add noise to a clean patch, predict the clean field, compare with the target, and update network parameters">
  <br>
  <em>The training cycle. Each noisy input is paired with its original clean target. The diagram shows the squared-error term; the objective above also applies the noise-dependent weight lambda(sigma).</em>
</p>

The fields in this schematic are generated Gaussian examples with an
SSH-like spectrum; they are not llc4320 samples or outputs of a trained
SongUNet. The diagram illustrates where the data, prediction, and loss
enter the training calculation.

**Why weight the error?** The weight `λ(σ)` (from EDM) rescales the error at each noise level so that
it is of order one for an untrained network: without it, large-σ samples,
where the error is naturally large, would dominate. It does not make every
noise level contribute equally in practice; that also depends on how often
each σ is drawn (below), how well the model can do at that σ, and the data.
The loss is computed in `src/obottle_starter/core/nn/loss.py` (`EDMLoss`)
and the loop is `src/obottle_starter/training/train.py`.

There is no adversarial or spectral loss. Spatial spectra and other
statistics are learned through the denoising objective rather than imposed
as separate constraints.

## Sampling noise levels during training

`noise_distribution: log_uniform` draws `ln σ` uniformly between
`ln σ_min` and `ln σ_max` (0.0002 to 200), so each decade of σ gets the same
number of training examples. This reproduces the released training setup.
For ocean fields with red spectra, larger noise levels generally obscure
smaller-amplitude, finer-scale features first. The relation between noise
level and spatial scale depends on the data spectrum (page 1).

You can change this distribution in `EDMLoss`:

- `log_normal` (EDM's default, `ln σ ~ N(p_mean, p_std²)`, with
  `p_mean: -1.2`, `p_std: 1.2`) concentrates on σ ≈ 0.1–1. Its effect on accuracy at
  very large or small σ has not been tested in a controlled comparison
  here. Its draws are not limited to `[σ_min, σ_max]`. To use it:
  `noise_distribution=log_normal`, optionally with `p_mean=` and `p_std=`.
- `power` follows the ρ = 7 schedule used by EDM's sampler.

Choose `σ_max` large enough that noise dominates the normalized data at
the start of sampling. Choose `σ_min` below the amplitude of the smallest
features you want to resolve. An inadequate range leaves part of the
sampling process poorly represented during training.

## Main training settings

For an initial experiment, choose the variables, frame count, network
width, and batch size first. The remaining options let you refine that
setup. All defaults are described in `configs/train_base.yaml`.

| Setting | Effect | Typical change |
|---|---|---|
| `model_channels` | Network width; parameters grow roughly as its square | 32–64 for experiments, 96 for the released size |
| `per_frame_adaptive_scale` | Frame-time embedding as scale and shift after normalization instead of a shift before it | new models only; compare against `false` |
| `time_length` | Frames per sample; fixed for a trained model | 5 (cheaper) or 11 (longer windows) |
| `batch_size` | Samples per step per GPU; limited by GPU memory | lower for 11 frames or smaller GPUs |
| `lr`, `lr_rampup_steps` | Adam learning rate, reached after a linear warm-up | the default 2e-4 works for SongUNet |
| `num_steps` | Total steps | the released models used 195k–315k |
| `time_sampling_mode` and its keys | How far apart frames are (page 3) | match it to the spacing you will assimilate |
| `variables` | Which fields, their units, preprocessing and normalization ([Units and scaling](quickstart/units.md)) | change for new data; config only, no code |

## Weight averaging (EMA)

During training, each batch changes the model parameters slightly. We
smooth these changes before using the model to generate samples.

The saved sampling weights (`ema_state_dict`) are an exponential moving
average (EMA) of the training weights:
`w_ema ← 0.999 · w_ema + 0.001 · w`. This smooths fluctuations over roughly
1,000 optimizer steps and generally improves sample quality.
`load_checkpoint` loads these averaged weights.

The decay is per optimizer step, so the number of *samples* it averages
over is about `batch_size × GPUs / (1 − ema_decay)`; training prints it.
Changing the batch size or the number of GPUs changes that averaging interval, which
matters when comparing runs.

## Precision and speed

GPU training computes network outputs and gradients in bfloat16
(`torch.autocast`) while storing weights in float32. SDA evaluates the
network in float32 to limit numerical error in the Jacobian products used
by the linear solver. `load_checkpoint` disables fp16, and the denoiser
interface sets `force_fp32`.

Freezing the weights prevents their gradients from being computed; it does
not prevent differentiation with respect to the network input, which SDA
requires.

Measured throughput on one NVIDIA L40S at the released model size is
about 9.5 samples/s for 5 frames (batch 32) and 4.3 samples/s for 11 frames
(batch 8). The released joint model saw about
41 million samples (315k steps at batch 130), which would take weeks on one
L40S. For your own experiments:

- shrink the network (`model_channels: 32`, see `configs/train_structured.yaml`)
  or use fewer frames;
- use several GPUs: `torchrun --nproc_per_node=4 scripts/train.py ...`
  (data-parallel; `batch_size` is per GPU). Each process gets different
  samples: patch windows come from a per-process random stream, and the
  generated datasets are split between processes. At every checkpoint the
  processes check that their model and EMA weights are identical;
- split long training into several jobs with `start=auto` (or
  `start=resume`); see "Starting or continuing a training run" below.

## Starting or continuing a training run

The `start` setting controls whether training creates or continues a run:

| `start` | What happens |
|---|---|
| `fresh` (default) | A new run. Fails if `output_dir` already holds a run (a `config.yaml` or any checkpoint), so an old run is never silently overwritten. |
| `resume` | Continues `output_dir`'s run from `checkpoint_latest.pt` (or `resume_from`). Fails if there is nothing to resume. |
| `auto` | `resume` if `checkpoint_latest.pt` exists, otherwise `fresh`: for requeued Slurm jobs. A directory with only a `config.yaml` (an attempt that stopped before its first checkpoint) starts over; one with numbered checkpoints but no latest one stops and asks for `resume_from`. |

An explicit `resume_from` must exist in every mode; a misspelled path is an
error, never a silent fresh start.

### Continuing the same experiment

**Configuration checks on resume.** Reordering variables or
changing a normalization constant leaves every tensor shape unchanged but
changes what the run learns. On resume, the new config is compared with the
run's own: only runtime settings (`num_steps`, `log_every`,
`checkpoint_every`, `num_workers`, `data_dir`, `output_dir`, `validation`,
...; `RUNTIME_KEYS` in `core/config.py`) may differ. Anything else stops
with a list of the differences. `config.yaml` stays the config the run
started with, and permitted changes are appended to `config_changes.yaml`.
Nothing is written until these checks pass.

### Starting a different experiment from existing weights

**`init_from`** starts a *new* run from another checkpoint's weights, with a
fresh optimizer (it cannot be combined with `start=resume`). The source must
have the same variables, in the same order and normalization, and the same
network; the only allowed difference is new observation inputs for a
conditional model, which are added to the first convolution as zeros. The
log says how many tensors were copied and which were padded; nothing is
newly initialized.

## Missing data

Training requires finite values throughout the modeled domain.
`PatchDataset` and the loss raise an error at the first NaN or infinity.

A missing observation during reconstruction is fine: SDA is designed for
that. A missing value in a training target is a different problem, because
the network needs a finite clean field to corrupt and learn to recover.

Supporting coastlines or incomplete training fields requires changes to
both inputs and loss calculation. Distinguish land, which is outside the
modeled domain, from missing observations within it. Fill network inputs
with finite values and normalize the loss by the number of valid elements.
Excluding a location from the loss does not establish that the model learns
the correct distribution there.

An implementation should check finite losses and gradients for one missing
value, a missing variable, an entirely missing batch, and a coastline mask.

## Validation

Training loss varies with the sampled fields and noise levels. To tell
whether the model is improving, evaluate it on the same examples each time,
using data excluded from training.

Set `validation.every > 0` to evaluate fixed, held-out windows at regular
training intervals. The EMA model denoises them at fixed noise levels
(`validation.sigmas`) using fixed noise realizations. Weighted errors are
reported per variable and saved to `<output_dir>/validation.csv`. Holding
samples and noise fixed makes these errors comparable across training
steps.

For real patches, validation windows come from
`validation.time_range_start` and `validation.time_range_end`. Separate
these hours from training by at least the maximum window span to prevent
shared frames. Nearby patches are also correlated, so a random split of
windows may overestimate generalization. The temporal split is recorded
in `config.yaml`.

## Evaluating training progress

- **The loss is noisy and does not go to zero.** Each step draws different σ,
  and at large σ the clean state cannot be uniquely recovered. Watch the trend
  over thousands of steps, not single values, and use validation.
- **Look at samples.** Use `scripts/run_sda.py` with your checkpoint and
  `observations.mode=none` to generate fields without observations. Early
  models may produce overly smooth or speckled fields; look for the
  structures present in the training data.
- **Compare statistics.** `obottle_starter.inference.metrics` has spectra
  (`isotropic_psd`), cross-variable correlation and coherence
  (`spatial_correlation`, `isotropic_coherence`) and temporal decorrelation
  (`frame_correlation`). Compare prior samples with the truth. Too little
  small-scale power usually means under-training or too large `σ_min`; for a
  joint model, check the relationship between variables, not only each one's
  spectrum.
- **Start small.** `configs/train_structured.yaml` trains a small model on
  generated fields whose statistics are known exactly (unit variance, a given
  spectrum, drift and cross-variable correlation), so you can see what a
  working model reproduces before committing to long training runs on real
  data. `configs/train_smoke_test.yaml` (white noise) only checks that the
  code runs.

## Code reference

| What | Where |
|---|---|
| Training loop, EMA, checkpoints, DDP, start modes | `src/obottle_starter/training/train.py` |
| Config checks, variables, what may change on resume | `src/obottle_starter/core/config.py` |
| Held-out validation loss | `src/obottle_starter/training/validation.py` |
| Loss and noise distributions | `src/obottle_starter/core/nn/loss.py` |
| All options and defaults | `configs/train_base.yaml` |
