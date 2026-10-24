# 2. Training

## The objective in one paragraph

Each training step takes a batch of clean samples `x` (in model units),
draws a noise level σ for each sample, adds noise `n ~ N(0, σ²)`, and asks
the denoiser to recover `x`:

```
loss = λ(σ) · ‖ D(x + n, σ) − x ‖²,      λ(σ) = (σ² + σ_data²) / (σ · σ_data)²
```

The weight `λ(σ)` (from EDM) rescales the error at each noise level so that
it is of order one for an untrained network: without it, large-σ samples,
where the error is naturally large, would dominate. It does not make every
noise level contribute equally in practice; that also depends on how often
each σ is drawn (below), how well the model can do at that σ, and the data.
The loss is computed in `src/obottle_starter/core/nn/loss.py` (`EDMLoss`)
and the loop is `src/obottle_starter/training/train.py`.

That is the whole objective. There is no adversarial term and no spectral
loss: the model learns the full distribution, spectra included, by learning
to denoise at every scale.

## Which noise levels to train on

`noise_distribution: log_uniform` draws `ln σ` uniformly between
`ln σ_min` and `ln σ_max` (0.0002 to 200), so each decade of σ gets the same
number of training examples. This reproduces the released training setup.
σ sets the signal-to-noise ratio, and for red-spectrum ocean fields that
roughly corresponds to a scale: at σ = 50 the noise hides everything except
the patch-scale structure, at σ = 0.01 only grid-scale detail is uncertain.
That correspondence depends on the data's spectrum, so treat it as an
intuition.

Alternatives in `EDMLoss`:

- `log_normal` (EDM's default, `ln σ ~ N(p_mean, p_std²)`, with
  `p_mean: -1.2`, `p_std: 1.2`) concentrates on σ ≈ 0.1–1. Whether that
  under-trains the large and small σ these ocean fields need is a
  hypothesis that has not been tested by a matched comparison here. Its
  draws are not limited to `[σ_min, σ_max]`. To use it:
  `noise_distribution=log_normal`, optionally with `p_mean=` and `p_std=`.
- `power` follows the ρ = 7 schedule used by EDM's sampler.

Why the range matters: `σ_max` must be well above the largest signal in
model units, or sampling starts from "noise" that still contains structure
the model never learned to create. `σ_min` must be below the smallest
detail you care about, or that detail is never trained.

## The settings that matter most

Every option is described in `configs/train_base.yaml`. The ones you are
most likely to change:

| Setting | Effect | Typical change |
|---|---|---|
| `model_channels` | Network width; parameters grow roughly as its square | 32–64 for experiments, 96 for the released size |
| `per_frame_adaptive_scale` | Frame-time embedding as scale and shift after normalization instead of a shift before it | new models only; compare against `false` |
| `time_length` | Frames per sample; fixed for a trained model | 5 (cheaper) or 11 (longer windows) |
| `batch_size` | Samples per step per GPU; limited by GPU memory | lower for 11 frames or smaller GPUs |
| `lr`, `lr_rampup_steps` | Adam learning rate, reached after a linear warm-up | the default 2e-4 works for SongUNet |
| `num_steps` | Total steps | the released models used 195k–315k |
| `time_sampling_mode` and its keys | How far apart frames are (page 3) | match it to the spacing you will assimilate |
| `variables` | Which fields, their units, preprocessing and normalization (root README, "Units and normalization") | change for new data; config only, no code |

## Weight averaging (EMA)

The weights that training updates jump around from step to step. The
saved model (`ema_state_dict`) is instead an exponential moving average,
updated each step as `w_ema ← 0.999 · w_ema + 0.001 · w`, so it averages
roughly the last 1,000 steps. EMA weights give noticeably better samples
and are what `load_checkpoint` loads.

The decay is per optimizer step, so the number of *samples* it averages
over is about `batch_size × GPUs / (1 − ema_decay)`; training prints it.
Changing the batch size or the number of GPUs changes that horizon, which
matters when comparing runs.

## Precision and speed

On a GPU, the forward and backward passes run in bfloat16 (`torch.autocast`)
and the weights stay in float32. SDA runs the network in float32
(`load_checkpoint` turns fp16 off and the denoiser bridge passes
`force_fp32`): guidance differentiates the network's output with respect to
its input, and low-precision Jacobian products would add noise to the
linear solve. (Freezing the parameters, as `load_checkpoint` does, only
stops gradients with respect to the weights; gradients with respect to the
input still flow.) Measured on one NVIDIA L40S at the released model size:
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
  `start=resume`); see "Starting, resuming and initializing" below.

## Starting, resuming and initializing

How a run starts is never implicit (`start` in the config):

| `start` | What happens |
|---|---|
| `fresh` (default) | A new run. Fails if `output_dir` already holds a run (a `config.yaml` or any checkpoint), so an old run is never silently overwritten. |
| `resume` | Continues `output_dir`'s run from `checkpoint_latest.pt` (or `resume_from`). Fails if there is nothing to resume. |
| `auto` | `resume` if `checkpoint_latest.pt` exists, otherwise `fresh`: for requeued Slurm jobs. A directory with only a `config.yaml` (an attempt that stopped before its first checkpoint) starts over; one with numbered checkpoints but no latest one stops and asks for `resume_from`. |

An explicit `resume_from` must exist in every mode; a misspelled path is an
error, never a silent fresh start.

**Resuming checks meaning, not just shapes.** Reordering two variables or
changing a normalization constant leaves every tensor shape unchanged but
changes what the run learns. On resume, the new config is compared with the
run's own: only runtime settings (`num_steps`, `log_every`,
`checkpoint_every`, `num_workers`, `data_dir`, `output_dir`, `validation`,
...; `RUNTIME_KEYS` in `core/config.py`) may differ. Anything else stops
with a list of the differences. `config.yaml` stays the config the run
started with, and permitted changes are appended to `config_changes.yaml`.
Nothing is written until these checks pass.

**`init_from`** starts a *new* run from another checkpoint's weights, with a
fresh optimizer (it cannot be combined with `start=resume`). The source must
have the same variables, in the same order and normalization, and the same
network; the only allowed difference is new observation inputs for a
conditional model, which are added to the first convolution as zeros. The
log says how many tensors were copied and which were padded; nothing is
newly initialized.

## Missing data

The baseline expects complete, finite targets over its modeled domain:
`PatchDataset` and the loss stop with an error at the first NaN or inf.
(The loss inherited from cBottle silently dropped a whole variable's window
for a single NaN and divided by zero for an all-NaN batch.) Coastlines or
gappy products need more than removing pixels from the loss: a structural
domain mask (land) is different from missing training observations, the
network's inputs must be filled with finite values, and the loss
normalized by the number of valid elements. Even then, leaving pixels out
of the loss does not by itself show the model learns the right distribution
there. Check one invalid target, an invalid channel, an all-invalid batch
and a coastline-shaped mask, for both the loss and finite gradients.

## Validation

`validation.every > 0` evaluates a fixed set of held-out windows every so
many steps: the EMA weights denoise them at fixed noise levels
(`validation.sigmas`) with fixed noise, and the weighted error is printed
per variable and appended to `<output_dir>/validation.csv`. Unlike the
training loss, these numbers are comparable between steps and show which
noise levels and variables are learned. With real patches the held-out
windows come from `validation.time_range_start`/`time_range_end`, which
must be separated from the training hours by at least the longest window
span (otherwise windows from the two sets share frames). Nearby patches are
also correlated, so a temporal hold-out is more honest than a random split
of windows; record the split with the run (it is in `config.yaml`).

## Is it working?

- **The loss is noisy and does not go to zero.** Each step draws different
  σ, and at large σ the clean state genuinely cannot be recovered. Watch the
  trend over thousands of steps, not single values, and use validation.
- **Look at samples.** The quickest check is prior samples from your
  checkpoint:
  `python scripts/run_sda.py ckpt=runs/<name>/checkpoints/checkpoint_latest.pt tile=... output=... observations.mode=none`.
  Early models produce smooth or speckled fields; a trained one produces
  eddies and fronts.
- **Compare statistics.** `obottle_starter.inference.metrics` has
  spectra (`isotropic_psd`), cross-variable correlation and coherence
  (`spatial_correlation`, `isotropic_coherence`) and temporal decorrelation
  (`frame_correlation`). Compare prior samples with the truth. Too little
  small-scale power usually means under-training or too large `σ_min`; for
  a joint model, check the relationship between variables, not only each
  one's spectrum.
- **Start small.** `configs/train_structured.yaml` trains a small model on
  generated fields whose statistics are known exactly (unit variance, a
  given spectrum, drift and cross-variable correlation), so you can see
  what a working model reproduces before spending GPU weeks on real data.
  `configs/train_smoke_test.yaml` (white noise) only checks that the code
  runs.

## Where to look in the code

| What | Where |
|---|---|
| Training loop, EMA, checkpoints, DDP, start modes | `src/obottle_starter/training/train.py` |
| Config checks, variables, what may change on resume | `src/obottle_starter/core/config.py` |
| Held-out validation loss | `src/obottle_starter/training/validation.py` |
| Loss and noise distributions | `src/obottle_starter/core/nn/loss.py` |
| All options and defaults | `configs/train_base.yaml` |
