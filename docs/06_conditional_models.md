# 6. Conditional models (experimental)

> **Status:** implemented and smoke-tested (the code builds, trains, loads
> and samples), but no conditional model has been trained on real data or
> evaluated. Treat this page as a starting point for exploration.

## Two ways to use observations

The released models are **unconditional priors**: they learn what the
ocean looks like, never see observations, and data assimilation (page 4)
brings the observations in at inference. A **conditional model** instead
takes the observations as extra input channels and learns to approximate
`p(state | observations)` directly.

| | Unconditional prior + SDA | Conditional model |
|---|---|---|
| Observations enter | at inference, through guidance | as network inputs, in training and inference |
| New mask, `σ_y` or observed variable | works with no retraining | only reliable if training included similar ones |
| Cost per sampling step | MMPS: several network passes; Rozet: about two | one pass (M_3), two (M_4) |
| Training data | states only | states plus simulated observations of them |
| Main risk | guidance not propagating information well | training observations unlike the real ones |

Suresh Babu, Sadam & Lermusiaux (2026) compared four approaches on
quasi-geostrophic turbulence. They called the conditional model **M_3** and
the conditional model with classifier-free guidance **M_4**. These had the
lowest errors in their tests; guidance with Diffusion Posterior Sampling
(M_2) smoothed fine scales. They did not test MMPS, this repository's
default. See [the literature page](07_literature.md).

## How the inputs are built

For each channel in `conditioning.observe` the network gets:

1. **Values:** the observed value in model units at observed cells, 0
   elsewhere.
2. **Mask** (if `mask_channel: true`): 1 at observed cells, 0 elsewhere.

All the value channels come first, then all the mask channels, then the
usual frame-time encodings. Observations are per frame, so a swath that
moves between frames is represented naturally.

**Why a mask channel?** Without one, an unobserved cell (0) and an observed
value of 0 look identical, and in model units values near 0 are common.
Suresh Babu et al. used no mask; it is optional here so you can compare.

**The null condition** (all values and masks zero) means "nothing
observed". A conditional model trained with some null conditions also
works as an unconditional prior.

## Training a conditional model

Set up in the `conditioning` section of `configs/train_base.yaml`;
`configs/train_conditional.yaml` is a ready-made example.

```yaml
conditioning:
  mode: observations          # none = unconditional (default)
  observe: [SSH, SST]         # which variables are observed; any subset
  mask_channel: true
  training_observations:      # synthetic observations, fresh for every sample;
                              # each value is a number or a map per variable
    masks: stripes            # bernoulli (random cells) | stripes (random bands)
    coverage: {SSH: 0.1, SST: 0.3}
    stripe_width: {SSH: 30, SST: 60}
    stripes_per_frame: {SSH: 1, SST: 2}
    frame_probability: 1.0    # chance a variable is observed in a frame at all
    noise_std: {SSH: 0.005, SST: 0.1}   # physical units
    null_probability: 0.1     # > 0 trains the null branch CFG (M_4) needs
```

`null_probability` (drop the whole *observation* condition) is unrelated
to `label_dropout` (drop the unused global label vector, kept for the
released checkpoints). Only `null_probability` trains the branch CFG uses;
`guidance=cfg` refuses a model trained with `null_probability: 0`.

During training, each sample's target is observed through freshly drawn
masks with Gaussian noise (`training/obs_simulation.py`), and the model
learns to denoise given those observations. With `null_probability: 0.1`,
one sample in ten sees the null condition instead.

**Start from a released prior.** `init_from=checkpoints/joint_ssh_sst_T5.pt`
copies its weights and adds the new input channels with zero weights, so
training begins from a model that ignores the observations and reproduces
the prior exactly. The variables, their order and normalization, and the
network must match the source's; the log lists what was copied and padded. This should need far fewer steps than training from
scratch:

```bash
python scripts/train.py config=configs/train_conditional.yaml \
    data_dir=/path/to/patches init_from=checkpoints/joint_ssh_sst_T5.pt
```

### Choosing the training observations

This is the important decision: **the model learns the observing system it
is trained on.**

- `bernoulli` scatters isolated observed cells. Easy, but unlike any real
  instrument.
- `stripes` draws straight bands at random angles, a crude stand-in for
  swaths (SWOT, nadir tracks) and for large cloud gaps in SST.
- To use real coverage (SWOT and nadir footprints, cloud masks), replace
  `simulate_masks` in `training/obs_simulation.py` with a sampler that
  draws masks from a library of real footprints. The OSSE tiles hold a few
  examples; a real library would need many.
- Vary coverage and noise during training if you want a model that copes
  with a range of conditions. Values and masks alone do not tell the
  network how noisy an observation is: a model trained on a mixture of
  noise levels learns their average, and gives no control over the assumed
  uncertainty at inference. Supplying the noise level as another input
  would; it is not implemented. Likewise, a coarse product interpolated to
  the fine grid is not a set of independent fine-grid observations.
- Validate on observing systems the model was trained on *and* on shifted
  ones (another mask family, other coverage).

## Sampling

In `configs/sda.yaml`, `guidance.method` chooses how the observations are
used:

| `guidance.method` | What happens | Model |
|---|---|---|
| `conditional` | Observations enter only as inputs; plain sampling (M_3) | conditional |
| `cfg` | Classifier-free guidance with weight `classifier_free.weight` (M_4) | conditional, trained with `null_probability > 0` |
| `mmps`, `rozet` | Unconditional prior: guidance only. Conditional model: observations enter as inputs **and** through guidance (hybrid, unvalidated) | any |

Classifier-free guidance (Ho & Salimans, 2022) runs the model twice per
step, with and without the observations, and extrapolates:
`x̂ = x̂_uncond + w (x̂_cond − x̂_uncond)`. `w = 1` is the plain conditional
model; `w > 1` pushes harder towards the observations, often at the cost of
ensemble spread.

Combining a conditional model with MMPS or Rozet guidance is possible but
unvalidated, and is not simply "extra consistency". A model that already
approximates `p(x | y)` is not automatically improved by applying the
likelihood of the same `y` again: the same evidence counts twice, which can
over-concentrate the ensemble. The precise effect of this approximation
needs analysis and experiments. Conditioning on one source and guiding with
a *different*, independent source is a cleaner design. `assimilate` prints
a note for the hybrid and records the route in the output.

In Python:

```python
post = obs.assimilate(net, cfg, times=..., center_lon=..., truth=..., masks=...,
                      sigma_y={"SSH": 0.005, "SST": 0.1},
                      guidance="cfg", cfg_weight=1.5)
```

The model only takes observations of the channels in its
`conditioning.observe` list. Others are ignored by `conditional` and `cfg`
(with a warning) but still used by `mmps` and `rozet`.

## Try it: notebook 04

`examples/04_conditional_and_cfg.ipynb` runs the whole route on the
structured synthetic fields, without the data bundle: it trains a small
conditional model that sees noisy bands of one variable, with
`null_probability: 0.2` (`configs/train_structured_conditional.yaml`), then
samples it with `guidance="conditional"` and with CFG at weights from 0
(the null branch, i.e. the prior) to 4, and plots ensemble-mean error and
spread inside and outside the observed bands, for the observed and the
unobserved variable.

## What you can change without new code, and what needs code

Supported by configuration alone: which variables are observed
(`observe`), the mask family, coverage and noise of the training
observations, the mask channel, `null_probability`, starting from an
unconditional checkpoint (`init_from`), and the guidance route and weight
at sampling. All of these are **observation maps of model variables**:
values and masks on the model grid.

Anything else the network should see needs code. Two common cases:

| Experiment | Files to change |
|---|---|
| Give the model the observation-noise level, so one model covers several noise levels and `sigma_y` means something at sampling | `training/obs_simulation.py` (`training_condition`: draw the noise level per sample, build its channel); `core/obs_conditioning.py` (`observation_channels`, `build_obs_condition`: one more channel); `inference/assimilate.py` (build the same channel from `sigma_y`) |
| Other context: wind, a coarse product, a forcing field | `training/data.py` (the dataset returns it); `core/obs_conditioning.py` or a sibling module (turn it into input channels); `training/train.py` and `inference/assimilate.py` (pass it in, as the observation condition is) |

In both cases the input layer grows, so start a new run (or use
`init_from`, which pads new input channels with zeros), and add a test
like those in `tests/test_conditional.py`.

## Ideas to explore

- Train M_3 and M_4 with `stripes`, compare them with MMPS on the OSSE
  tiles, then test them on real masks they never saw.
- `mask_channel: true` against `false`.
- Observing only SSH (`observe: [SSH]`) in a joint SSH + SST model, to
  learn SST reconstruction from SSH.
- A library of real SWOT, nadir and cloud footprints for training masks.

## Where to look in the code

| What | Where |
|---|---|
| Building the observation inputs | `src/obottle_starter/core/obs_conditioning.py` |
| Synthetic training observations | `src/obottle_starter/training/obs_simulation.py` |
| Starting from an unconditional checkpoint | `src/obottle_starter/core/model.py`, `load_weights_padded` |
| Classifier-free guidance | `src/obottle_starter/inference/sda/cfg.py` |
| Sampling with a conditional model | `src/obottle_starter/inference/assimilate.py` |
