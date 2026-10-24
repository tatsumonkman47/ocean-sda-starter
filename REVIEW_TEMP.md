# Temporary review: preparing obottle-starter for graduate-student use

Review date: 2026-10-23. This is a proposal and review record, not a description of changes already implemented.

## 1. Purpose and scope

The intended handoff is a starter repository that students can use to train diffusion models, change architectural choices, experiment with score-based data assimilation (SDA) and learned conditioning, and apply these methods to different datasets. The main applications considered here are gap filling, coarse-to-fine reconstruction, and joint-field generation.

The existing repository is a useful foundation. The separation between `core/`, `training/`, and `inference/` is sensible. Normalization is saved with the checkpoint, the physical-unit assimilation wrapper reduces opportunities for mistakes, and analytic Gaussian tests provide a valuable mathematical reference. The documentation already explains more of the scientific motivation than many research repositories do.

The central recommendation is to preserve the released-model implementation as a compatibility baseline while providing a smaller, explicitly configurable route for new experiments. Most of the immediate benefit will come from better contracts, validation, and explanations rather than replacing the U-Net.

### Review basis and limitations

The review inspected the model construction and network implementation, loss, training loop, data loading, normalization, observation conditioning, SDA methods, configuration files, documentation, scripts, and tests. Scientific descriptions were checked against the primary EDM and MMPS sources listed at the end.

`python -m pytest -q` was attempted, but test collection stopped because the available Python environment did not contain PyTorch (`ModuleNotFoundError: No module named 'torch'`). Consequently:

- Findings about explicit code paths are based on source inspection.
- Distributed-training concerns are hypotheses requiring runtime reproduction.
- Existing tests were read, but their successful execution was not independently verified in this review.
- No model training, checkpoint comparison, or scientific benchmark was performed.

The review itself did not modify implementation files.

## 2. Priorities

| Priority | Work | Intended outcome |
|---|---|---|
| Before handoff | Input/config validation, missing-value policy, climatology behavior, resume semantics, DDP verification, mathematical wording | Students can run supported workflows without silently changing their meaning. |
| First development pass | Structured config, variable specifications, dataset contract, conditioning separation | Students can change datasets and experiments without editing unrelated internals. |
| Second development pass | Application tutorials, validation datasets, scientific diagnostics | Students can judge whether an experiment works scientifically. |
| Subsequent experiments | Architecture ablations, richer operators, alternative backbones, variable-length windows | Research changes are measurable and isolated. |

Avoid combining a major network rewrite with changes to normalization and sampling. Keeping those changes separate makes regressions much easier to identify.

## 3. Concrete issues to address before handoff

### 3.1 Grid shape is more restricted than the configuration suggests

Relevant code: [`core/model.py`](src/obottle_starter/core/model.py), [`core/nn/domain.py`](src/obottle_starter/core/nn/domain.py), and [`core/nn/networks.py`](src/obottle_starter/core/nn/networks.py).

The configuration exposes separate `H` and `W`, and model construction creates a `Plane(nx=W, ny=H)`. However, `SongUNet` takes a scalar resolution from the domain and builds each level with `img_size = (res, res)`. A student could reasonably infer that rectangular patches are supported when the network construction assumes square patches.

Immediate change:

- Validate `H == W` for this architecture.
- Validate that the patch dimensions are compatible with the number of downsampling stages.
- Report the actual resolutions, widths, and attention locations at model construction.
- Reject attention settings that do not select any available stage, unless an explicit empty selection is intended.

Later, rectangular support can track `(height, width)` through each level. This should include forward/backward tests on non-square patches and checks that skip connections and resampling preserve the intended geometry.

### 3.2 Missing targets need an explicit policy

Relevant code: [`core/nn/loss.py`](src/obottle_starter/core/nn/loss.py), `EDMLoss.__call__`.

The current validity mask reduces over the frame and pixel axes. A single NaN therefore invalidates an entire variable's window for that sample. The reduction then divides by the mean validity mask; a completely invalid batch divides by zero.

This is especially important when students move from open-ocean patches to coastlines or datasets with gaps. The present behavior is not a general masked-target loss.

Immediate change:

- Document that the baseline expects complete, finite targets over its modeled domain.
- Reject unsupported missing targets with a useful message.
- Detect batches with no valid targets before computing the loss.

For a future masked-domain implementation, carry an explicit domain/target-validity mask, fill invalid network inputs with finite values, and normalize the loss by the number or weight of valid target elements. Distinguish a structural domain mask, such as land, from missing training observations. Simply removing missing pixels from the loss does not establish that the model learns the desired distribution in those pixels.

Meaningful checks include one invalid target, an entirely invalid channel, an entirely invalid batch, and a coastline-shaped domain mask. Check both loss values and gradient finiteness.

### 3.3 Observation masking does not safely remove NaNs

Relevant code: [`core/obs_conditioning.py`](src/obottle_starter/core/obs_conditioning.py), `build_obs_condition`.

The helper multiplies values by a mask. Since `NaN * 0` is still NaN, this does not uphold its documented contract that unobserved entries may contain arbitrary values and are zeroed.

Use `torch.where(mask, values, 0)` for unobserved entries and validate finite values at observed entries. Keep this guarantee in the shared helper, even though the high-level assimilation wrapper already reconstructs zero-filled observation fields.

### 3.4 Climatology lookup silently clamps dates

Relevant code: [`core/normalization.py`](src/obottle_starter/core/normalization.py), `Normalizer.climatology`.

The implementation calculates days since the beginning of the stored climatology and clips indices to the stored interval. A date outside that interval therefore uses the first or last value. It does not automatically repeat by day of year.

This behavior may be necessary to reproduce the released model, but it is surprising for a reusable seasonal climatology and can produce a constant offset when applying a model to another year.

Recommended design:

- Preserve the existing rule in an explicitly named legacy normalization mode.
- Reject out-of-range dates in that mode unless endpoint clipping is explicitly requested.
- Provide a calendar/day-of-year mode for new models, with documented leap-day and interpolation rules.
- Store the climatology rule, asset identity, and checksum in checkpoint metadata.
- Do not silently switch existing checkpoints to a new rule.

Test dates inside and outside the original interval, year boundaries, and leap days.

### 3.5 Resume and initialization are currently too implicit

Relevant code: [`training/train.py`](src/obottle_starter/training/train.py).

Training automatically resumes from `output_dir/checkpoints/checkpoint_latest.pt` when it exists. This can override a requested `init_from`. The new config is written before the resume operation is validated, and there is no semantic comparison with the checkpoint config.

Tensor-shape compatibility is insufficient. Reordering two channels or changing normalization may leave all parameter shapes unchanged while changing the meaning of the run.

Use explicit modes for fresh training, resume, and initialization from weights. For resume:

- Compare architecture, variable order, normalization, preprocessing, and conditioning settings.
- Distinguish immutable scientific settings from permitted runtime changes such as logging frequency or total steps.
- Preserve the original resolved config and record permitted changes separately.
- Write run metadata only after validation succeeds.

For weight initialization, report exactly which parameters are loaded, padded, or newly initialized. Restrict padding to known input projections instead of treating arbitrary shape changes as automatically meaningful. Compare source and target variable/condition order, especially when initializing one conditional model from another.

### 3.6 Verify distributed training before advertising it as a standard workflow

Relevant code: [`training/train.py`](src/obottle_starter/training/train.py) and `TemporalAttention` in [`core/nn/networks.py`](src/obottle_starter/core/nn/networks.py).

Two concerns need runtime verification:

1. Temporal attention retains a learned discrete positional fallback that is unused when continuous `time_offsets` are supplied. DDP is constructed with default unused-parameter handling. Verify that multiple optimizer steps do not fail because parameters are absent from the active computation graph.
2. Each rank seeds initialization differently and creates its EMA copy before DDP synchronizes the model. Ensure EMA is initialized from synchronized model parameters on every rank, or explicitly define and enforce rank-zero-only EMA ownership.

A two-process CPU/Gloo smoke test is sufficient to exercise much of the synchronization logic; CUDA testing is still needed for the advertised GPU workflow. Run more than one iteration, because DDP reduction problems may surface on the next forward pass. Compare model and EMA state across ranks and verify checkpoint reload.

## 4. Configuration and checkpoint design

### 4.1 Group configuration by scientific responsibility

The current comments are helpful, but the flat training config combines dataset identity, model construction, legacy keys, conditioning, optimization, and numerical choices.

Keep OmegaConf; a more elaborate framework is not necessary. Add a structured schema and organize settings approximately as follows:

```yaml
data:          # source, variables, grid, splits, window sampling
model:         # backbone, widths, blocks, attention, embeddings
conditioning:  # context and learned observation inputs
diffusion:     # preconditioning and training noise distribution
training:      # optimizer, EMA, precision, checkpoints
evaluation:    # held-out data, sampling tasks, diagnostics
```

For inference, retain separate `observations`, `guidance`, and `sampler` sections. A sampler schedule and a training noise distribution are related choices, but they are not the same object.

### 4.2 Validate before loading data or allocating a large model

The current CLI checks missing required keys but does not enforce a complete schema. Add validation for:

- Unknown or misspelled keys.
- Variable names and unique ordering.
- Positive normalization scales and finite offsets.
- Positive noise bounds with `sigma_min < sigma_max`.
- Probability ranges, nonnegative observation noise, and valid solver settings.
- Supported patch shapes, stage counts, and attention locations.
- Even dimensions for sinusoidal encodings where required.
- Compatibility of conditional sampling and the checkpoint's conditioning inputs.
- CFG use when the null branch was actually trained.
- Conflicting resume and initialization modes.

Use error messages that say what the user can change, rather than allowing a low-level reshape or dictionary error later.

### 4.3 Replace parallel channel lists with variable specifications

At present, changing a variable can require editing `channels`, `outfields`, `num_channels`, `out_transform_list`, and `llc_standards` consistently. A generic variable specification would put related information together:

```yaml
variables:
  - name: SSH
    source_variable: ssh
    units: m
    normalization:
      kind: affine
      offset: 0.0
      scale: 0.051
```

Derive channel count and ordering from this list. Ordinary affine normalization should not require adding Python entries to `_SCALE_KEY` and `_UNITS`. Reserve custom transform implementations for genuinely different behavior.

Record preprocessing separately from normalization. For example, subtracting a 300 km low-pass SSH field changes the modeled variable; it is not merely a unit conversion. Grid spacing, projection, filtering, and climatology should be discoverable from a checkpoint.

### 4.4 Version the schema and preserve old checkpoints

Add a checkpoint/config schema version and a translator from the current flat configuration. Keep compatibility behavior explicit rather than adding increasingly scattered `cfg.get` fallbacks.

Retain the released-checkpoint numerical comparison as a regression gate. Module splitting or renamed parameters should either preserve state-dict keys or have an explicit migration with equivalence checks.

Normalization assets should be portable. A machine-specific path is not sufficient provenance; include an asset identifier/checksum and either package the asset or document how to obtain the exact one. The current repository-relative climatology lookup works for editable installs but needs a deliberate packaging strategy for ordinary installations.

## 5. Defaults and experiment presets

### 5.1 Separate teaching, smoke testing, and reproduction

The base config currently combines `data_dir: null`, which selects synthetic white noise, with a 300,000-step run. A student can accidentally launch a long training job on the wrong data source.

Provide explicit presets:

| Preset | Characteristics |
|---|---|
| `smoke_test` | Synthetic source explicitly selected, tiny model, very short run. |
| `student_baseline` | Modest model, real dataset required, validation enabled, manageable checkpoint cadence. |
| `released_joint` | Existing joint SSH/SST architecture, preprocessing, and normalization. |
| `released_ssh` | Existing SSH-only architecture and normalization. |

For a new baseline, set global label dimensions and label dropout to zero unless meaningful labels are supplied. Retain the existing label settings only in compatibility presets.

### 5.2 Use one source of defaults

The Python inference interfaces and CLI currently have different defaults, including MMPS iteration counts. A user should not unknowingly run a different numerical method by switching interfaces.

Construct shared settings objects or pass a validated settings structure through the API. Document deviations explicitly when a low-level function intentionally has a minimal default.

### 5.3 Treat scientific defaults as starting points

Retain the existing diffusion noise range and log-uniform training distribution as a reproducible reference. Do not describe them as universally appropriate for ocean fields or other datasets.

Likewise, an iteration count of four, 64 sampling steps, or a CFG weight of 1.5 should be described as a starting configuration. The accompanying tutorial should demonstrate a convergence or sensitivity check.

Precision should be configurable and reported. Training currently enables bfloat16 autocast on CUDA; expose a supported precision choice and explain why SDA differentiation may use float32 even when training uses mixed precision.

## 6. U-Net architecture and model extensibility

### 6.1 Keep the existing decomposition initially

The current architecture has an understandable structure:

1. Spatial convolutions and attention process individual frames.
2. Temporal attention exchanges information across frames.
3. Time and calendar embeddings describe when those frames occur.
4. EDM preconditioning wraps the raw network to produce a denoised estimate.

This is a useful teaching baseline. Improve its interface before introducing additional backbones.

### 6.2 Expose a small set of meaningful choices

Recommended controls:

- Width, stage multipliers, and blocks per stage.
- Spatial and temporal attention by downsampling factor or stage index.
- Attention head configuration.
- Temporal embedding routes.
- Residual conditioning style: `none`, `add_before_norm`, or `scale_shift_after_norm`.
- Calendar conditioning independently of relative frame times.
- Activation checkpointing and precision.
- Explicit padding/boundary behavior.

`per_frame_adaptive_scale` already exists in `SongUNet` but is not forwarded by the public model builder. Exposing it would enable a useful experiment without introducing a new implementation. It changes parameter shapes, so it is a new-model choice rather than an inference switch for released checkpoints.

Use attention stage indices to avoid a silent change when patch size changes. If absolute resolutions remain supported for compatibility, print their resolved stage mapping.

### 6.3 Explain temporal redundancy as an experimental question

The model has four time-related routes: offset maps, relative-time attention bias, per-frame embeddings, and calendar features. They have different purposes, but their combined benefit has not been established here by matched comparisons.

A useful student ablation holds data, seeds, compute budget, and evaluation tasks fixed while disabling one route at a time. Evaluate multiple frame spacings, prior temporal correlations, and reconstruction error. Avoid interpreting one training-loss curve as sufficient evidence.

### 6.4 Make boundary and resolution assumptions visible

Zero padding, periodic boundaries, and coastlines are different scientific assumptions. A periodic turbulence dataset may need a different boundary implementation from a regional ocean patch.

Changing pixel count is also different from changing physical resolution. A network trained on 4 km cells has learned relationships at that scale. Increasing `H` or changing input coordinates does not by itself establish transfer to a finer physical grid.

### 6.5 Refactor internals after compatibility checks are available

The large network file could eventually be split into:

```text
core/nn/
  blocks.py
  attention.py
  embeddings.py
  unet.py
  preconditioning.py
```

Keep imported/vendor provenance intact. Move legacy-only options out of the primary student-facing interface, while retaining a tested compatibility implementation where needed.

Define a small denoiser contract before adding another backbone: accepted state shape, noise-level semantics, context inputs, returned denoised state, and differentiability requirements for guidance. Avoid a large plugin system until there is a second concrete implementation that needs it.

## 7. Conditioning methods and their interpretation

### 7.1 Separate four meanings of conditioning

The current naming uses conditioning for several different mechanisms:

- Physical frame times and calendar context.
- Static fields or external forcing, such as bathymetry or wind.
- Observations presented as network inputs.
- Likelihood guidance applied during sampling.

Represent these as named concepts in the public interface. An internal adapter can still concatenate tensors for the existing network. Explicit names make it possible to say which information is dropped for CFG and which remains available to both branches.

For example, null observation conditioning should remove observation values, masks, and related observation metadata while retaining frame timestamps and any context the intended prior is conditioned on.

### 7.2 Do not silently reuse observations through two routes

The current assimilation wrapper feeds observations into conditional networks and also applies MMPS or Rozet when those methods are selected. The documentation describes the hybrid as additional consistency enforcement.

This needs a stronger qualification: a model already approximating `p(x | y)` is not automatically improved by applying the same likelihood again. Reusing the same evidence changes the inference procedure and can overconcentrate the result. The precise effect of the implemented approximation requires analysis and experiments.

Make standard routes explicit:

| Route | Network input | Sampling guidance |
|---|---|---|
| Prior + SDA | Context only | Observation likelihood |
| Learned conditional | Context and observations | None |
| CFG | Conditional and null-observation branches | CFG combination |
| Context-conditioned prior + SDA | Context such as forcing | Separate observation likelihood |
| Experimental hybrid | Observations also enter the network | Additional guidance, explicitly labeled unvalidated |

Conditioning on one source and assimilating a distinct source can be meaningful, but the intended conditional-independence assumptions should be stated.

### 7.3 Supply uncertainty when it varies

Values and masks alone do not tell the network whether an observation is precise or noisy. If training varies observation noise, supply its level or variance as a per-variable/per-pixel input or embedding.

Without this input, the network learns a distribution averaged over the training noise mixture. That may be a deliberate experiment, but it does not give explicit control over the assumed uncertainty at inference.

Likewise, coarse observations may require resolution, footprint, or operator metadata. An interpolated coarse image should not be silently treated as dense independent fine-grid observations.

### 7.4 Improve the synthetic observation interface

Keep Bernoulli and stripe masks as simple examples, but separate mask generation, observation operators, noise generation, and null-condition dropout.

Useful extensions include held-out footprint libraries, temporal persistence of coverage, per-sample coverage distributions, and varying noise levels. The validation set should include both matched observing systems and deliberately shifted ones.

Validate CFG training history. `label_dropout` is not the same as observation-condition dropout; the docs and configuration should make that distinction unmistakable.

## 8. Observation operators and application support

### 8.1 Make the state-to-observation map the primary extension point

`ObservationTerm` is a useful implementation for same-grid masked observations and weighted channel combinations. However, its common `(T, H*W)` mask convention makes arbitrary observation grids less natural.

A general observation interface should map state tensors to an observation vector, and keep the following metadata separate:

- Observation values and coordinates.
- Variable/type/group labels.
- Measurement units and normalization.
- Error variances or a supported covariance operator.
- Operator definition and parameters.
- Optional synthetic truth and noise, which are not required for real observations.

Start with linear/affine operators and diagonal errors, which match the current implementation. Do not imply that arbitrary nonlinear operators or correlated errors are already supported by the high-level API.

### 8.2 Map the intended applications to operators

| Application | Modeled state | Observation operator |
|---|---|---|
| Gap filling | Complete field or trajectory | Select observed cells. |
| Coarse-to-fine reconstruction | Fine-grid field | Average/filter and subsample to the coarse grid. |
| Joint-field generation | Multiple variables together | Select a subset or measure combinations. |
| Track/swath reconstruction | Complete gridded field | Interpolate/integrate at measurement locations. |

For coarse-to-fine reconstruction, train or obtain a prior at the desired fine resolution. Ensure the operator matches how the coarse product was formed, including averaging and filtering where relevant.

For joint-field generation, evaluate cross-variable relationships as well as separate marginal spectra. A plausible-looking SST field does not establish that its relationship with SSH is correct.

### 8.3 Define normalization in observation space

State normalization and observation normalization are distinct. If physical state is `u = S x + b` and physical observations satisfy `y_phys = H u + e`, the normalized inference problem must transform the complete observation equation consistently.

For an observation-space scaling `Q`, for example:

```text
y_model = Q (y_phys - H b)
A_model(x) = Q H S x
R_model = Q R_phys Q^T
```

This is particularly useful for mixed-variable measurements and averaged fields. Dividing by one state channel's standard deviation is not sufficient in every case. Record the chosen observation-space scaling so finite Krylov solves can be reproduced.

## 9. Dataset contracts, training, and scientific evaluation

### 9.1 Separate storage from sample construction

The current patch loader assumes hourly data, treats hour offsets as integer indices, selects the first data variable, and takes timestamps from the first field.

Introduce a dataset adapter that returns a consistent sample containing the state, timestamps, grid metadata, and optional validity/context fields. Keep temporal-window selection separate from file access.

Require explicit variable and dimension mappings. Validate cadence, alignment across variables, patch shape, coordinate orientation, and finite values. For irregular source times, select using timestamps rather than assuming hours equal indices.

The existing per-patch zarr loader should remain a reference adapter. A second small xarray adapter would demonstrate how to use a different dataset without rewriting training.

### 9.2 Treat split design as part of the experiment

Nearby tiles and overlapping temporal windows are correlated. Randomly splitting individual windows can make validation unrealistically easy.

Support spatial and temporal holdouts with buffers appropriate to patch overlap and window span. Store split definitions in the run metadata. Fit any learned normalization statistics on training data only.

### 9.3 Add evaluation beyond training loss

Recommended diagnostic groups:

- Denoising: validation loss by channel and noise range.
- Prior: means, variances, distributions, spectra, and temporal correlations.
- Joint prior: cross-variable correlations or cross-spectra relevant to the application.
- Reconstruction: observed and unobserved-region errors separately, plus observation-space misfit.
- Ensemble: coverage, rank diagnostics, spread, and error across multiple cases.
- Numerics: sensitivity to sampling steps and solver iterations; solver residuals where enabled.

Use a fixed small validation set for frequent checks and a separate held-out test set for conclusions. Synthetic-observation experiments should record both the noise actually added and the uncertainty assumed by inference.

### 9.4 Add a structured teaching dataset

White noise is appropriate for checking that training executes, but it cannot demonstrate spatial interpolation or cross-variable inference. Add a small reproducible dataset with spatial structure and known relationships, such as correlated smooth fields or moving synthetic features.

Keep the exact Gaussian examples as analytic verification, and distinguish those from demonstrations intended to teach qualitative behavior.

### 9.5 Improve reproducibility metadata

Save RNG states, resolved config, dataset/split identity, code revision when available, dependency versions, and normalization asset checksums. Exact continuation with worker-prefetched random samples requires additional care; do not promise bitwise resume merely because the model and optimizer are saved.

EMA currently uses a fixed decay per optimizer step. Changing batch size or world size changes its sample horizon. Either define EMA in processed samples or report its effective horizon clearly so experiments can be compared.

## 10. Notation and mathematical explanation

### 10.1 Use one shared notation table

Physical time and diffusion time are especially easy to confuse in a model of trajectories. Prefer physical timestamps `tau_i` and diffusion noise level `sigma` in introductory explanations.

| Symbol | Meaning |
|---|---|
| `u` | State in physical units. |
| `x = N(u)` | State in model units. |
| `tau_i` | Physical timestamp of frame `i`. |
| `sigma` | Diffusion noise standard deviation. |
| `epsilon ~ N(0, I)` | Standard Gaussian diffusion noise. |
| `z_sigma = x + sigma epsilon` | Noisy normalized state. |
| `F_theta` | Raw U-Net. |
| `D_theta` | EDM-preconditioned denoiser. |
| `c` | Context, such as frame times or forcing. |
| `y = h(x) + e` | Observation model in explicitly declared units. |
| `R` | Observation-error covariance. |
| `J_theta` | Jacobian of the denoiser with respect to its noisy input. |

Use `R = diag(sigma_y,j^2)` for independent errors that can differ by observation. `sigma_y^2 I` is the special case of equal errors.

Document `(batch, variable, frame, row, col)` at public boundaries and `(batch, variable, frame, pixel)` inside the existing network. Specify flattening order and distinguish feature channels from physical variables.

### 10.2 Explain the denoising objective and score connection precisely

For the variance-exploding corruption used here:

```text
z_sigma = x + sigma epsilon
D_theta(z_sigma, sigma, c) approximately equals E[x | z_sigma, c]
score_theta = (D_theta(z_sigma, sigma, c) - z_sigma) / sigma^2
```

The expected-clean-state interpretation is the population optimum of the denoising squared-error objective. Tweedie's identity relates that mean to the score. Calling the expected-value interpretation itself “Tweedie's formula” skips this distinction.

Show the raw U-Net and preconditioned denoiser separately:

```text
D_theta(z, sigma, c)
  = c_skip(sigma) z
    + c_out(sigma) F_theta(c_in(sigma) z, c_noise(sigma), c)
```

Explain which conditioning maps are scaled with the noisy state and which are supplied separately, following the actual implementation.

### 10.3 Separate exact identities from approximations in SDA

For the exact conditional-mean denoiser under additive Gaussian corruption:

```text
Cov[x | z_sigma, c] = sigma^2 d E[x | z_sigma, c] / d z_sigma
```

The learned denoiser only approximates that mean. MMPS then uses a moment-matched approximation to the conditional distribution. Finally, finite Krylov iterations and a discretized sampler introduce numerical error.

These are separate approximation layers:

1. Learned prior/denoiser error.
2. Moment-matching approximation for a generally non-Gaussian conditional distribution.
3. Truncated linear solve.
4. Finite-step sampling.

A linear observation operator does not make all four layers exact. Gaussian tests establish correctness in a tractable special case; they do not establish calibrated posterior sampling for learned ocean priors.

### 10.4 Connect autodiff code to the mathematics

Add a comment beside each Jacobian product explaining its direction and shape. In particular:

```text
autograd.grad(output, input, v) computes J^T v
```

It does not compute `J v`. The ideal denoiser's covariance identity supplies symmetry; a learned Jacobian need not be symmetric or positive semidefinite. This makes the solver assumptions understandable and motivates the GMRES default without claiming GMRES guarantees accurate inference for every operator.

Distinguish network-parameter gradients from input gradients: freezing parameters is compatible with differentiating the denoiser output with respect to its input for guidance.

### 10.5 Explain uncertainty in observation space for the Rozet heuristic

The implementation adds a scalar diffusion-dependent variance term directly in observation space. Replacing state covariance by `gamma sigma^2 I` and then propagating it through an arbitrary linear operator would instead produce `gamma sigma^2 A A^T`.

These coincide for appropriate unit-weight selection operators, but not generally for smoothing, averaging, or weighted combinations. Document the implemented heuristic and its assumptions rather than presenting the general-operator interpretation as exact.

## 11. Specific documentation corrections

| Existing wording or implication | Recommended replacement |
|---|---|
| Noise level corresponds directly to spatial wavelength. | Noise level controls signal-to-noise ratio. Its relationship with wavelength depends on the data spectrum. |
| Large noise always fixes large spatial scales first. | This is a useful intuition for some fields, not a guaranteed ordering by wavelength. |
| Log-normal training under-trains the noise levels ocean fields need. | This is an empirical hypothesis. The current log-uniform choice reproduces the released training setup. |
| EDM weighting makes every noise level contribute equally. | It rescales the denoising objective; effective contributions also depend on the noise sampling distribution, model error, and data. |
| MMPS uses the covariance exactly and nothing is tuned. | The covariance identity is exact for the ideal denoiser; learned moments, moment matching, and numerical settings introduce approximations and choices. |
| Linear observations make MMPS exact. | Linear observations simplify the update, but a general non-Gaussian prior still requires approximation. |
| `gamma=0.01` makes the model 100 times more certain than it really is. | It reduces the heuristic variance contribution by a factor of 100 relative to `gamma=1`; it does not identify true uncertainty. |
| Ensemble spread is the model's uncertainty. | Spread describes variability under the learned model and chosen sampler; calibration needs independent evaluation. |
| Members should fit observations to about `sigma_y`. | Observation-space residuals are useful diagnostics, but their expected behavior depends on posterior uncertainty, prior constraints, and the observation model. |
| Every public function accepts and returns physical units. | The high-level `assimilate` interface uses physical units; low-level SDA interfaces use declared model/observation units. |
| Fine-tuning is unsupported. | `init_from` and padded weight loading already implement initialization from released weights; update the stale extensions table. |

Additional cleanup:

- Fix the stale checkpoint-module reference in `inference/sda/denoiser.py`.
- Replace references to unavailable parent-project markdown files with self-contained explanations or accessible links.
- Make “real” coverage in the OSSE CLI clearly mean real instrument masks applied to synthetic truth, rather than real measured values.
- Distinguish nominal maximum sampled span from the actual first-to-last frame span consistently across config comments and data documentation.
- Identify parent-project measurements as historical evidence, with enough settings to interpret them; avoid presenting them as reproduced starter benchmarks.
- Keep literature-specific labels such as M_3/M_4 in the comparison discussion, while using descriptive names in the primary API/tutorial path.

## 12. A consistent style for comments and design rationale

For each consequential option, document four things:

> **Purpose:** What problem does this option address?
>
> **Current choice:** What does this repository use?
>
> **Evidence:** Is this inherited from a checkpoint, analytically tested, empirically compared, or untested?
>
> **Changing it:** Does the change require retraining, alter parameter shapes, or affect only inference?

For example:

> Relative-time attention bias tells the model how far apart two frames are. The released models use sinusoidal features projected to one bias per head. This preserves compatibility; its incremental value relative to the other time inputs has not been established by a matched ablation here. Changing its dimensions requires a new model or an explicit weight migration.

Prefer comments that explain units, assumptions, shape changes, and reasons for a choice. Avoid comments that only restate an assignment. Mathematical helper docstrings should specify inputs, outputs, units, and whether an expression is exact or approximate.

## 13. Proposed student tutorials

### Tutorial 1: Train and inspect a small prior

- Use a small structured dataset that runs without the external ocean bundle.
- Show normalized and physical data side by side.
- Visualize one corruption/denoising example at several noise levels.
- Train briefly, sample, and compare distributions and spectra.
- Explain what the smoke test checks versus what scientific validation checks.

### Tutorial 2: Gap filling with a fixed prior

- Define a mask operator and synthetic observation noise.
- Compare prior samples and guided samples.
- Report errors separately inside and outside the mask.
- Change assumed observation noise while holding the actual observations fixed.
- Check sensitivity to sampling steps and Krylov iterations.

### Tutorial 3: Coarse-to-fine reconstruction

- Define averaging/filtering and subsampling explicitly.
- Generate coarse observations from fine truth.
- Compare reconstruction with a simple interpolation baseline.
- Evaluate coarse observation consistency and fine-scale statistics separately.
- Explain why interpolation is not a measurement operator and why a fine-grid prior is needed.

### Tutorial 4: Joint-field inference and learned conditioning

- Train or load a joint prior and observe only one variable.
- Evaluate both variables and their relationship.
- Compare prior + SDA, learned conditioning, and CFG under matched observations.
- Show the role of null-condition dropout and observation-noise inputs.
- Hold out an observation-mask family to demonstrate distribution shift.

Each tutorial should state its expected output, approximate compute requirements, and which conclusions a short run cannot support.

## 14. Suggested implementation sequence and acceptance checks

### Phase 1: Make the existing workflows dependable

Implement validation, finite-value handling, explicit resume semantics, and legacy climatology safeguards. Correct the documentation claims. Reproduce and resolve the DDP concerns.

Acceptance checks: supported configurations run; unsupported shapes/data fail early; NaNs outside observation masks are harmless; invalid observed values fail clearly; resume rejects semantic changes; multi-process training completes multiple steps and reloads its checkpoint.

### Phase 2: Introduce reusable interfaces

Add structured settings, variable specifications, a checkpoint schema translator, a dataset sample contract, and named context/observation inputs. Keep legacy behavior available through explicit adapters.

Acceptance checks: released checkpoints retain their reference outputs; a new variable with affine normalization requires configuration only; Python and CLI runs resolve to the same settings; a second dataset adapter can use the existing training loop.

### Phase 3: Add scientific examples and diagnostics

Add the structured dataset, held-out splits, and gap-filling/coarse-to-fine/joint-field examples. Preserve analytic Gaussian tests and extend them when introducing new operator behavior.

Acceptance checks: tutorial outputs are reproducible at an appropriate tolerance; observation normalization is consistent; diagnostics distinguish observed from unobserved errors; evaluation reports uncertainty calibration across multiple cases.

### Phase 4: Enable controlled architecture research

Expose temporal conditioning variants, attention head settings, and boundary choices. Refactor the network internals only with compatibility tests in place. Add alternate backbones or variable-length windows when a concrete experiment needs them.

Acceptance checks: each ablation changes one documented mechanism; results record effective compute and sample counts; checkpoint compatibility is explicit; performance claims are supported by matched evaluations.

## 15. Primary scientific references

- Karras et al., [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364): separation of architecture, preconditioning, noise distributions, and sampling choices.
- Rozet and Louppe, [Score-based Data Assimilation](https://arxiv.org/abs/2306.10574): trajectory inference with diffusion priors and observation guidance.
- Rozet et al., [Learning Diffusion Priors from Observations by Expectation Maximization](https://arxiv.org/html/2405.13712v5): MMPS derivation, moment matching, and Tweedie identities. Cite the version when tying documentation to particular equations.

These sources support the mathematical explanations. They do not establish which architecture, noise distribution, or conditioning mechanism is best for the datasets students will use; those remain experimental questions.
