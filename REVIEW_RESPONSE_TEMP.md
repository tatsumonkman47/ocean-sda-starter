# Temporary response to REVIEW_TEMP.md

Date: 2026-10-23. This records what was changed in response to the review, how each change addresses the review's points, how it was checked, and what was deliberately left for later. Section numbers (§) refer to `REVIEW_TEMP.md`.

## 1. Scope agreed before starting

- **Do now:** everything in §3 (issues before handoff), config validation (§4.2), one source of defaults (§5.2), exposing `per_frame_adaptive_scale` (§6.2), the mathematical and wording corrections (§10–§11), and a separate smoke-test preset (§5.1).
- **Also requested:** the calendar climatology mode (§3.4, full version), variable specifications (§4.3), scientific diagnostics (§9.3) and a structured teaching dataset (§9.4).
- **Constraints:** the released checkpoints must keep giving the same outputs. Existing configs could change, since nothing had been run with this repository yet. The code base should not become much more complex.

## 2. How it was verified

| Check | Result |
|---|---|
| Baseline before any change | 39 tests passed, including `test_released_checkpoints.py` (both released checkpoints reproduce the reference outputs of the original training code). The repository had no commits, so a snapshot of the original was kept for comparison. |
| Full suite after the changes | **92 passed** (39 original tests, updated for the new API, plus 53 new ones). The released-checkpoint regression test passes unchanged. |
| Two-process DDP training on CPU (gloo) | Passes: 4 steps, replica check at every checkpoint, then resume to step 6 (`tests/test_distributed.py`). |
| Both DDP concerns (§3.6) reproduced | Each fix was reverted and the DDP test rerun. Without freezing the unused parameters, training crashed on the second step (`RuntimeError: Expected to have finished reduction in the prior iteration ... parameters that were not used in producing loss`). With the original per-process seeding, the new replica check failed because EMA weights differed between processes. |
| Command-line scripts on CPU | Typos, invalid grids and invalid SDA options fail before anything runs, with messages naming the option to change. The smoke-test preset trains. Rerunning into an existing run directory is refused. |
| `examples/02_train_small.ipynb` | Code cells run end to end (shortened run) on the structured data. |
| Lint | `ruff format` clean. `ruff check` reports the same findings as the original code under the installed ruff 0.17 (its newer default rules flag pre-existing patterns, such as `Optional[...]`, in both). |
| SDA regression on GPU (job 20773128) | The released joint model on the Mid-Atlantic tile, run with the original code (snapshot) and with the refactored code under identical settings and seeds, gives **bitwise-identical ensembles**: max \|new − old\| = 0.0 for SSH and SST. |
| Two-GPU DDP with NCCL (job 20773129) | 40 steps, replica checks at steps 20 and 40, resume to step 60, then prior samples from the resulting checkpoint through `run_sda.py`. All completed. |
| Real-data training with validation (jobs 20773128, 20774510) | The first attempt failed at data loading: the validation hours I chose (150 h) could not hold an 11-frame window of up to 336 h. The config check now rejects such ranges before any data is read. Rerun with valid hours: 100 steps, validation at 50 and 100, `validation.csv` written. This checks the plumbing only (learning rate still warming up). |
| Structured preset on GPU (job 20773473) | `train_structured.yaml` (5,000 steps, 12 min on one L40S). Held-out loss at σ = 1 fell from 0.92 to 0.34. Prior samples: std 0.96 and 0.98 (data: 1), correlation between A and B **0.804 (data: 0.8)**, spectra within about 15% of the truth except at the two largest scales (0.66 and 0.77). SDA observing only A on 20% of cells, over 8 cases: B, never observed, has RMSE 0.73, against about 1.41 for prior samples alone, and error/spread 1.01. |

GPU scripts, logs and outputs are in `/scratch/tm3076/obottle-starter-verify/refactor/`.

## 3. Point-by-point

### §3.1 Grid shape

- **Validated before the network is built** (`core/config.py`, `validate_model`, called by `build_model`): `H == W`; `H` divisible by `2^(levels − 1)`; every entry of `attn_resolutions` and `temporal_attn_resolutions` must name an existing level (an empty list is still allowed). The error lists the available grid sizes.
- **Reported at the start of training** (`describe_architecture`): each level's grid size, width and blocks, and where spatial and temporal attention sit, including the bottom block's spatial attention, which is always on.
- **Not done:** rectangular support, and attention given by stage index instead of absolute size. Absolute sizes are kept because the released checkpoints use them; they are now validated and printed, so a patch-size change can no longer silently remove attention.

### §3.2 Missing targets

- **Policy:** the baseline expects complete, finite targets. `PatchDataset` stops at the first NaN or inf, naming the store and frames. The loss also refuses non-finite targets, which covers batches with no valid targets.
- The misleading NaN-masking code in the vendored `EDMLoss` was replaced by the explicit check. For finite data it computes exactly the same loss.
- `docs/02_training.md` ("Missing data") explains what a real masked-domain loss would need, following the review's text.
- **Not done:** a masked-target loss.

### §3.3 Observation masking with NaNs

- `build_obs_condition` now uses `torch.where(mask, values, 0)` and raises if an *observed* entry is not finite.
- `assimilate` deliberately passes NaN at unobserved entries (the same field is stored in its output), so the fixed path is exercised in normal use.
- Tests cover NaN and inf outside the mask (the output equals the zero-filled one) and NaN inside the mask (error).

### §3.4 Climatology

- **New `Climatology` class** (`core/normalization.py`) with two rules:
  - **`elapsed_days`** (the released rule): whole days since the file's first day. Dates outside the file's record are an error by default. `out_of_range: clip` restores the old endpoint behaviour, but only when chosen explicitly.
  - **`day_of_year`** (for new models): looks up the same calendar day in any year. Days that appear twice in the 377-day file use the mean of the two values. Feb 29 uses the file's own Feb 29, falling back to the mean of Feb 28 and Mar 1. Values are piecewise constant per UTC day, with no interpolation; this is documented.
- **Released checkpoints are not switched silently:** their config translates to `lookup: elapsed_days, out_of_range: error`. The file covers 2011-09-13 to 2012-09-23, which includes all training data and all three OSSE tiles, so no supported workflow changes. Dates outside it now fail with a message offering the two explicit alternatives.
- **Checksum:** training records the file's SHA-256 in the config, so it is saved in every checkpoint and checked when the file is first read. A resume also refuses if the file has changed.
- **Changing the rule at inference** goes through `sda.yaml`, `normalization.climatology.{path, lookup, out_of_range}`, and is reported as a difference from training.
- **Tests:** dates inside and outside the record, both endpoints, a year boundary, a leap day, a day present twice, and the checksum.
- **Not done:** packaging the climatology for non-editable installs (it is still found relative to the repository).

### §3.5 Resume and initialization

- **Explicit `start`:**
  - `fresh` (default) refuses an `output_dir` that already holds a run;
  - `resume` refuses if there is nothing to resume;
  - `auto` resumes if possible, for requeued jobs.
  - `init_from` cannot be combined with `start=resume`, and `resume_from` cannot be combined with `start=fresh`.
- **Resume compares meaning, not shapes:** the new config is compared with the checkpoint's (`run_differences`). Only `RUNTIME_KEYS` may differ (`num_steps`, `log_every`, `checkpoint_every`, `num_workers`, `data_dir`, `output_dir`, `validation`, ...). Anything else, including variable order, normalization, network, noise, conditioning and optimizer settings, stops with a list of the differences.
- **Run metadata:** `config.yaml` stays the config the run started with. Permitted changes are appended to `config_changes.yaml`. Nothing is written until all checks pass.
- **`init_from`:**
  - the source's variables (names, order, units, normalization), network settings and conditioning must match, except for adding observation inputs to an unconditional model;
  - padding is restricted to the input projection (`*_conv_embed.module.weight`);
  - the log reports how many tensors were copied and which were padded; nothing is newly initialized, because the parameter names must match exactly.
  - Resuming a run with an `init_from` different from the one it started with is refused.

### §3.6 Distributed training

The review's two hypotheses were both confirmed (see §2).

1. **Unused parameters:** temporal attention's discrete positional table never gets a gradient, because training always passes frame times. Under DDP this crashed on the second step, so multi-GPU training could not have worked before. The table is now frozen during training (it stays in the state dict for compatibility), and the optimizer only receives trainable parameters.
2. **EMA initialization:** each process seeded its own weights and copied its EMA before DDP synchronized them. Saved checkpoints were not affected, because only rank 0 saves and its EMA matched the weights DDP broadcasts; but the other ranks' EMA was wrong. Now every process uses the same seed for the initial weights (then per-process seeds for data and noise), and the EMA is copied after any loading. At every checkpoint the processes compare checksums of model and EMA weights and stop if they differ.

The GPU (NCCL) result is in §2.

### §4.2 Validation before loading data

- **`validate_training`** collects every problem into one error message:
  - unknown variable keys and duplicate variable names;
  - positive scales, finite offsets, valid climatology settings;
  - `0 < sigma_min < sigma_max`, positive `sigma_data`;
  - probabilities in [0, 1] (`label_dropout`, coverage, `frame_probability`, `null_probability`), non-negative noise;
  - even sinusoidal channel counts;
  - grid and attention settings (as in §3.1);
  - time-sampling settings, optimizer ranges, `ema_decay` in [0, 1);
  - data source and `data_dir`;
  - conflicting start modes;
  - validation hours that overlap the training hours;
  - training or validation hour ranges too short for the longest window span. This check was added after a GPU verification run failed at data loading for exactly that reason.
- **`validate_sda`** checks sampler, solver and observation settings, and compatibility with the checkpoint: `conditional`/`cfg` need a conditional model, and `cfg` needs a model trained with `null_probability > 0`. `assimilate` makes the same CFG check for Python users.
- **Misspelled options:** `cli.load_config` rejects options the defaults file does not define and suggests the closest one ("unknown option num_step (did you mean num_steps?)").
- **List items:** overrides can now target list items, e.g. `variables.0.normalization.scale=0.05`. OmegaConf's merge cannot do this, so the loader applies overrides with `OmegaConf.update`.

### §4.3 Variable specifications

- **One `variables` list** replaces the parallel `outfields`, `channels`, `num_channels`, `out_transform_list`, `llc_standards` and `climatology_path`. Each entry has `name`, `field`, `source_variable`, `units`, `preprocessing` and `normalization: {scale, offset, climatology}`. The channel count and order come from it.
- **Affine normalization needs only config:** `_SCALE_KEY`, `_MEAN_KEY` and `_UNITS` are gone, and a new variable (with or without a climatology) needs no code. A test adds a `U` variable from configuration alone.
- **Preprocessing is recorded separately** (e.g. "scales longer than 300 km removed"), printed with the normalization, and saved in checkpoints.
- **Older checkpoints:** `core/config.py` translates the released checkpoints' flat format in one place (`variables`, `upgrade`). `load_checkpoint` returns the current format, and `export_checkpoint` writes it.
- **Not done:** grid spacing and projection metadata in the config.

### §4.4 Schema version (partly)

- A `config_version: 2` marker and the legacy translator came with §4.3. The released-checkpoint numerical comparison remains the regression gate, and passes.
- **Not done:** asset packaging.

### §5.1 Presets

- `train_tiny.yaml` became **`train_smoke_test.yaml`**, which selects white noise explicitly (`data_source: synthetic`).
- The base config now requires real data (`data_source: patches` with `data_dir`), so a long run on white noise cannot start by accident.
- **New `train_structured.yaml`:** a small model on the structured data, with label inputs off and validation on.
- The released presets are the base config (joint) and `train_ssh_only.yaml`. Label settings stay in the base config for compatibility, marked "(released)", with a note that new models without labels can use 0.
- **Not done:** a "student baseline" preset on real data, which needs a decision about the dataset.

### §5.2 One source of defaults

- `inference/sda/defaults.py` (`DEFAULTS`) supplies the defaults of `assimilate`, `run_inference` and `build_sampler`. A test checks that `configs/sda.yaml` holds the same values.
- This changed the Python defaults to match the command line: 16 → 8 members, 2 → 4 per batch, 256 → 64 steps, 1 → 4 MMPS iterations, CFG weight 1.0 → 1.5.
- The low-level `MMPSDenoiser(iterations=1)` keeps azula's minimal default; its docstring now says so.
- **`sigma_y` is now required** whenever something is observed, instead of defaulting to 0.01 for every variable regardless of units.

### §6.2 `per_frame_adaptive_scale`

- New config key (default `false`, as released), passed through to `SongUNet`. It is validated to need `use_per_frame_adagn`, and comparing runs treats it as an architecture setting.
- A test checks the doubled projection shape and a forward pass.
- **Not done:** the other controls the review lists (attention heads, residual conditioning style, boundary behaviour).

### §9.3 Evaluation beyond training loss

- **Held-out denoising loss** (`training/validation.py`): with `validation.every > 0`, the EMA weights denoise a fixed set of held-out windows at fixed noise levels with fixed noise. The weighted error per variable is printed and appended to `validation.csv`. The held-out windows come from separate hours for real patches, or from a separate generated stream otherwise. The config check enforces a gap of at least the maximum window span between training and validation hours (a temporal holdout with a buffer, part of §9.2).
- **Diagnostics** (`inference/metrics.py`):
  - errors at observed and unobserved pixels;
  - observation misfit in units of `sigma_y`;
  - spread against skill, with the finite-ensemble factor;
  - rank histogram and interval coverage;
  - spatial correlation between variables and isotropic coherence;
  - frame-to-frame correlation, alongside the existing spectra.
  - `summarize` prints the usual set, and `run_sda.py` now prints it instead of only RMSE and spread.
  - Tests check each against cases with known answers: a calibrated Gaussian ensemble, an under-dispersed one, and fields with a known correlation of 0.8.
- **`assimilate` output** now includes `observations` (the values the guidance used, NaN elsewhere, in physical units) and a `route` attribute.
- **Not done:** automated sensitivity sweeps over steps or iterations (documented as a procedure instead), and multi-case evaluation scripts.

### §9.4 Structured teaching dataset

- `data_source: structured` (`StructuredDataset`, `structured_fields`): Gaussian random fields with a power-law spectrum, all drifting together at a set speed in a random direction (an exact periodic shift), with every later variable correlated with the first at a chosen correlation at every scale.
- It is reproducible per sample index, with a separate validation stream.
- Tests check unit variance, the 0.8 correlation and the exact drift.
- The GPU check in §2 trains the preset and tests whether a learned prior reproduces these statistics.

### §10 and §11 Mathematics and wording

- **§10.1 Notation:** a notation table in `docs/README.md` covering `u`, `x`, `τ_i`, `σ`, `z_σ`, `F_θ`, `D_θ`, `c`, `y = h(x) + e`, `R = diag(σ_y,j²)` and `J_θ`, plus the shape and flattening conventions.
- **§10.2 Denoising objective:**
  - page 1 now separates "the squared-error optimum is the conditional mean" from "Tweedie's identity links it to the score";
  - the preconditioning formula shows which inputs are scaled by `c_in` and which context enters separately, as implemented;
  - the literature page is corrected to match.
- **§10.3 Approximation layers:** page 4 and `mmps.py` list the four layers (learned denoiser, moment matching, truncated solve, finite steps) and say what the Gaussian tests do and do not establish.
- **§10.4 Autodiff:** comments beside each Jacobian product in `mmps.py` and `rozet.py` state direction and shapes (`autograd.grad(out, inp, v) = Jᵀ v`, not `J v`).
- **§10.5 Rozet heuristic:** `rozet.py` and page 4 explain that the variance is added in observation space and equals `γ σ_t² A Aᵀ` only for unit-weight selection operators.
- **§8.3 Observation-space scaling:** documented on page 4.
- **§11 table:** every row was applied:
  - noise level and wavelength;
  - large-scale-first as an intuition;
  - log-normal under-training as a hypothesis;
  - what EDM weighting does;
  - MMPS "exact, nothing tuned";
  - linear observations not making MMPS exact;
  - `γ = 0.01` wording;
  - ensemble spread;
  - misfit about `σ_y`;
  - "every public function takes physical units";
  - fine-tuning marked supported.
- **"Additional cleanup" list, all done:**
  - the stale `checkpoint` module reference in `denoiser.py`;
  - references to unavailable parent-project markdown files in `mmps.py` (replaced by short explanations) and to a parent-project file in `rozet.py` (replaced by the public `sda` repository link);
  - "real" observation mode renamed `instrument`, with comments and a printed note that values come from the simulated truth;
  - nominal maximum span versus actual span, made consistent;
  - parent-project numbers labelled historical and not reproduced;
  - M_3/M_4 kept in the comparison discussion only.
- **Other pages:** §6.3 (temporal routes as an open question), §6.4 (boundaries, pixel count versus resolution), §7.2 (hybrid route: documented, printed as unvalidated, recorded in the output), §7.3 (noise-level inputs), and the `label_dropout` versus `null_probability` distinction (§7.4).

## 4. Changes users will notice (breaking)

Nothing had been run with the repository, so these were made directly rather than kept compatible.

- **Training config format:** a `variables` list instead of the flat channel and normalization keys; `data_source` and `start` added; per-variable conditioning maps moved from the base config into `train_conditional.yaml`.
- **Presets:** `configs/train_tiny.yaml` is now `configs/train_smoke_test.yaml`.
- **No silent resume:** rerunning into an existing run directory now needs `start=resume` or `start=auto`.
- **`sda.yaml`:** `observations.mode: real` is now `instrument`. The `normalization` section takes `climatology: {...}` and `variables` instead of `climatology_path`, `out_transform_list` and `llc_standards`.
- **Python API:**
  - `assimilate` requires `sigma_y` when observing, and its sampler defaults now equal `sda.yaml`.
  - `Normalizer(...)` takes a list of variable specs. Output attributes `normalization_transforms` and `normalization_standards` are replaced by `normalization_variables` (JSON).
  - `channel_names` lives in `core/config.py` (still importable from `core.model` and the top level).
- **Released checkpoints load unchanged.** Their outputs are the same; see the regression test and the GPU comparison in §2.

## 5. Held off for later

| Item | Why held |
|---|---|
| §4.1 Nested config groups (`data`, `model`, `diffusion`, ...) | The largest complexity cost for the least immediate benefit: validation and the variables list deliver most of it. Worth revisiting with a schema translator if the config keeps growing. |
| §4.4 Asset packaging and a fuller schema translator | Not needed while the repository is used as an editable install. |
| §5.1 Student baseline preset on real data | Needs a decision about which real dataset students start from. |
| §5.3 Configurable precision | Small, but not requested. Docs now explain why SDA runs in float32. |
| §3.1 Rectangular patches; §6.2 attention by stage index, heads, residual style, boundaries; §6.5 splitting `networks.py` | Architecture work that changes state-dict keys or parameter shapes. The review itself advises doing it after compatibility checks, as separate experiments. |
| §7.1 Named conditioning concepts; §7.3 noise-level inputs; §7.4 separating mask, operator and noise generation | A redesign of the conditioning interface; best done together with a concrete conditional-model experiment. |
| §8.1–§8.2 General observation interface and operators (coarse-to-fine, tracks) | Done later as an opt-in module; see §8. |
| §9.1 Dataset adapter abstraction, irregular source times | Added the cheap safeguards (finite values, `source_variable`, time alignment across variables); a second adapter should wait for a second dataset. |
| §9.2 Spatial holdouts | Temporal holdout with a buffer is supported for validation; spatial splits need a decision about tile layout. |
| §9.5 Saving RNG states, code revision and dependency versions | Not requested; the climatology checksum and EMA horizon report were added. |
| §13 Tutorials; §9.3 automated sensitivity sweeps and multi-case evaluation | Need scientific choices (cases, metrics, compute budget). The structured dataset, presets and diagnostics are the building blocks. |
| A masked-target loss (§3.2, future part) | Needs a design for domain masks versus missing observations; the current policy is explicit and enforced. |

## 6. Where to look

| Topic | Files |
|---|---|
| Variables, validation, legacy translation, run comparison | `src/obottle_starter/core/config.py` (new) |
| Climatology rules, normalization from variables | `src/obottle_starter/core/normalization.py` |
| Start modes, DDP fixes, validation passes | `src/obottle_starter/training/train.py`, `training/validation.py` (new) |
| Structured data, missing-value checks | `src/obottle_starter/training/data.py` |
| Diagnostics | `src/obottle_starter/inference/metrics.py` |
| Shared defaults | `src/obottle_starter/inference/sda/defaults.py` (new) |
| Configs | `configs/train_base.yaml`, `train_smoke_test.yaml`, `train_structured.yaml`, `sda.yaml` |
| New tests | `tests/test_config.py`, `test_metrics.py`, `test_distributed.py`, plus additions to the existing files |
| GPU verification scripts and logs | `/scratch/tm3076/obottle-starter-verify/refactor/` |

## 7. Independent follow-up evaluation

This section records the subsequent review of the response above and its corresponding implementation. It preserves the original response as a historical record. The recommendations below were not implemented as part of the follow-up review.

### 7.1 Overall assessment

The changes are a substantial improvement, and the scope decisions are sensible. Keep the current design and make a small corrective pass before handoff. Broader configuration swapping can remain deferred.

The strongest changes are:

- The consolidated variable specifications.
- An explicit policy for missing training targets.
- Climatology rules that preserve the released model's supported behavior while making other choices explicit.
- Shared inference defaults.
- Clearer distinctions between mathematical identities and approximate inference.
- Reproducing the original distributed-training failures before fixing them.
- Structured teaching data and validation at fixed noise levels with fixed randomness.

These changes make the repository substantially easier to understand and experiment with without requiring a larger framework.

### 7.2 Independent verification

The full test suite was rerun in the existing scientific Python container, with the released checkpoints enabled:

```text
92 passed, 18 warnings in 216.65s
```

The warnings were PyTorch deprecation warnings concerning `torch.jit.script`. The passing suite includes the released-checkpoint comparisons and CPU/Gloo distributed-training test.

The saved GPU logs were inspected and support the reported bitwise-identical SDA outputs and successful distributed resume. Those GPU jobs were not rerun during this follow-up review.

Additional targeted checks reproduced the observation, dataset, resume, and fractional-offset cases described below. The resume checks exercised the checkpoint-selection helper without overwriting an actual run. No implementation files were changed.

### 7.3 High priority: an explicit missing resume checkpoint can silently start a new run

Relevant code: [`training/train.py`](src/obottle_starter/training/train.py), `_resume_source` and the fresh-run metadata/checkpoint paths.

With `start=auto`, a nonexistent explicit `resume_from` returns `None`, even if the output directory contains `checkpoint_latest.pt`. Training then follows the fresh-run path and can overwrite the existing metadata and eventually the latest checkpoint.

This decision path was reproduced with a temporary directory containing an existing latest-checkpoint marker and an explicit nonexistent checkpoint path.

The fresh-run protection is also narrower than the response implies: `start=fresh` checks only for `checkpoint_latest.pt`. An existing `config.yaml` or numbered checkpoint alone does not protect the directory. A temporary directory containing an existing `config.yaml` but no latest checkpoint was accepted by the helper as a fresh run.

Recommended changes:

- Require an explicitly supplied `resume_from` to exist, regardless of `start=auto`.
- Permit automatic fresh-run fallback only when no explicit checkpoint was requested.
- Treat existing run metadata and numbered checkpoints as evidence of an existing run, not only the latest-checkpoint alias.
- Keep failures before any metadata or checkpoint writes.

Add regression cases for a misspelled explicit path with an existing latest checkpoint, an output directory containing only run metadata, and one containing only numbered checkpoints.

### 7.4 High priority: nonfinite observed values are still accepted by unconditional SDA

Relevant code: [`inference/assimilate.py`](src/obottle_starter/inference/assimilate.py) and [`inference/sda/observations.py`](src/obottle_starter/inference/sda/observations.py), `build_observation`.

The `build_obs_condition` fix is correct, but unconditional MMPS/Rozet does not call that helper. The assembled observation vector is not checked for finite values before sampling.

A targeted check passed one observed NaN through `assimilate` with an unconditional model. It returned nonfinite samples instead of rejecting the input. Thus the statement that observed NaNs are rejected currently applies to the conditional-input helper, not to every assimilation route.

The Python wrapper's positivity check also accepts `sigma_y=inf`; positivity alone is insufficient.

Recommended changes:

- Validate the assembled observation vector centrally before either conditional or unconditional sampling.
- Require assumed observation-error standard deviations to be finite and positive.
- Require synthetic-noise standard deviations to be finite and nonnegative.
- Continue allowing arbitrary nonfinite values outside observation masks, since those entries are not observations.

Test both unconditional and conditional routes, including observed NaN/inf, unobserved NaN/inf, and invalid error/noise levels. These checks can be centralized without expanding the public interface.

### 7.5 Medium priority: generated training data are duplicated across distributed ranks

Relevant code: [`training/data.py`](src/obottle_starter/training/data.py), `StructuredDataset`, and the DataLoader construction in [`training/train.py`](src/obottle_starter/training/train.py).

`StructuredDataset` deterministically generates each sample from `(cfg.seed, split, idx)`. The training DataLoader has no distributed sampler, so every rank visits the same indices in the same order. Changing Torch's per-rank seed does not change those fields.

A targeted check constructed the first batch under two different rank seeds and confirmed that the clean targets were identical. Diffusion noise differs, so gradients are not identical, but adding GPUs does not increase the number of distinct clean training examples per step as intended. The index-deterministic white-noise dataset has the same sampling concern.

Recommended changes:

- Use a distributed sampler for index-deterministic training datasets.
- Advance the sampler's epoch when restarting the DataLoader iterator.
- Keep validation on its fixed, separate stream.
- Add a test that ranks receive distinct training indices, in addition to the existing synchronization and execution checks.

The current CPU/GPU DDP tests establish that training executes and replicas remain consistent. They do not establish that data are partitioned correctly.

### 7.6 Medium priority: fractional time spacing passes validation but is not supported by sampling

Relevant code: [`core/config.py`](src/obottle_starter/core/config.py), `validate_training`, and [`training/data.py`](src/obottle_starter/training/data.py), `draw_offsets`.

The validator accepts any positive numeric `nonuniform_minimum_offset_hours`, but `draw_offsets` converts that value to an integer.

A targeted check verified that `nonuniform_minimum_offset_hours=0.5` passes validation and then raises `ZeroDivisionError` because the integer step is zero. A value such as `1.5` is silently truncated to one hour.

For the current hourly-index implementation, require integer offsets and spans. Fractional timestamp support can remain deferred. Add validation tests for fractional values below and above one hour so unsupported values cannot silently change meaning.

### 7.7 Medium priority: the structured benchmark compares different estimators

Relevant material: the structured-data row in §2 above and `/scratch/tm3076/obottle-starter-verify/refactor/check_structured.py`.

The reported posterior RMSE of approximately 0.73 is computed from the ensemble mean. The prior reference of approximately 1.41 comes from the expected difference between two independent unit-variance fields, which describes a single prior draw rather than a prior ensemble mean.

For an ideal zero-mean, unit-variance prior and an independent truth from that prior:

| Prediction | Expected squared error | Corresponding RMS error |
|---|---|---|
| A single prior draw | `2` | `sqrt(2)`, approximately 1.41 |
| The exact prior mean | `1` | 1 |
| The mean of 16 independent prior draws | `1 + 1/16` | approximately 1.03 |

The result still suggests useful transfer of information from A to B, but the fair reference is roughly 1–1.03, rather than 1.41. These are idealized expectations; the preferred correction is to evaluate a prior ensemble of the same size on the same eight cases, using the same error aggregation as the posterior experiment.

Also soften the spectral statement in §2: the logged ratios include 0.82 and 0.83 at high wavenumbers, so “within about 15% except at the two largest scales” is too strong. Report the actual range or describe the departures explicitly.

### 7.8 Smaller explanation and diagnostic refinements

**Temporal validation buffer.** A buffer between training and validation periods is a reasonable conservative choice against temporal correlation. However, the current loader keeps sampled frames within each configured range, so nonoverlapping ranges already prevent shared frames. The validation error should distinguish overlap from an insufficient decorrelation buffer rather than claiming that every smaller buffer necessarily causes shared frames.

**Replica comparison.** `_check_replicas` compares sums of parameters, not full state checksums. This catches the reproduced initialization problem, but equal sums do not guarantee equal tensors. Describe it as a lightweight consistency check, or use an actual state comparison/checksum if equality is the claimed guarantee. The distinction does not invalidate the DDP fixes themselves.

### 7.9 Configuration flexibility: defer a larger abstraction

Do not introduce registries or interchangeable component schemas solely to anticipate future experiments. The existing `build_model`, dataset factory, variable specifications, and sampler factory already provide useful extension points.

A second concrete architecture, dataset adapter, or conditioning experiment will reveal which interfaces need generalization. For now, fixing the specific cases above offers more value than expanding the configuration hierarchy.

Suggested next step: resolve the two high-priority input/run-safety issues, fix distributed sampling and integer-offset validation, and correct the benchmark comparison. Then retain the current structure for the student handoff.

## 8. Follow-up: general observation interface (§8.1–§8.3)

Added after the review response, at the user's request, as an **opt-in module on main**. Nothing that uses masks changed: `assimilate()`, `ObservationTerm`, the configs, scripts and notebooks 01–02 behave as before.

### What was added

- **`inference/sda/operators.py`:**
  - `SparseObservation`: each observation is a weighted sum of state values plus a constant, stored as a sparse `(N, K)` index and weight. One class covers:
    - off-grid points (`point_observations`: bilinear in space, linear between frames, for fractional observation times);
    - coarse cell averages (`block_average`);
    - footprint kernels (`footprint_observations`, `gaussian_kernel`);
    - combinations of variables and time averages, by adding and scaling rows: `0.5 * (obs_t0 + obs_t1)`.
  - `ObservationSet` concatenates several observation types.
  - `ObservationCovariance` is a block-diagonal `R`: vector blocks for independent errors, dense blocks for correlated ones (`exponential_covariance` for along-track errors). Dense blocks are factorized, so each should stay at a few thousand observations or fewer.
- **`inference/general.py`:**
  - `model_unit_observation` rewrites a physical-unit observation equation consistently: `A_model = Q(H S x + H b + offset)`, `y_model = Q y_phys`, `R_model = Q R Qᵀ`. This addresses §8.3. The default `scaling="error"` (`Q = 1/σ_y` per observation) puts every observation in units of its own error. `scaling="row_norm"` (unit-size rows in model units, i.e. what `assimilate` does for masks) and explicit per-part `Q` are available.
  - **Why that default:** the first notebook run used `row_norm`, and in the joint SST-footprint plus SSH-track example the 4-iteration solve left the SST footprints 4–5 error stds from their observations. A GPU comparison (`/scratch/tm3076/obottle-starter-verify/refactor/nb03/fp_*.out`) showed SST footprints alone fit fine at 4 iterations (0.98). In the joint problem, the diagonal preconditioner reached 1.30, 16 iterations 1.49, and error scaling 1.09 at no extra cost. The operator itself was checked separately: it reproduces the observations from the truth to 1e-6.
  - `assimilate_general` takes physical units in and out and returns samples plus observation-space diagnostics: `y`, each member's `predicted`, `error_std`, `scaling`, operator coordinates, and a whitened `misfit` per member.
- **Guidance:** MMPS and Rozet accept an `ObservationCovariance`. MMPS multiplies by `R`; Rozet solves with `R + γσ_t² I` block by block and whitens its diagnostics with `R`. A plain variance tensor is wrapped and computes bit for bit as before (tested). `run_inference` reads the state shape from general operators.
- **Input checks:** observations must be finite, variances finite and positive, and dense blocks symmetric positive definite. These apply to the general path; follow-up §7.4 asks for the same on the mask path, which is not yet done.
- **Notebook `examples/03_general_observations.ipynb`** (released joint model, GPU):
  1. coarse-to-fine SSH from 64 km averages, against cubic interpolation of the coarse field;
  2. along-track SSH points at fractional times with exponentially correlated errors, against wrongly assuming independent errors;
  3. SST footprints averaged over two frames, assimilated jointly with the SSH tracks.
- **Docs:** README ("General observations"), page 4 ("Observations that are not pixels"), page 5 (extensions table).

### Verification

- **`tests/test_operators.py`** (15 tests):
  - operators reproduce linear fields exactly; block averages equal manual means; footprints average constants to themselves at edges; sums and scaling give time averages;
  - covariance algebra (multiply, solve, whiten, shift, scale, and the empirical covariance of samples) matches dense matrices;
  - **MMPS with points, block averages and a correlated block reproduces the exact Gaussian posterior** for CG, GMRES and MINRES, with and without the diagonal preconditioner;
  - Rozet with correlated errors matches its closed form;
  - plain variance tensors are bitwise unchanged;
  - physical-to-model scaling is consistent;
  - an end-to-end `assimilate_general` run, and the input validation.
- Full suite: **107 passed** (92 earlier tests plus 15 new), including the released-checkpoint regression test.
- Notebook executed on one L40S (job 20783165, 25 minutes; executed copy in `/scratch/tm3076/obottle-starter-verify/refactor/nb03/`):
  - **Coarse-to-fine:** SSH RMSE 0.0166 m for the SDA ensemble mean, against 0.0291 m for cubic interpolation of the 64 km averages (truth std 0.073 m). Members match the averages (whitened misfit 0.90) and carry fine-scale power that interpolation lacks.
  - **Tracks:** the correlated and the independent `R` give nearly the same result here (SSH RMSE 0.0569 vs 0.0564 m, error/spread 1.32 vs 1.29). The notebook says so and does not claim a benefit from one window.
  - **Joint footprints and tracks:** SST footprints fit at 1.07 error stds and tracks at 1.11 and 1.14, with SST RMSE 0.32 °C. With `row_norm` the footprints are left at 4.35 and SST RMSE is 0.75 °C; the notebook shows both.

### Not included

- Nonlinear operators: MMPS would linearize them at each step, which is an approximation.
- General observations with conditional models: their observation inputs are mask-shaped.
- Correlation between different observation types (blocks are per type), and non-block-diagonal `R`.


## 9. Fixes for the follow-up evaluation (§7)

All recommendations in §7.3–§7.8 were implemented; §7.9 (no new abstractions) was followed.

| § | Finding | Fix | Test |
|---|---|---|---|
| 7.3 | A misspelled `resume_from` with `start=auto` silently started a fresh run; `fresh` only checked for `checkpoint_latest.pt`. | An explicit `resume_from` must exist in every mode. A directory holds a run if it has `config.yaml` or *any* checkpoint, and `fresh` refuses it. `auto` resumes from the latest checkpoint; with numbered checkpoints but no latest it stops and asks for `resume_from`; with only `config.yaml` (an attempt that died before its first checkpoint) it starts over. `config.yaml` is now written after the dataset and loader are built, so a data error leaves no metadata. | `test_resume_source_needs_evidence_and_existing_paths` (misspelled path with an existing latest; metadata only; numbered checkpoints only) |
| 7.4 | Observed NaN or inf reached unconditional MMPS/Rozet; `sigma_y=inf` was accepted. | Checked centrally in `build_observation`, which every mask route uses: observed values finite, `sigma_y` finite and positive, synthetic-noise std finite and non-negative. `assimilate` also requires finite positive `sigma_y`. NaN outside masks stays allowed. (The general interface already had these checks.) | `test_nonfinite_observations_are_rejected`, run for an unconditional and a conditional model: unobserved NaN accepted; observed NaN in `y`, observed inf in the truth, infinite `sigma_y` and NaN noise rejected |
| 7.5 | Generated datasets gave every DDP rank the same clean samples. | `training_sampler` returns a `DistributedSampler` for `structured` and `synthetic` data, with the epoch advanced on each pass. `patches` keeps its per-rank random stream; validation keeps its fixed stream. | `test_generated_data_is_split_across_processes` (disjoint indices across two ranks, reshuffled per epoch); the two-process DDP test still passes |
| 7.6 | Fractional `nonuniform_minimum_offset_hours` passed validation, then crashed (0.5) or was truncated (1.5). | The minimum offset and maximum span must be whole hours (whole-valued floats such as `336.0` are accepted). | `test_offsets_must_be_whole_hours` (0.5, 1.5) |
| 7.7 | The structured benchmark compared a posterior ensemble mean against a single prior draw, and the spectral statement was too strong. | Rerun with a 16-member prior ensemble on the same eight cases and the same aggregation (job 20785441; `/scratch/tm3076/obottle-starter-verify/refactor/check_structured_fair.py`). Ensemble-mean RMSE for B (never observed): **posterior 0.726, prior 0.999**; for A at unobserved cells: posterior 0.589, prior 0.989. The prior matches the idealized reference of about 1–1.03, so observing A on 20% of cells reduced B's error by about 27%, not the 49% implied in §2. Spectral ratio (samples / truth) ranged from **0.66 to 1.06**: 0.66–0.85 at the three largest scales, 0.82–0.85 at the three smallest, and within 6% in between. Correlation between A and B in prior samples: 0.804 (data: 0.8). | GPU log `structured_fair_20785441.out` |
| 7.8 | The validation-gap message claimed any small gap shares frames; the replica check compared parameter sums. | Overlapping hours ("would share frames") and too small a gap ("correlated, makes validation optimistic") now have separate messages; the buffer requirement stays as a conservative choice. `_check_replicas` broadcasts rank 0's tensors and compares them exactly. | Updated `test_validation_hours_must_not_overlap_training`; DDP test |

Full suite after these fixes: **113 passed** (CPU, released checkpoints enabled). Sections 2–3 above are left as written, as the historical record §7 refers to; this table supersedes the structured-benchmark row in §2.
