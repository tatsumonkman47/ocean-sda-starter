# 4. Reconstructing fields with score-based data assimilation (SDA)

Once a model has learned a distribution of ocean fields, we can use
observations to steer the fields it generates. The practical question is
how strongly each observation should influence a prediction that is itself
uncertain. This page develops that update, compares the available
approximations, and explains what to check in the resulting ensemble.

For runnable examples, see [notebook 01](../examples/01_prior_and_sda.ipynb)
and the [README commands](../README.md#reconstruct-fields-from-observations).

## Observation model and posterior distribution

We have sparse, noisy observations `y` (SWOT swaths, nadir tracks, SST with
cloud gaps) of an ocean state `x`. Write the observations as

```
y = A x + ε,     ε ~ N(0, Σ_y)
```

Here `A` maps a model field to what an instrument would observe. A mask
selects grid cells; a more general operator can average cells or interpolate
to observation locations. For independent errors,
`Σ_y = R = diag(σ_y,j²)`, with one variance per observation; equal variances
give `σ_y² I`. `assimilate` accepts one `σ_y` per variable.

Our target is the **posterior distribution**,
`p(x | y) ∝ p(y | x) p(x)`. The diffusion model supplies the prior `p(x)`;
the observation model supplies the likelihood `p(y | x)`. Sampling the
posterior gives an ensemble of plausible reconstructions rather than only
one best estimate.

**Units must agree.** The public function takes physical units, then
transforms the state, observations, and error covariance consistently into
model units. The same prior can be used with different observation patterns,
error levels, and linear operators without retraining.

## How observations modify the denoising steps

Sampling from the prior repeatedly evaluates `x̂ = D(x_t, σ_t)`,
the expected clean state given the current noisy one, and takes a small
step towards it (the DDIM sampler, `src/obottle_starter/inference/sda/sampling.py`).
Guidance replaces `x̂` with an estimate of the clean state given the noisy
state **and the observations**, `E[x | x_t, y]`. The methods differ in how
they approximate this.

<p align="center">
  <img src="../examples/figures/c_sda_loop.png" alt="SDA cycle: denoise, map the estimate to observation locations, calculate the residual, apply guidance, and take the next sampling step">
  <br>
  <em>Observation guidance is recomputed at every sampling step. The diagram uses H for the observation operator called A in this page, and tau for diffusion time.</em>
</p>

At every step, the guided estimate has the form

```
x̂_guided = x̂ + γ_t Jᵀ Aᵀ g,      where  ( Σ_y + A Cov[x | x_t] Aᵀ ) g = y − A x̂
```

Here `J = ∂x̂/∂x_t` is the denoiser's Jacobian and `γ_t = σ_t²`.
Automatic differentiation evaluates `Jᵀ v` in one backward pass,
`autograd.grad(x̂, x_t, v)`, without constructing `J`.

The update first computes the observation residual `y − A x̂`, then
weights it by the combined observation and prediction uncertainty. The
Jacobian propagates this correction across the field according to the
spatial, temporal, and cross-variable relationships learned by the model.
This allows an SSH observation to constrain nearby SSH and, in the joint
model, SST.

There are two sources of uncertainty in this calculation: observation
error (`Σ_y`) and uncertainty in the clean field given the current noisy
state (`Cov[x | x_t]`). Early in sampling, the latter can be large. Treating
the denoised estimate as exact can therefore make an observation correction
too strong. The methods below differ in how they represent this uncertainty.

### Watching observations guide a reconstruction

The animation follows a reconstruction constrained by two crossing
SWOT-like swaths. Compare it with [sampling without observations](01_network_architecture.md#what-a-diffusion-model-learns):
the prediction now responds to observed values, while the prior supplies
structure in the gaps.

<p align="center">
  <img src="../examples/figures/c_sda.gif" alt="Observation-guided denoising with two crossing swaths, showing the noisy state and estimated clean field">
  <br>
  <em>Illustration using a Gaussian random field with an SSH-like spectrum and a closed-form denoiser. This shows the sampling process, not a benchmark of the released ocean model.</em>
</p>

<details>
<summary>View still frames</summary>

<p align="center">
  <img src="../examples/figures/c_sda_strip.png" alt="Six stages of guided sampling alongside the observations and hidden truth">
  <br>
  <em>Selected steps from the animation, arranged from high to low noise.</em>
</p>

</details>

Both denoising illustrations can be regenerated with
[`examples/figures/make_sketches.py`](../examples/figures/make_sketches.py).

## Three ways to handle the uncertainty

The figure below illustrates how these choices can change a sampling
trajectory in a one-dimensional example. Read from right to left: all
curves start at the same noisy state, while the observation is marked by
the horizontal dotted line.

<p align="center">
  <img src="figures/mmps_vs_SDA_vs_DPS.png" alt="One-dimensional sampling trajectories under unconditional generation, DPS, SDA heuristic guidance, and MMPS">
  <br>
  <em>Different uncertainty approximations produce different trajectories. This is an illustration for one observation and parameter setting, not a performance benchmark. The SDA curve uses gamma = 1 rather than this repository's default of 0.1. MMPS avoids that variance parameter but still requires observation-error and numerical settings.</em>
</p>

### Rozet & Louppe (2023): prescribed uncertainty

Rozet guidance adds a prescribed variance to the observation error:
`σ_y² + γ σ_t²`, where `γ` is a tuning parameter. With independent errors,
the linear system reduces to an elementwise division.

**Where the variance is added matters.** This implementation adds
`γ σ_t² I` directly in observation space. Assigning that covariance to the
state and then applying `A` would instead give `γ σ_t² A Aᵀ`. The two
expressions agree for masks that select individual model-unit values with
unit weight. They generally differ for averages, smoothing, or weighted
combinations of variables.

- **Cost:** one backward pass per step; no iterative linear solve.
- **Limitation:** the variance does not reflect spatial differences in
  prediction uncertainty, and `γ` requires tuning. Large values weaken the
  influence of observations early in sampling; small values can make updates
  unstable.
- **Previous results (parent project; not reproduced here):** tested only
  inside the multi-window SDA of the parent work (May 2026). With the settings
  tried (γ ≥ 1, 6–100 iterations), guidance either left streak artefacts or
  produced almost no adjustment towards the observations. The analysis
  suggested `γ ≪ 1` or a schedule might work; that was not pursued. There is
  no single-window comparison against MMPS on these models.
- **In this repository:** `guidance.method: rozet` with `rozet.gamma` in
  `configs/sda.yaml`, or `assimilate(..., guidance="rozet", rozet_gamma=...)`.
  The code is `inference/sda/rozet.py`, following `GaussianScore` in Rozet's
  original `sda` code.

**Choosing γ.** The heuristic stands in for MMPS's `A Cov[x | x_t] Aᵀ`.
For unit-weight observations in model units, that true term is about
`σ_t²` per observation at small σ_t (the denoiser's Jacobian is close to
the identity) and about `σ_data² = 1` at large σ_t. So:

| γ | Behaviour |
|---|---|
| 1 | Matches MMPS late in sampling; much weaker guidance early (at σ_t = 100 the assumed variance is 10⁴, against a true value near 1) |
| 0.1 (this repository's default) | 10× stronger guidance than γ = 1 at every step |
| 0.01 (the default in Rozet & Louppe's code) | 100× stronger than γ = 1. That says nothing about the model's true uncertainty; it only sets how hard each step is pulled |
| → 0 | Guidance as strong as `σ_y` allows; risks overshooting and instability early, when `x̂` is still poor |

The added variance `γ σ_t²` is defined in model units, matching the
observations constructed by `assimilate`. Tune it by checking the misfit at observed
pixels (`record_diagnostics`) and the error at unobserved pixels.

**Why the default is 0.1.** At high noise the guidance step is about
`σ_t² / (γ σ_t²) = 1/γ` times the observation residual, so small `γ` makes
large early steps. Rozet & Louppe's code uses `γ = 0.01`, but together with
Langevin corrector steps (below), which this repository's DDIM sampler does
not have. Without them, `γ = 0.01` diverged for the released joint model.
On one MidAtlantic window (5 frames, 64 steps, 4 members), the
ensemble-mean errors were:

| Guidance | SSH RMSE | SST RMSE |
|---|---|---|
| Rozet, γ = 0.01 | 0.16 m | about 10⁹ °C (diverged) |
| Rozet, γ = 0.1 | 0.070 m | 0.115 °C |
| Rozet, γ = 1 | 0.078 m | 3.1 °C |
| MMPS (default guidance) | 0.045 m | 0.110 °C |

So 0.1 is a working value for these models and this sampler, from one
window, not a tuned optimum. Treat `γ` as a tuning parameter, and expect
the best value to change with the model, the observations, the number of
`steps` and whether corrector steps are used.

### Langevin corrector steps (not implemented here)

A sampler step (the *predictor*) moves the state from one noise level to
the next using the current estimate of the score. Every approximation in
that estimate, from the discretization, the likelihood heuristic (`γ`) or
the sliding windows of page 8, leaves the state slightly off the
distribution it should follow at the new noise level. The next step starts
from there, so errors can build up over the run.

**Langevin dynamics** is a way to pull a state back onto a distribution
whose score you know. At a fixed noise level, repeat

```
x ← x + δ ∇log p(x) + √(2δ) z,      z ~ N(0, I)
```

The score term moves `x` towards higher probability; the noise keeps it
from collapsing onto the most likely state. With a small enough step `δ`
and enough iterations, `x` ends up distributed according to `p`, wherever
it started. Here the score is the *guided* one, `(x̂_guided − x) / σ_t²`,
so the target is the posterior at that noise level.

A **corrector** is a few such iterations after each predictor step (the
predictor-corrector samplers of Song et al., 2021). They give the sampler
a chance to undo the errors of the step it just took instead of carrying
them forward. Rozet & Louppe (2023) use them with their likelihood
heuristic and report that a few are enough; their code chooses `δ`
adaptively from the size of the score, with a tunable amplitude `τ`. That
is a likely reason their `γ = 0.01` works there and diverges here: strong
guidance overshoots, and the correctors let the state relax back.

The cost is one more guided denoiser evaluation (a network forward and
backward pass) per corrector iteration, so one corrector per step roughly
doubles the cost of a run. Whether that buys more than spending the same
budget on more `steps`, or on MMPS, is an open question for these models.

**An experiment to try.** Add a `corrections` option (and `τ`) to
`GuidedDDIMSampler` in `inference/sda/sampling.py`: after each DDIM update,
evaluate the guided denoiser at the new noise level, form the score, and
apply that many Langevin updates. Then vary the number of corrector steps
(0, 1, 2, 5) and `τ`, and check whether smaller `γ` becomes stable, how the
error and spread change, and what it costs, for single windows and for the
sliding windows of page 8. The MMPS Gaussian test (`tests/test_mmps.py`)
is a good template for checking that, with an exact score, the corrector
leaves the posterior unchanged up to the effect of its finite step size.

### MMPS: uncertainty estimated from the denoiser (default)

Moment Matching Posterior Sampling (Rozet et al., 2024) uses
`Cov[x | x_t] ≈ σ_t² J`. For the exact conditional-mean denoiser, this follows from
`Cov[x | z_σ] = σ² ∂E[x | z_σ]/∂z_σ`. It requires no prescribed `γ` and
allows uncertainty to vary with the state.

**Why a linear solver is needed.** The covariance couples observations:
a change inferred from one observation can affect the prediction at another.
MMPS solves for these corrections together.

The Jacobian can have millions of rows, so it is never constructed
explicitly. Automatic differentiation supplies products `Jᵀ v`, and a
Krylov method solves the linear system iteratively. The code uses `Jᵀ v`
in place of `J v` because the exact conditional-mean Jacobian is symmetric;
a learned approximation need not be.

#### What MMPS approximates

**Sources of approximation.** MMPS differs from exact posterior sampling
in four respects, even with a linear observation operator:

1. **The learned denoiser** only approximates the conditional mean, so
   `σ² J` only approximates the covariance; the learned `J` need not be
   symmetric or positive semidefinite.
2. **Moment matching**: MMPS treats `p(x | x_t)` as Gaussian with that mean
   and covariance. For a non-Gaussian prior that is an approximation even
   with linear observations.
3. **The truncated linear solve**: a few Krylov iterations, not
   convergence.
4. **The finite-step sampler**: a discretized reverse process.

The Gaussian tests (`tests/test_mmps.py`) check the implementation in the
one case where (1) and (2) are exact; they do not show that posterior
samples from a learned ocean prior are calibrated. Observation errors, sampling
steps, solver iterations, and solver choice still require evaluation.

### DPS (Chung et al., 2023), for comparison

Diffusion Posterior Sampling drops the uncertainty term altogether and steps
along the gradient of `‖y − A x̂‖²` with a manually chosen step size. This
avoids estimating the denoiser's covariance but requires step-size tuning. In
the parent project (historical; not reproduced here) a naive form diverged
(produced NaN fields) for the small `σ_y` used here. It is not implemented in
this repository.

## Solving the MMPS linear system

A Krylov solver builds an approximate solution using repeated products of a
matrix with vectors. This is useful here because automatic differentiation
can compute those products without storing the full covariance matrix.

The MMPS system `M g = b`, with `M = Σ_y + A Cov[x | x_t] Aᵀ`, is solved
with a fixed number of iterations per sampling step (`mmps_iterations`).
Each iteration needs products with the network's Jacobian (a backward pass
through the network). Measured on an L40S for the joint model, a guided step
with 4 iterations took 2.2 s against 0.17 s for an unconditional step,
about 13× as long.

| Solver | Assumes | Minimizes | Notes |
|---|---|---|---|
| **GMRES** (default) | No symmetry or definiteness requirement | the residual ‖b − M g‖ over the current Krylov subspace | Supports nonsymmetric systems. Memory grows with the iteration count. |
| **CG** | `M` symmetric positive definite | the error in the `M`-norm | Cheapest per iteration and the classic choice when its assumption holds. |
| **MINRES** | `M` symmetric | the residual | Like GMRES but with short recurrences; tolerates indefinite `M`. |

For the exact denoiser, `Cov[x | x_t]` is positive semidefinite, so `M` is
symmetric positive definite when observation-error variances are positive. But
the network's Jacobian is only approximately symmetric (and the code uses `Jᵀ
v` where the formula has `J v`), so CG and MINRES rest on an approximation.
That is why GMRES is the default; it does not require symmetry or positive
definiteness, but solver choice alone does not ensure accurate inference.
Whether CG or MINRES do as well at 2–8 iterations on these models has **not
been tested**; the tests only confirm that all three match the exact answer
for a Gaussian prior when run to convergence (`tests/test_mmps.py`). To
compare CG with the default, set `mmps_solver="cg"`.

### Why units and scaling matter for a truncated solve

With only a few iterations, the linear solve may be far from converged.
Residual minimization is sensitive to the relative scaling of observations.
Observation types with larger numerical residuals can therefore dominate the
solve. This happened in this project twice:

- In the starter's development, with SSH in metres and SST in °C, 4 GMRES
  iterations fitted SST and essentially ignored SSH: ensemble-mean SSH RMSE
  0.073 m, no better than the prior. Working in model units gave 0.028 m on
  the same window (one window, joint model, 64 steps). `assimilate`
  therefore always works in model units.
- In the parent project (October 2026; historical, not reproduced here), a
  diagonal preconditioner fixed the same problem for a five-variable model,
  cutting U error from 0.36 to 0.10 m/s.

In general, if the physical state is `u = S x + b` (scale `S`, offset `b`)
and the physical observations are `y_phys = H u + e` with covariance
`R_phys`, then with an observation-space scaling `Q` the model-unit problem
is

```
y_model = Q (y_phys − H b),     A_model = Q H S,     R_model = Q R_phys Qᵀ
```

`assimilate` uses `Q = S⁻¹` for each observed variable, which is right for
plain masks. For averages over several cells or combinations of variables,
choose and record `Q` deliberately so the truncated solve is reproducible.

If you build your own observation operator, keep its rows of comparable
size (model units, or dividing each row by its `σ_y`).

## Guidance and solver settings

| Option | What it does | Status |
|---|---|---|
| `guidance` (`assimilate`), `guidance.method` (`sda.yaml`) | `mmps` or `rozet` | MMPS is the default |
| `rozet_gamma`, `rozet.gamma` | γ for the Rozet heuristic | 0.1 by default (Rozet & Louppe's code uses 0.01, with corrector steps); see above |
| `mmps_iterations` | Krylov iterations per step | 4 (`configs/sda.yaml` and the Python API); more iterations improve the linear solve at higher cost; check reconstruction quality |
| `mmps_solver` | `gmres`, `cg`, `minres` | GMRES default; others untested on these models |
| `mmps_warm_start` | start each solve from the previous step's solution | off; helped in the parent project's multi-window SDA (historical) |
| `mmps_preconditioner="diagonal"` | rescale rows by an estimate of `M`'s diagonal | off; redundant when already in model units with similar `σ_y` |
| `cov_y_inflation` (κ) | use `Σ_y + κ σ_t² I`; eases early steps, vanishes as σ_t → 0 | off; untested |
| `record_diagnostics` | save per-step residuals (MMPS) and the misfit in units of `σ_y` (Rozet), per observation type | off; the first thing to check if a variable is not being fitted |

## Choosing settings and evaluating reconstructions

Start with the defaults, then ask two separate questions: do the numerical
settings give stable results, and does the ensemble represent the errors
seen against independent truth? A close fit at observed pixels answers
neither question on its own.

- **`σ_y`** is the error you *assume*. Values that are too small can cause
  overfitting to noise and require more solver iterations; values that are too
  large underweight the observations. Start near the real instrument error
  plus representation error (the mismatch between what the instrument sees and
  what the model represents).
- **Steps and iterations.** 64 DDIM steps and 4 MMPS iterations are a starting
  configuration (the defaults of both `configs/sda.yaml` and the Python API,
  `inference/sda/defaults.py`), not a converged one. Check that results stop
  changing when you double each.
- **Ensembles.** Members differ only in their starting noise. The spread
  describes variability under the learned model and the chosen sampler;
  whether it is a calibrated uncertainty needs independent checks over many
  cases: spread against error at unobserved pixels, rank histograms and
  interval coverage (`inference/metrics.py`; `run_sda.py` prints a summary).
- **Check the fit.** The residuals at observed pixels, in units of `σ_y`, are
  a useful diagnostic (`observation_misfit`; the output's `observations`
  variable holds the values used). Their expected size depends on the
  posterior uncertainty, the prior and the observation model, so do not tune
  toward a fixed number. If a variable's misfit stays near its prior error,
  that variable is not being assimilated (see the units section above).
- **Unobserved regions.** Report errors away from observations separately from
  errors at observed locations (`reconstruction_errors`).
- **Timing.** On one L40S with 64 steps and 4 iterations, 4 members take about
  2.5 minutes for the joint 5-frame model and 6 minutes for the 11-frame
  SSH-only model.

## Models trained with observations as inputs

Everything above uses an unconditional prior. A conditional model takes the
observations as input channels; see [page 6](06_conditional_models.md). The
routes are:

| Route | Network input | Sampling guidance | `guidance` |
|---|---|---|---|
| Prior + SDA | frame times only | observation likelihood | `mmps`, `rozet` |
| Learned conditional (M_3) | frame times and observations | none | `conditional` |
| CFG (M_4) | conditional and null-observation branches | CFG combination | `cfg` |
| Hybrid (unvalidated) | observations as inputs | the same observations again | `mmps`, `rozet` with a conditional model |

Conditioning on one data source and assimilating a *different* one through
guidance can be meaningful, if the two are conditionally independent given
the state. Feeding the *same* observations through both routes is not
automatically better: a model that already approximates `p(x | y)` gets the
likelihood of `y` applied again, which can over-concentrate the ensemble.
The output's `route` attribute records which route was used.

## Tracks, averages, footprints, and correlated errors

The same formulation applies to cell averages, off-grid altimeter tracks,
radiometer footprints, and time averages by changing the linear operator
`A`. Correlated observation errors require a non-diagonal `Σ_y`.
`inference/sda/operators.py` provides these operators and block-diagonal
error covariances.

`assimilate_general` (`inference/general.py`) handles the unit conversions
described above and returns samples in physical units, together with
observation residual diagnostics. By default, it scales each observation
by its error standard deviation (`Q = 1/σ_y`).

This scaling matters when products have different accuracies. In the
notebook's combined SST-footprint and SSH-track example, scaling rows to
unit size in model units left SST residuals near 5 observation-error
standard deviations after four iterations. Scaling by observation error
reduced them to about 1.1.

For correlated errors, MMPS evaluates products with `Σ_y`; Rozet guidance
solves with `Σ_y + γσ_t² I` one covariance block at a time. Both are checked
against exact solutions in `tests/test_operators.py`. For worked examples,
see `examples/03_general_observations.ipynb`.

## Larger domains

**Longer time series** can be reconstructed with
[temporal sliding windows](08_windowed_sda.md), using the existing prior.

**Larger spatial regions** require a different extension.

One window is 128 × 128 cells. Covering larger regions means tiling
overlapping windows and enforcing consistency where they overlap. The parent project
has a windowed SDA for this; it is not included here yet.

## Code reference

| What | Where |
|---|---|
| One-call SDA in physical units | `src/obottle_starter/inference/assimilate.py` |
| MMPS guidance | `src/obottle_starter/inference/sda/mmps.py` |
| Rozet & Louppe (2023) guidance | `src/obottle_starter/inference/sda/rozet.py` |
| Solvers | `src/obottle_starter/inference/sda/solvers.py` |
| DDIM sampler and timestep grid | `src/obottle_starter/inference/sda/sampling.py` |
| Observation operators | `src/obottle_starter/inference/sda/observations.py` |
| Diagnostics (errors, spread, ranks, spectra) | `src/obottle_starter/inference/metrics.py` |
