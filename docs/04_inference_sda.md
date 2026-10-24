# 4. Inference with score-based data assimilation (SDA)

## The problem

We have sparse, noisy observations `y` (SWOT swaths, nadir tracks, cloudy
SST) of an ocean state `x`. Write the observations as

```
y = A x + ε,     ε ~ N(0, Σ_y)
```

where `A` picks out (or combines) the observed pixels and
`Σ_y = R = diag(σ_y,j²)` is the observation-error covariance for
independent errors, one variance per observation (`σ_y² I` when all are
equal). `assimilate` takes one `σ_y` per variable. The units must be
declared with the equation: `assimilate` works in physical units at its
interface and solves in model units, with `A`, `y` and `R` transformed
consistently (below). We want not one best guess but many plausible states
consistent with `y`: samples of the posterior `p(x | y) ∝ p(y | x) p(x)`.
The trained diffusion model supplies the prior `p(x)`. SDA combines the two
without retraining: the same model can be used with any mask, any `σ_y`,
any linear observation operator.

## How guidance works

Unconditional sampling repeatedly asks the denoiser for `x̂ = D(x_t, σ_t)`,
the expected clean state given the current noisy one, and takes a small
step towards it (the DDIM sampler, `src/obottle_starter/inference/sda/sampling.py`).
Guidance replaces `x̂` with an estimate of the clean state given the noisy
state **and the observations**, `E[x | x_t, y]`. The methods differ in how
they approximate this.

At every step, the guided estimate has the form

```
x̂_guided = x̂ + γ_t Jᵀ Aᵀ g,      where  ( Σ_y + A Cov[x | x_t] Aᵀ ) g = y − A x̂
```

with `J = ∂x̂/∂x_t` (the denoiser's Jacobian) and `γ_t = σ_t²`. (`Jᵀ v`
is one backward pass through the network: automatic differentiation gives
vector-Jacobian products, `autograd.grad(x̂, x_t, v) = Jᵀ v`, never `J`
itself.) In words:
compare the observations with what the current estimate predicts
(`y − A x̂`), weigh that misfit against how uncertain the estimate is at
the observed points, and push the whole field (not just the observed
pixels) in the direction that fixes it. The push spreads away from the
observations through the Jacobian, which encodes the spatial and temporal
correlations the model learned. This is how an SSH swath informs SSH beside
the swath, and SSH informs SST in the joint model.

The hard part is the term `Cov[x | x_t]`: how uncertain the denoiser's
estimate is.

## Three ways to handle the uncertainty

### Rozet & Louppe (2023): a simple heuristic

Replace the uncertainty term by a constant: the likelihood variance of each
observation becomes `σ_y² + γ σ_t²` for a tuning constant `γ`, the system
above is solved by a division, and each step costs one backward pass
through the network. This variance is added directly in observation space.
It equals "`Cov[x | x_t] ≈ γ σ_t² I` in state space, propagated through
`A`" (which gives `γ σ_t² A Aᵀ`) only when each observation selects one
model-unit value with unit weight, as plain masks do; for smoothing,
averaging or weighted channel combinations the two differ, and the code
implements the observation-space version.

- **Pros:** cheap, simple, no linear solver.
- **Cons:** ignores that the model knows some places better than others,
  and needs `γ` tuned by hand. Too large a `γ` lets the prior swamp the
  observations early in sampling; too small makes guidance unstable.
- **Record in this project (historical, parent project; not reproduced
  here):** tested only inside the multi-window SDA of the parent work
  (May 2026). With the settings tried (γ ≥ 1, 6–100
  iterations), guidance either left streak artefacts or produced almost no
  adjustment towards the observations. The analysis suggested `γ ≪ 1` or a
  schedule might work; that was not pursued. There is no single-window
  comparison against MMPS on these models.
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
| 0.01 (Rozet & Louppe's default) | Stronger guidance at every step: the heuristic variance term is 100× smaller than with γ = 1. That says nothing about the model's true uncertainty |
| → 0 | Guidance as strong as `σ_y` allows; risks overshooting and instability early, when `x̂` is still poor |

γ is a variance in model units, so it assumes observations in model units,
as `assimilate` builds them. Tune it by checking the misfit at observed
pixels (`record_diagnostics`) and the error at unobserved pixels.

### MMPS: use the network's own uncertainty (the default here)

Moment Matching Posterior Sampling (Rozet et al., 2024) uses
`Cov[x | x_t] ≈ σ_t² J`. For the exact conditional-mean denoiser this is an
identity (`Cov[x | z_σ] = σ² ∂E[x | z_σ]/∂z_σ`), so there is no hand-set
`γ`, and the assumed uncertainty is large where the model is unsure and
small where it is confident. The price is that `J` is a huge matrix
(millions of rows) that we never form. We only compute products with it by
automatic differentiation (`Jᵀ v`, one backward pass each, used in place
of `J v` because the exact `J` is symmetric). So the linear system is
solved iteratively, with a Krylov method.

What is exact and what is not. Four approximations sit between MMPS and
the true posterior, and a linear observation operator removes none of
them:

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
samples from a learned ocean prior are calibrated. The choices that remain
(`σ_y`, steps, iterations, solver) are real settings to check.

### DPS (Chung et al., 2023), for comparison

Diffusion Posterior Sampling drops the uncertainty term altogether and
steps along the gradient of `‖y − A x̂‖²` with a hand-set step size. It is
the cheapest but the least principled. In the parent project (historical;
not reproduced here) a naive form diverged (produced NaN fields) for the
small `σ_y` used here. It is not implemented in this repository.

## Krylov solvers: GMRES, CG or MINRES

The MMPS system `M g = b`, with `M = Σ_y + A Cov[x | x_t] Aᵀ`, is solved
with a fixed number of iterations per sampling step (`mmps_iterations`).
Each iteration needs products with the network's Jacobian (a backward pass
through the network). Measured on an L40S for the joint model, a guided step
with 4 iterations took 2.2 s against 0.17 s for an unconditional step,
about 13× as long.

| Solver | Assumes | Minimizes | Notes |
|---|---|---|---|
| **GMRES** (default) | nothing | the residual ‖b − M g‖ over the iterations so far | Safe for any `M`. Memory grows with the iteration count, which is fine for the few iterations used here. |
| **CG** | `M` symmetric positive definite | the error in the `M`-norm | Cheapest per iteration and the classic choice when its assumption holds. |
| **MINRES** | `M` symmetric | the residual | Like GMRES but with short recurrences; tolerates indefinite `M`. |

For the exact denoiser `M` is symmetric positive definite: `Cov[x | x_t]`
is a covariance. But the network's Jacobian is only approximately
symmetric (and the code uses `Jᵀ v` where the formula has `J v`), so CG and
MINRES rest on an approximation. That is why GMRES is the default; it
assumes nothing about `M`, but that does not make the inference accurate
for every operator.
Whether CG or MINRES do as well at 2–8 iterations on these models has
**not been tested**; the tests only confirm that all three match the exact
answer for a Gaussian prior when run to convergence (`tests/test_mmps.py`).
It is a cheap experiment: `mmps_solver="cg"`.

### Why units and scaling matter for a truncated solve

With only a few iterations the solve is far from converged, and Krylov
methods reduce the largest residuals first. If one observation type has
much larger numbers than another, it gets all the attention. This happened
in this project twice:

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

## Options in `mmps.py`

| Option | What it does | Status |
|---|---|---|
| `guidance` (`assimilate`), `guidance.method` (`sda.yaml`) | `mmps` or `rozet` | MMPS is the default |
| `rozet_gamma`, `rozet.gamma` | γ for the Rozet heuristic | 0.01 by default; see above |
| `mmps_iterations` | Krylov iterations per step | 4 (`configs/sda.yaml` and the Python API); more fits the observations better at higher cost |
| `mmps_solver` | `gmres`, `cg`, `minres` | GMRES default; others untested on these models |
| `mmps_warm_start` | start each solve from the previous step's solution | off; helped in the parent project's multi-window SDA (historical) |
| `mmps_preconditioner="diagonal"` | rescale rows by an estimate of `M`'s diagonal | off; redundant when already in model units with similar `σ_y` |
| `cov_y_inflation` (κ) | use `Σ_y + κ σ_t² I`; eases early steps, vanishes as σ_t → 0 | off; untested |
| `record_diagnostics` | save per-step residuals (MMPS) and the misfit in units of `σ_y` (Rozet), per observation type | off; the first thing to check if a variable is not being fitted |

## Practical guidance

- **`σ_y`** is the error you *assume*. Too small and the sampler strains to
  fit noise (and needs more iterations); too large and observations are
  under-used. Start near the real instrument error plus representation
  error (the mismatch between what the instrument sees and what the model
  represents).
- **Steps and iterations.** 64 DDIM steps and 4 MMPS iterations are a
  starting configuration (the defaults of both `configs/sda.yaml` and the
  Python API, `inference/sda/defaults.py`), not a converged one. Check that
  results stop changing when you double each.
- **Ensembles.** Members differ only in their starting noise. The spread
  describes variability under the learned model and the chosen sampler;
  whether it is a calibrated uncertainty needs independent checks over many
  cases: spread against error at unobserved pixels, rank histograms and
  interval coverage (`inference/metrics.py`; `run_sda.py` prints a
  summary).
- **Check the fit.** The residuals at observed pixels, in units of `σ_y`,
  are a useful diagnostic (`observation_misfit`; the output's
  `observations` variable holds the values used). Their expected size
  depends on the posterior uncertainty, the prior and the observation
  model, so do not tune toward a fixed number. If a variable's misfit stays
  near its prior error, that variable is not being assimilated (see the
  units section above).
- **Errors where it matters.** Fitting observed pixels is easy; report the
  error at unobserved pixels separately (`reconstruction_errors`).
- **Timing.** On one L40S with 64 steps and 4 iterations, 4 members take
  about 2.5 minutes for the joint 5-frame model and 6 minutes for the
  11-frame SSH-only model.

## Feeding observations to the network instead

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

## Observations that are not pixels

Everything above observes model pixels through masks. Coarse products
(cell averages), altimeter tracks (off-grid points, between frame times),
radiometer footprints and time-averaged products are also linear in the
state, so the same machinery applies with a different `A`. The errors may
also be correlated, e.g. along a track, so `Σ_y` is no longer diagonal.
`inference/sda/operators.py` provides these operators and block-diagonal
covariances. `assimilate_general` (`inference/general.py`) applies the
observation-space scaling above (`A_model = Q H S`, `y_model = Q y_phys`,
`R_model = Q R Qᵀ`) and returns physical-unit samples with
observation-space diagnostics. Its default puts every observation in units
of its own error (`Q = 1/σ_y`). With mixed products of very different
accuracy that matters: in the notebook's joint SST-footprint and SSH-track
example, scaling rows to unit size in model units instead left the SST
footprints about 5 error stds from their observations after 4 iterations,
against about 1.1 with the default. With correlated `Σ_y`, MMPS multiplies by
`Σ_y` and Rozet solves with `Σ_y + γσ_t² I`, block by block; the exactness
tests cover both (`tests/test_operators.py`). See
`examples/03_general_observations.ipynb`.

## Larger domains

One window is 128 × 128 cells. Covering larger regions means tiling
overlapping windows and making them agree at the seams. The parent project
has a windowed SDA for this; it is not included here yet.

## Where to look in the code

| What | Where |
|---|---|
| One-call SDA in physical units | `src/obottle_starter/inference/assimilate.py` |
| MMPS guidance | `src/obottle_starter/inference/sda/mmps.py` |
| Rozet & Louppe (2023) guidance | `src/obottle_starter/inference/sda/rozet.py` |
| Solvers | `src/obottle_starter/inference/sda/solvers.py` |
| DDIM sampler and timestep grid | `src/obottle_starter/inference/sda/sampling.py` |
| Observation operators | `src/obottle_starter/inference/sda/observations.py` |
| Diagnostics (errors, spread, ranks, spectra) | `src/obottle_starter/inference/metrics.py` |
