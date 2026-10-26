# 8. Long sequences: sliding windows

The released models were trained on short sequences: **5 frames for joint
SSH/SST** and **11 frames for SSH alone**. At the example tiles' 12-hour
spacing, five frames span two days. A reconstruction often needs to cover
much longer than that.

Rozet and Louppe's sliding-window construction lets us reuse a short-window
model within **one diffusion process over the whole reconstruction period**.
This page explains how it works, why that is useful for data assimilation,
and what remains approximate.

For a worked comparison, [notebook 05](../examples/05_sliding_windows.ipynb)
reconstructs 20 frames (9.5 days) using sliding windows and independent
five-frame blocks. You will need the
[released models and example data](../README.md#download-models-and-example-data)
and a GPU. The commands below provide a shorter starting point.

## Choosing how to reconstruct a long period

There are several ways to extend a model beyond its training window:

| Approach | How it uses the short-window model | Main consideration |
|---|---|---|
| Independent blocks | Reconstruct consecutive blocks separately | Each block uses only its own observations; discontinuities can appear at boundaries. |
| Overlap and average | Reconstruct overlapping windows separately, then blend them | Averaging can hide discontinuities but changes sample variability and tends to smooth the fields. |
| Autoregressive generation | Generate the next frame or block conditioned on previously generated states | A forward-only rollout can carry early errors forward and does not revise earlier states when later observations become available. |
| Sliding-window SDA | Assemble local predictions at every step of one full-sequence diffusion process | All frames remain adjustable, but the local-window and guidance approximations still need evaluation. |

`assimilate` uses sliding windows when given more frames than the model was
trained on. Set `window_step=cfg.time_length` for the independent-block
comparison, subject to the sequence-length requirements below. An
autoregressive rollout is not implemented by this option.

### Why not generate the sequence autoregressively?

Generating a sequence one frame or block at a time is a natural choice for
forecasting. For retrospective reconstruction, however, an observation
later in the period may tell us something about an earlier state. A SWOT
pass on day 5, for example, may help locate an eddy during a gap on day 4.

A forward-only rollout that fixes day 4 before processing day 5 cannot
use that information to revise day 4. Errors in the generated history can
also become inputs to later predictions. **Data assimilation over a fixed
period is often a smoothing problem:** we want earlier and later
observations to constrain the trajectory together.

Autoregression itself is not incompatible with that goal. A joint posterior
can be factorized sequentially if each conditional accounts for all relevant
observations, and an autoregressive model can be used with backward smoothing
or repeated updates. The limitation is the simpler procedure that generates
forward once and freezes its past.

Sliding-window SDA keeps the whole noisy sequence adjustable at every
diffusion step. This makes two-way information exchange possible without
training a network on the entire period. It does **not** establish that
sliding windows outperform a well-designed autoregressive smoother; the
notebook compares them with independent blocks, not with such a smoother.

## How the windows are assembled

Let the model's window have `T = 2k + 1` frames and the sequence `L`
frames. At each sampling step:

1. cut the noisy sequence `x_t` into the `L − T + 1` windows of `T`
   consecutive frames (frames 0-4, 1-5, 2-6, ... for `T = 5`);
2. denoise every window with the network, each with its own frame offsets
   and calendar, exactly as in training;
3. keep, for each frame, the estimate from the window centred on it; the
   first and last `k` frames come from the first and last windows.

<p align="center">
  <a href="figures/rozet_sliding_window.pdf"><img src="figures/rozet_sliding_window.png" alt="A long sequence is denoised by evaluating overlapping temporal windows at each diffusion step"></a>
  <br>
  <em>Physical time runs vertically and denoising progresses horizontally. At each step, local window predictions are assembled into a full-sequence estimate. Click for the original PDF. In this figure, t labels physical time and tau labels the diffusion process.</em>
</p>

For `L = 9`, `T = 5` (`k = 2`), with `•` the frames each window keeps:

```
frame        0  1  2  3  4  5  6  7  8
window 0     •  •  •  ·  ·
window 1        ·  ·  •  ·  ·
window 2           ·  ·  •  ·  ·
window 3              ·  ·  •  ·  ·
window 4                 ·  ·  •  •  •
```

The assembled estimate has one prediction per frame. We apply observation
guidance to that full-sequence estimate and update the full noisy sequence
before evaluating the windows again. We are not generating complete windows
independently and joining them at the end.

The figure's physical-time axis and diffusion axis serve different roles:
physical time locates a frame in the ocean sequence; a diffusion step
updates the reconstruction of all those frames.

## Why local windows can describe a long sequence

For a first-order Markov chain, an interior state is conditionally
independent of the rest of the trajectory given its immediate neighbours.
Its contribution to the joint score,
`∇ᵢ log p(x₀, …, x_{L−1})`, can therefore be obtained from a local window.
Tweedie's relation, `score = (D(x_t) − x_t) / σ_t²`, connects this score
construction to the denoiser used here.

Adding noise generally destroys that exact locality. Rozet & Louppe use a
wider neighbourhood, called a *pseudo-Markov blanket*, to approximate it.
Their limiting and unbiasedness results rely on the assumptions in the
paper; they do not guarantee accuracy for a learned ocean-surface prior.
See [section 3.1 and Appendix A](https://arxiv.org/html/2306.10574v2).

### What changes for ocean surface fields

There are three additional considerations here:

- **SSH and SST are not the ocean's full state.** The ocean interior, which the
  model never sees, carries memory, so we should not assume that surface fields sampled every 12 hours
  form a first-order Markov chain. The SSH-only model's 11-frame window (`k = 5`) covers a longer interval
  than the joint model's five-frame window (`k = 2`) at the same spacing,
  but the models also differ in variables and training; this is not an
  isolated test of window length.
- **The window must be wide enough at every noise level.** Results for
  the limits of very low and very high noise do not ensure that a chosen
  window captures the dependencies at intermediate noise levels.
- **No corrector steps.** Rozet & Louppe add a few Langevin corrector steps
  after each sampler step so that the errors of the window and likelihood
  approximations do not accumulate. The DDIM sampler here has none, which
  reduces cost. Increasing `steps` refines the DDIM discretization, but is
  not equivalent to adding corrector steps or removing the window approximation.
  [Page 4](04_inference_sda.md#langevin-corrector-steps-not-implemented-here)
  explains corrector steps and the code changes needed to experiment with them.

**Local windows can transmit information across the sequence.** An interior
frame uses a window with `k` neighbours on either side; boundary frames use
the first or last full window. Repeated sampling steps can propagate a
change through successive overlapping windows. An observation in frame 2 can
therefore shape frame 15, through the frames in between, as in a smoother.

## Guidance on a long sequence

The observation operator covers all `L` frames. Both guidance methods
differentiate through the assembled prediction, including the local
networks that produced it.

- **Rozet guidance** uses one backward pass through the windows per step.
  The prescribed variance is `σ_y² + γ σ_t²`; smaller `γ` generally makes
  the correction stronger. The current default is **`γ = 0.1`**. The
  tutorial reports divergence with `γ = 0.01` for the released joint model
  under this DDIM setup, even for a single window. Rozet and Louppe used
  0.01 with a predictor-corrector sampler. These results do not isolate the
  effect of corrector steps, so treat `γ`, the sampler, and the step count
  as settings to evaluate together.
- **MMPS** uses `σ_t²` times the assembled denoiser's Jacobian as an
  approximation to conditional covariance. Rows selected from different
  windows need not form a symmetric matrix, even if each local Jacobian
  is symmetric. **Use GMRES**, which does not require symmetry; CG and
  MINRES do. GMRES addresses the linear solve, not the accuracy of the
  covariance approximation.

See [the guidance discussion](04_inference_sda.md) for the equations and
sampler limitations.

## Cost and memory

Each denoiser call runs the network on every window: `(L − T) / step + 1` of
them. For 20 frames and the 5-frame model, sliding windows (`step = 1`) cost
16 window evaluations per call, against 4 for independent blocks. With
guidance, the activations of every window are kept for the backward pass,
so memory also grows with the number of windows times the members per
batch. Actual runtime and memory use depend on the model, sequence length,
solver settings, and GPU. The notebook records about 10.5 minutes for
sliding windows versus 2.4 minutes for independent blocks in its illustrated
four-member, 64-step run. Treat these as example timings, not a general
performance guarantee.

To fit in memory, lower `n_batch` (members per GPU batch) first.
`window_batch` (Python) or `sliding.window_batch` (`run_sda.py`) sets the
most windows per network call; it lowers the peak of each call, but with
guidance all activations are still kept until the backward pass.

## Choosing the spacing between windows

`window_step` (Python) or `sliding.step` (`run_sda.py`) sets how far apart
consecutive windows start, from 1 to `T`. Each window keeps its central
`step` frames, so `step = 1` is Rozet & Louppe's construction and
`step = T` is independent blocks. Values in between trade cost for
coupling: with `step = 3` and `T = 5`, each retained interior frame has at least one
neighbour on each side inside its window. Sequence endpoints are exceptions.

**Valid sequence lengths:** `(L − T)` must be a multiple of `step`; otherwise the error message suggests the nearest lengths that
work.

## Run a reconstruction

### Python

After loading `net`, `cfg`, and `names` as in the
[README example](../README.md#python), request a longer sequence from the
tile. The timestamps, masks, and truth (or real observations `y`) must all
cover the same frames.

```python
win = load_window("osse_tiles/MidAtlantic_35N_50W.nc", names, 20)   # 20 frames
post = obs.assimilate(net, cfg, times=win.times, center_lon=win.center_lon,
                      truth=win.truth, masks=win.masks,
                      sigma_y={"SSH": 0.005, "SST": 0.1},
                      guidance="rozet", window_step=1, n_batch=1)
```

### Command line

```bash
python scripts/run_sda.py ckpt=checkpoints/joint_ssh_sst_T5.pt \
    tile=osse_tiles/MidAtlantic_35N_50W.nc output=out/long.nc \
    sliding.frames=20 guidance.method=rozet sampler.n_batch=1
```

**Current scope:** sliding windows work for unconditional priors. A conditional model takes
observation maps for its own `T` frames as inputs; extending that to
windows would need each window's observations as its inputs, which is not
implemented.

## Evaluate the reconstruction

The notebook separates temporal continuity from reconstruction accuracy.
Removing a visible boundary does not by itself make a sample more accurate
or its uncertainty better calibrated.

- **Changes between frames:** compare each member's temporal increments
  with the truth, both at block boundaries and elsewhere. Reduced boundary
  jumps can coexist with excessive variability throughout the sequence.
- **Accuracy:** compare errors on the same frames using identical
  observations. Extra observations may help, but approximation errors and
  finite ensemble size mean that improvement is not guaranteed.
- **Single-window agreement:** when the sequence is exactly one model
  window long, check numerical agreement with the single-window path.
  This is a code check, distinct from comparing long- and short-sequence
  reconstruction quality.
- **Observation coverage:** inspect masks alongside error curves. A rise
  in error during an observation gap is different from a window-boundary
  artefact.
- **Ensemble spread and endpoints:** evaluate uncertainty against error,
  including the first and last frames, which have less surrounding context.
- **Multiple cases:** repeat across locations, dates, and seeds. One tile
  illustrates behaviour; it does not establish general performance.

This page covers windows in **time**. Covering a domain larger than the
model's 128 × 128 patch with overlapping **spatial** windows is a different
method, with its own consensus problem between neighbouring patches; it is
not included here.

## Code reference

| What | Where |
|---|---|
| Window layout, assembled denoiser | `src/obottle_starter/inference/sda/windows.py` |
| Long sequences in `assimilate` | `src/obottle_starter/inference/assimilate.py` |
| Command-line options | `configs/sda.yaml`, `sliding:` |
| Tests (layout, conditioning, gradients, one-window agreement) | `tests/test_windows.py` |

## Reference

Rozet, F. & Louppe, G. (2023). *Score-based Data Assimilation.* NeurIPS.
[arXiv:2306.10574](https://arxiv.org/abs/2306.10574). The construction is
their section 3.1 and Algorithm 2; their reference implementation is
[github.com/francois-rozet/sda](https://github.com/francois-rozet/sda).
