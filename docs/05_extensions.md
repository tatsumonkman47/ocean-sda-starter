# 5. Extensions and modifications

If you want to adapt the examples, first identify what needs to change:
the fields the model learns, the observations it receives, or the network
itself. Many changes require only a configuration file; others need new
code and validation.

The tables below distinguish those cases. **Implemented** means a code
path exists; it does not necessarily mean that it has been evaluated on
ocean reconstructions. Results from the parent projects are identified
separately because they have not necessarily been reproduced here.

## Choosing a first experiment

- **New fields or regions:** start with the `variables` configuration and
  [data preparation](03_data_loading.md).
- **Different satellite products:** start with the observation operators in
  [data assimilation](04_inference_sda.md).
- **Observations as network inputs:** use the small example in
  [conditional models](06_conditional_models.md) before training on real data.
- **A longer reconstruction period:** see [sliding windows](08_windowed_sda.md).

For comparisons, change one feature at a time and keep the training data,
compute budget, and evaluation cases comparable.

## Network

### Frame-dependent feature adjustments (`use_per_frame_adagn`)

**What it changes.** Each residual block receives a numerical
representation of the frame's time offset, added to its noise-level input.
This lets the block adjust its features according to frame time (page 1).
The option is enabled by default because the released models use it.

**Evidence.** No controlled comparison exists: no model has been trained
with it on and off under otherwise identical settings, in this repository
or the parent projects. The frame times already reach the network through
the offset channels and the attention time bias, so it may add little. It
adds little computational cost: two small linear layers, initialized to zero.

**Two variants exist:**

- *Shift before normalization* (released models, `per_frame_adaptive_scale`
  off): the per-frame signal is added to the features just before a
  GroupNorm. This is EDM's original SongUNet design for the noise-level
  signal too.
- *Scale and shift after normalization* (`per_frame_adaptive_scale` on, the
  usual meaning of "AdaGN"): the signal multiplies and shifts the normalized
  features. The parent project uses this scale-and-shift variant for new
  SongUNet runs. Set `per_frame_adaptive_scale: true` (with
  `use_per_frame_adagn: true`). It changes layer shapes, so it is a choice
  for new models; the released checkpoints cannot use it.

**Suggested comparison:** train two small models, identical except for
`use_per_frame_adagn`, and compare prior spectra and SDA error at several
frame spacings.

### Network size, attention, and grid choices

| Change | Config | Notes |
|---|---|---|
| Wider or deeper | `model_channels`, `num_blocks`, `channel_mult` | In the parent project's EDM2 capacity tests, early results suggested a wider finest level improved velocity spectra. |
| More spatial attention | `attn_resolutions: [16, 32]` | Attention memory grows quadratically with the number of grid points. |
| Larger patches | `H`, `W` (256) | Patches must be square. For four levels at 256 cells, valid attention sizes are 256, 128, 64, and 32; `[16]` is rejected. Larger patches at 4 km spacing cover more area without increasing spatial resolution. |
| Rectangular patches | — | Not supported: `SongUNet` builds square levels. Supporting them means tracking (height, width) through every level, with tests on non-square inputs. |
| Boundaries | — | Convolutions zero-pad the patch edges. Periodic domains or coastlines are different assumptions and would need their own padding or masking. |
| Different frame counts | `time_length` | Fixed per model in this repository's code; `frame_conditioning` checks it. |
| Window-span label | `span_label_channels` | An older way to signal the window length; superseded by the per-frame signal. |

### EDM2 networks

Karras et al. (2024)'s EDM2 network keeps activations and weights at a
controlled magnitude ("magnitude-preserving" layers) and learns a
per-noise-level loss weighting. The parent project has an ocean adaptation
with temporal attention, trained on Lagrangian-filtered SSH, U, V and SST.
It is not included yet: the choice of model is still being made, it needs
a different dataset and normalization, and NVIDIA's EDM2 code is licensed
CC BY-NC-SA 4.0 (non-commercial, share-alike), unlike this repository's
Apache-2.0.

## Training

| Idea | Status |
|---|---|
| `noise_distribution: log_normal` or `power` | Implemented; `log_normal` takes `p_mean` and `p_std`. The default and the released models use `log_uniform`. |
| Fine-tuning a released model | Supported: `init_from=checkpoints/joint_ssh_sst_T5.pt` starts a new run from its EMA weights with a fresh optimizer (the shared checkpoints have no optimizer state). Variables, normalization and network must match; new observation inputs are zero-initialized (page 2). |
| Conditional models (observation maps as network inputs; M_3 / M_4) | Basic training and sampling checks pass; reconstruction quality has not been evaluated: `conditioning` section, page 6. |
| Weighting variables in the loss | Not implemented here. The parent projects tried per-channel and noise-level-dependent weights. |
| Other variables (e.g. U, V) | Add an entry to `variables` with its directory, units and normalization (root README, [Training data format](../README.md#training-data-format)). No code changes for affine normalization. |
| Incomplete training fields or coastlines | Not supported: targets must be complete (page 2, "Missing data"). |
| Calendar-based climatology | `climatology.lookup: day_of_year` for new models (any year); the released models use `elapsed_days`, valid only inside the climatology file's dates. |

## Data

| Idea | Status |
|---|---|
| Native-grid patches from a large zarr store | Used in the parent project; not included (page 3). |
| Other regions | Any data in the tile layout works. The released models saw only the North Pacific; the OSSE tiles test transfer to the Atlantic. |
| Wind or other forcing as conditioning | Being tested in the parent project with per-frame wind labels and maps. Not included. |

## Data assimilation and sampling

| Idea | Status |
|---|---|
| Rozet & Louppe (2023) covariance heuristic | Implemented (`guidance: rozet`, page 4). Checked against exact formulas on a Gaussian prior. Default γ = 0.1 from a single-window comparison (0.01 diverged without corrector steps); not systematically tuned. |
| Langevin corrector steps after each sampler step (Song et al. 2021; used by Rozet & Louppe) | Not implemented. Background, and where to add a `corrections` option to experiment with the number of steps, on page 4. |
| CG or MINRES instead of GMRES | Implemented; untested on these models. |
| Reuse of the previous solution, diagonal scaling, observation-error inflation | Implemented and checked against an exact Gaussian posterior; untested on these models. |
| Observations beyond individual grid cells (off-grid points, coarse averages, footprints, time averages, sums of variables) | Available through `inference/sda/operators.py` and `assimilate_general` (`inference/general.py`), in physical units with consistent scaling; `examples/03_general_observations.ipynb`. Linear operators only. Mask-based `ObservationTerm(transforms=...)` still works for operators represented on the model grid. |
| Correlated observation errors | Supported with the general operators: `ObservationCovariance` with dense blocks (e.g. one per track, `exponential_covariance`), in MMPS and Rozet. Dense blocks are factorized, so keep each block to a few thousand observations. |
| Nonlinear observation operators | Not provided. MMPS would linearize them at the denoised estimate each step, which is an approximation. |
| General observations with conditional models | Not supported: a conditional model's observation inputs are mask-shaped. |
| Longer time sequences | Implemented for unconditional priors through temporal sliding windows; see [page 8](08_windowed_sda.md) for assumptions and checks. |
| Larger spatial domains (windowed SDA) | In the parent project; to be added after publication. |
| Real observations | `assimilate(..., y=...)`. Resample them to the 4 km tile grid first, and match the SSH filtering (300 km high-pass) the model was trained on. |
