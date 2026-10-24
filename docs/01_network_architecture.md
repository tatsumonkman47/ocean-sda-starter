# 1. The network

## What a diffusion model learns

Suppose you have many example ocean states `x` (a few frames of SSH and SST
on a 128 × 128 patch). A diffusion model learns to produce new, realistic
states that look like they came from the same distribution.

It does this indirectly. Take a real sample `x`, add Gaussian noise,
`z_σ = x + σ ε` with `ε ~ N(0, I)`, and train a network `D_θ(z_σ, σ)` to
recover the clean sample. This is the **denoiser**. Once it works for every
σ from tiny to huge, you can generate a sample: start from pure noise (very
large σ) and repeatedly denoise a little, lowering σ step by step until it
is near zero.

σ sets the signal-to-noise ratio. A useful intuition is that large σ
settles the large-scale structure and small σ the fine detail: noise of a
given σ hides the features whose amplitude is below σ, and in ocean fields
the large scales usually carry the most variance. How σ relates to
wavelength therefore depends on the data's spectrum; it is not a fixed
ordering by scale.

Three facts make this useful for data assimilation (page 4):

1. The training loss is a squared error, so the best possible denoiser (the
   population optimum, with unlimited data and capacity) is the
   **expected clean state given the noisy one**,
   `D(z_σ, σ) = E[x | z_σ]`. A trained network approximates it.
2. **Tweedie's identity** links that conditional mean to the gradient of
   the log-density of noisy states (the "score"):
   `∇ log p(z_σ) = (E[x | z_σ] − z_σ) / σ²`. Replacing the mean by
   `D_θ` gives the learned score, which is what lets us steer sampling
   towards observations.
3. For the exact conditional mean, its Jacobian gives the uncertainty of
   that estimate: `Cov[x | z_σ] = σ² ∂E[x | z_σ]/∂z_σ`. With a learned
   denoiser this is an approximation (the learned Jacobian need not even be
   symmetric). MMPS uses it.

## EDM preconditioning: keeping numbers of order one

Inputs at σ = 200 are huge and inputs at σ = 0.0002 are almost clean, so the
raw network would see wildly different magnitudes. Karras et al. (2022)
wrap the network `F` in fixed scalings that keep its inputs and targets near
unit size:

```
D_θ(z, σ, c) = c_skip(σ) · z  +  c_out(σ) · F_θ( [c_in(σ) · z, c_maps],  c_noise(σ),  c_rest )

c_in   = 1 / sqrt(σ² + σ_data²)
c_skip = σ_data² / (σ² + σ_data²)
c_out  = σ · σ_data / sqrt(σ² + σ_data²)
c_noise = ln(σ) / 4
```

Only the noisy state is scaled by `c_in`. The context maps (`c_maps`: the
frame-offset encodings and, for a conditional model, observation maps) are
concatenated to it unscaled, and the rest of the context (`c_rest`: frame
offsets, calendar, labels) enters through embeddings, as `c_noise` does.

`σ_data` is the typical standard deviation of the (normalized) data, 1.0
for the released models. At small σ, `c_skip ≈ 1`: the input is nearly
clean, so `D` mostly passes it through and `F` predicts a small correction.
At large σ, `c_skip ≈ 0`: `F` must predict the clean field itself. This
wrapper is `EDMPrecond` in `src/obottle_starter/core/nn/networks.py`.

## The U-Net (`SongUNet`)

`F` is a U-Net, named for its shape:

```
128x128 ─ blocks ────────────────────────────── blocks ─ 128x128   (96 channels)
   │ downsample                                     ▲ upsample
 64x64 ─ blocks ──────────────────────── blocks ─ 64x64            (192)
   │                                                ▲
 32x32 ─ blocks + temporal attn ── blocks + temporal attn ─ 32x32  (192)
   │                                                ▲
 16x16 ─ blocks + spatial attn + temporal attn ─────┘              (192)
```

- The **encoder** (left) halves the grid at each level and adds feature
  channels, so deep layers see large areas with many features. The
  **decoder** (right) builds the grid back up.
- **Skip connections** (horizontal lines) pass encoder features directly to
  the decoder at the same size, so fine detail does not have to survive the
  trip through 16 × 16.
- Each level has `num_blocks` **residual blocks**: convolution, GroupNorm
  and SiLU activation, added back onto the block's input.
- **Spatial self-attention** at 16 × 16 (`attn_resolutions`) lets every
  point compare itself with every other point on the patch, which
  convolutions (3 × 3 neighbourhoods) cannot do in one step.

In the config these are `model_channels` (96), `channel_mult` ([1, 2, 2, 2]),
`num_blocks` (2) and `attn_resolutions` ([16]). Patches must be square and
divisible by 2 once per downsampling; `attn_resolutions` and
`temporal_attn_resolutions` are grid sizes, so they must name levels that
exist. Both are checked before the network is built, and training prints
the resulting levels, widths and attention placement. With these the released
models have 23.7 million parameters. "SongUNet" refers to Song et al.
(2021)'s DDPM++ architecture, as configured in EDM.

## How the network handles several frames

A sample is a short movie of `time_length` frames (5 or 11). The network
sees all frames at once, as a tensor of shape `(batch, channels, frames,
pixels)`.

- **Convolutions and spatial attention act on each frame separately**
  (`mixing_type: spatial`). On their own they could not relate one frame
  to another.
- **Temporal attention** (at 32 × 32 and 16 × 16, `temporal_attn_resolutions`,
  plus the bottom and upsampling blocks of the decoder) lets each grid
  point look at its own value in all the other frames. This is how the
  model learns that an eddy at frame 1 should have moved a little by
  frame 2. The pattern, spatial layers plus temporal attention, follows
  Ho et al. (2022).

### Telling the network when the frames are

Frames may be 1 hour or 10 days apart, and the network needs to know which.
The released models get the frame times in four ways (all set up by
`src/obottle_starter/core/conditioning.py`):

| Mechanism | Config key | What it gives the network |
|---|---|---|
| Offset channels | `temporal_offset_channels: 32` | 32 sine/cosine encodings of each frame's offset from frame 0 (hours), as extra input maps. Periods from 7.2 h to 7,200 h. |
| Attention time bias | always on | Temporal attention adds a learned function of the time difference between each pair of frames (periods 1–1,000 h) to its attention scores. |
| Per-frame AdaGN | `use_per_frame_adagn: true` | Each frame's offset is embedded and added to the noise-level embedding, so every residual block gets a frame-specific signal. See below and page 5. |
| Calendar | `calendar_embed_channels: 8` | Day of year (seasons) and local solar time of day (diurnal cycle), as extra input maps. |

The time of day is local to the patch: UTC time plus longitude × 4 minutes
per degree. This is why `assimilate` needs `center_lon`.

The four routes have different purposes, but whether all four are needed
has not been tested: no matched comparison (same data, seeds, compute and
evaluation, one route switched off at a time) exists. It is a good student
experiment; judge it on SDA error at several frame spacings and on temporal
correlations of prior samples, not on one training-loss curve.

### What "per-frame AdaGN" actually does in these models

Every residual block receives an embedding vector. Normally this carries
only the noise level σ. With `use_per_frame_adagn`, a vector computed from
each frame's time offset is added to it, so the embedding differs between
frames. The block then turns the embedding into one number per feature
channel (per frame) and **adds it to the features before the second
GroupNorm**. (This is EDM's SongUNet design: `adaptive_scale` off. The
name "AdaGN", adaptive group normalization, usually means a scale *and*
shift applied after normalization; that variant is available for new models
as `per_frame_adaptive_scale: true`, but the released models do not use it
and it changes parameter shapes. See page 5.)

The new projection is initialized to zero, so a model starts training as if
the option were off. Whether it helps has not been tested by a matched
comparison; page 5 discusses this.

## Other inputs that are present but inactive

- `label_dim: 8`. There is a slot for a global label vector, but the code
  always passes zeros, so it acts as a learned constant. It is kept because
  the released checkpoints contain its weights.
- No observation maps are fed to the released networks. They are
  unconditional; observations enter only at inference, through SDA. The
  `conditioning` section can add observation inputs for new models; see
  [page 6](06_conditional_models.md).

## Where to look in the code

| What | Where |
|---|---|
| Build the network from a config | `src/obottle_starter/core/model.py`, `build_model` |
| U-Net, blocks, attention, preconditioning | `src/obottle_starter/core/nn/networks.py` (`SongUNet`, `UNetBlock`, `TemporalAttention`, `EDMPrecond`) |
| Time and calendar embeddings | `src/obottle_starter/core/nn/embedding.py`, `src/obottle_starter/core/conditioning.py` |
