# Documentation

These pages explain how the examples work and what you would need to change
for your own application. The aim is to connect the mathematics to practical
choices: what the model learns, how we prepare its training data, and how
observations steer a reconstruction.

You do not need prior experience with diffusion models. Some familiarity
with probability, linear algebra, and Python will help; the ocean examples
provide the physical context.

## Practical guides

Choose a guide for the task you want to run. Commands assume you are in the
repository root; each guide links to its prerequisites and relevant background.

| Guide | Purpose |
|---|---|
| [Installation](quickstart/installation.md) | Set up the package and tested environment |
| [Downloads](quickstart/downloads.md) | Get released models and data and verify checkpoints |
| [Inference](quickstart/inference.md) | Reconstruct fields from observations and inspect outputs |
| [Training](quickstart/training.md) | Train, resume, and configure models for your own data |
| [Units and scaling](quickstart/units.md) | Keep data, observations, and model normalization consistent |

## Technical background

The [conceptual introduction](introduction.md) uses diagrams to explain
training and SDA. The pages below develop the mathematics and implementation.

| Page | Read it to learn |
|---|---|
| [1. The network](01_network_architecture.md) | What a diffusion model is, the U-Net inside it, and how it handles time |
| [2. Training](02_training.md) | The training objective, the noise schedule, and the settings in `configs/train_base.yaml` |
| [3. Training data](03_data_loading.md) | Preparing local patches from llc4320 and selecting their frame times |
| [4. Data assimilation](04_inference_sda.md) | Posterior sampling, covariance approximations, and numerical solvers |
| [5. Extensions](05_extensions.md) | Available modifications, proposed experiments, and their validation status |
| [6. Conditional models](06_conditional_models.md) | Training models that take observations as inputs, with optional classifier-free guidance |
| [7. References](07_literature.md) | Papers grouped by topic, with their contributions and connections to the code |
| [8. Sliding windows](08_windowed_sda.md) | Reconstructing longer time sequences with a model trained on short windows |

The [project README](../README.md) provides a short overview. See also
[research background](background.md) and [planned additions](roadmap.md).

## Reading order

You can read the guide in order, or start with the task you have in mind:

- **Understand diffusion models:** start with [the network](01_network_architecture.md),
  then [training](02_training.md) and [data assimilation](04_inference_sda.md).
- **Use the released models:** start with [data assimilation](04_inference_sda.md).
  For longer time series, continue to [sliding windows](08_windowed_sda.md).
- **Train on your own fields:** read [training](02_training.md) and
  [data preparation](03_data_loading.md) together.
- **Try a different method:** see [extensions](05_extensions.md) and
  [conditional models](06_conditional_models.md).

The [references](07_literature.md) explain where each method comes from.
The notation below is shared across the guide.

## Terminology

| Term | Meaning here |
|---|---|
| **Channel** | One variable on the grid, e.g. SSH or SST (the network's input and output channels). |
| **Frame** | One snapshot in time. A sample is several frames of the same patch. |
| **Patch / window** | The 128 × 128 grid the model sees. "Window" also includes its frames. |
| **Variable** | One field the model represents, e.g. SSH or SST, listed in the config's `variables`. |
| **Model units** | Data shifted and scaled to have typical magnitudes of order one. See [Units and scaling](quickstart/units.md). |
| **Noise level σ** | The standard deviation of the Gaussian noise added to a sample, in model units. |
| **Denoiser** | The network: given a noisy sample and σ, it estimates the clean sample (its conditional mean, approximately). |
| **Prior** | The distribution of ocean states the model has learned. Sampling it without observations gives "prior samples". |
| **Posterior** | The distribution of ocean states consistent with both the prior and the observations. SDA samples it. |
| **OSSE** | Observing System Simulation Experiment: synthetic observations taken from a known simulated truth, so errors can be measured. |
| **Embedding** | A numerical representation of an input such as noise level or frame time, supplied to network layers. |
| **Guidance** | A correction to each sampling step that accounts for observations or changes the influence of conditioning inputs. |
| **Checkpoint** | A saved model and its configuration; training checkpoints also store the state needed to resume a run. |
| **EMA** | Exponential moving average of the weights during training; the saved weights used for sampling. |

## Notation

Physical time and diffusion noise level are distinct: `τ_i` denotes a
frame timestamp and `σ` denotes the standard deviation of added noise.
Some implementation descriptions use `x_t` for the noisy state `z_σ`;
there, `t` indexes the diffusion process, not physical time.

| Symbol | Meaning |
|---|---|
| `u` | State in physical units (m, °C). |
| `x = N(u)` | State in model units (see [Units and scaling](quickstart/units.md)). |
| `τ_i` | Physical timestamp of frame `i`. |
| `σ` | Diffusion noise standard deviation, in model units. |
| `ε ~ N(0, I)` | Standard Gaussian noise. |
| `z_σ = x + σ ε` | Noisy normalized state (the code calls it `x_t`). |
| `F_θ` | The raw U-Net (`SongUNet`). |
| `D_θ` | The denoiser obtained by scaling the U-Net inputs and outputs (`EDMPrecond`; page 1). |
| `c` | Additional network inputs: frame times, calendar information, and, for conditional models, observation maps. |
| `y = h(x) + e` | Observation model; `h` is linear here (`A x`), units are declared with it. |
| `R` | Observation-error covariance, `diag(σ_y,j²)` for independent errors (`σ_y² I` when all are equal). |
| `J_θ` | Jacobian `∂D_θ/∂z_σ` of the denoiser with respect to its noisy input. |

**Shapes.** Public functions use `(batch, variable, frame, row, col)`;
inside the network a frame's grid is flattened row by row to
`(batch, variable, frame, pixel)` with `pixel = row * W + col`. "Channel"
inside the network means a feature channel; the model's input and output
channels are its variables.

## Main references

- Ho, Jain & Abbeel (2020), *Denoising Diffusion Probabilistic Models*. The
  modern starting point.
- Song et al. (2021), *Score-Based Generative Modeling through Stochastic
  Differential Equations*. The continuous-noise view used here.
- Song, Meng & Ermon (2021), *Denoising Diffusion Implicit Models* (DDIM). The
  sampler.
- Karras et al. (2022), *Elucidating the Design Space of Diffusion-Based
  Generative Models* (EDM). The preconditioning and the SongUNet
  configuration.
- Karras et al. (2024), *Analyzing and Improving the Training Dynamics of
  Diffusion Models* (EDM2). A newer network, see page 5.
- Rozet & Louppe (2023), *Score-based Data Assimilation*. SDA and its
  covariance heuristic.
- Rozet et al. (2024), *Learning Diffusion Priors from Observations by
  Expectation Maximization*. Moment Matching Posterior Sampling (MMPS), used
  here.
- Chung et al. (2023), *Diffusion Posterior Sampling for General Noisy Inverse
  Problems* (DPS). The simplest guidance method; a useful contrast.
- Ronneberger, Fischer & Brox (2015), *U-Net*. The network shape.
- Ho et al. (2022), *Video Diffusion Models*. Spatial layers plus temporal
  attention, the pattern used for frames here.

## Repository structure

<details>
<summary>Expand the directory and module reference</summary>

```
src/obottle_starter/
  core/                shared by training and inference
    nn/                  SongUNet + EDMPrecond (from NVIDIA cBottle)
    config.py            variables list, config checks, legacy translation
    model.py             build_model, load_checkpoint, export_checkpoint
    obs_conditioning.py  observations as network inputs (conditional models)
    normalization.py     physical units <-> model units
    conditioning.py      frame times -> the network's time inputs
  training/            training a model
    data.py              PatchDataset (per-patch zarr), StructuredDataset, SyntheticDataset
    train.py             training loop (single GPU or DDP), start modes
    validation.py        held-out denoising loss by variable and noise level
    obs_simulation.py    synthetic observations for conditional models
  inference/           reconstructing fields with SDA
    assimilate.py        one call from observations to a posterior ensemble
    osse.py              OSSE tiles, windows, random masks
    metrics.py           errors, spread/skill, ranks, coverage, spectra, coherence
    general.py           assimilate_general: general linear observations, physical units (opt-in)
    sda/                 MMPS, Rozet and CFG guidance, Krylov solvers, DDIM, observation operators
                         (masks: observations.py; points, averages, footprints, covariances: operators.py)
  cli.py               config loading for the scripts
data/                  SST climatology used by the released models' normalization
configs/               train_base.yaml (every training option) + experiment presets; sda.yaml
scripts/               train.py, run_sda.py; maintainer/ prepares model and data downloads
docs/                  background: network, training, data, SDA, extensions, conditional models, literature
examples/              notebooks 01-05 (pretrained SDA, small training, general observations,
                         conditional models and CFG, sliding windows)
  figures/             README figures and the script that draws them
tests/
```

</details>

Model development belongs in `core/` and `training/`; assimilation belongs
in `inference/`. `core/` does not import either module. Main functions such
as `load_checkpoint` and `assimilate` are also available from
`obottle_starter`.
