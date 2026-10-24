# Background documentation

These pages explain the ideas behind the code: what the network is, how it
is trained, where its data come from, and how data assimilation uses it.
They assume you know some Python and some ocean dynamics, but not diffusion
models.

| Page | Read it to learn |
|---|---|
| [1. The network](01_network_architecture.md) | What a diffusion model is, the U-Net inside it, and how it handles time |
| [2. Training](02_training.md) | The training objective, the noise schedule, and the settings in `configs/train_base.yaml` |
| [3. Data loading](03_data_loading.md) | How training samples are cut from llc4320: the tile method used here and the native-grid method used in newer work |
| [4. Inference with SDA](04_inference_sda.md) | How observations steer the model, what MMPS does, and when to use GMRES, CG or Rozet & Louppe's (2023) heuristic |
| [5. Extensions](05_extensions.md) | Options that are off, untested or not yet implemented, with what is known about each |
| [6. Conditional models](06_conditional_models.md) | Feeding observations to the network as inputs (M_3, M_4): experimental options to train and sample them |
| [7. Literature](07_literature.md) | The papers behind the methods, with links and why each matters here |

The README in the repository root covers installation and commands.

## Reading order

New to diffusion models? Read pages 1, 2 and 4 in order. To run the
pretrained models, page 4 is enough. To train your own, read pages 2 and 3.
Page 7 lists the papers; the key references below are the shortest path.

## A short glossary

| Term | Meaning here |
|---|---|
| **Channel** | One variable on the grid, e.g. SSH or SST (the network's input and output channels). |
| **Frame** | One snapshot in time. A sample is several frames of the same patch. |
| **Patch / window** | The 128 × 128 grid the model sees. "Window" also includes its frames. |
| **Variable** | One field the model represents, e.g. SSH or SST, listed in the config's `variables`. |
| **Model units** | Data after normalization, of order one. See the root README, "Units and normalization". |
| **Noise level σ** | The standard deviation of the Gaussian noise added to a sample, in model units. |
| **Denoiser** | The network: given a noisy sample and σ, it estimates the clean sample (its conditional mean, approximately). |
| **Prior** | The distribution of ocean states the model has learned. Sampling it without observations gives "prior samples". |
| **Posterior** | The distribution of ocean states consistent with both the prior and the observations. SDA samples it. |
| **OSSE** | Observing System Simulation Experiment: synthetic observations taken from a known simulated truth, so errors can be measured. |
| **EMA** | Exponential moving average of the weights during training; the saved weights used for sampling. |

## Notation

The pages use one set of symbols. Physical time and diffusion "time" are
easy to confuse in a model of trajectories, so frame times are `τ_i` and
the diffusion noise level is `σ`.

| Symbol | Meaning |
|---|---|
| `u` | State in physical units (m, °C). |
| `x = N(u)` | State in model units (the normalization, root README). |
| `τ_i` | Physical timestamp of frame `i`. |
| `σ` | Diffusion noise standard deviation, in model units. |
| `ε ~ N(0, I)` | Standard Gaussian noise. |
| `z_σ = x + σ ε` | Noisy normalized state (the code calls it `x_t`). |
| `F_θ` | The raw U-Net (`SongUNet`). |
| `D_θ` | The EDM-preconditioned denoiser built around it (`EDMPrecond`). |
| `c` | Context the network sees besides `z_σ`: frame times, calendar, and, for conditional models, observation maps. |
| `y = h(x) + e` | Observation model; `h` is linear here (`A x`), units are declared with it. |
| `R` | Observation-error covariance, `diag(σ_y,j²)` for independent errors (`σ_y² I` when all are equal). |
| `J_θ` | Jacobian `∂D_θ/∂z_σ` of the denoiser with respect to its noisy input. |

**Shapes.** Public functions use `(batch, variable, frame, row, col)`;
inside the network a frame's grid is flattened row by row to
`(batch, variable, frame, pixel)` with `pixel = row * W + col`. "Channel"
inside the network means a feature channel; the model's input and output
channels are its variables.

## Key references

- Ho, Jain & Abbeel (2020), *Denoising Diffusion Probabilistic Models*. The modern starting point.
- Song et al. (2021), *Score-Based Generative Modeling through Stochastic Differential Equations*. The continuous-noise view used here.
- Song, Meng & Ermon (2021), *Denoising Diffusion Implicit Models* (DDIM). The sampler.
- Karras et al. (2022), *Elucidating the Design Space of Diffusion-Based Generative Models* (EDM). The preconditioning and the SongUNet configuration.
- Karras et al. (2024), *Analyzing and Improving the Training Dynamics of Diffusion Models* (EDM2). A newer network, see page 5.
- Rozet & Louppe (2023), *Score-based Data Assimilation*. SDA and its covariance heuristic.
- Rozet et al. (2024), *Learning Diffusion Priors from Observations by Expectation Maximization*. Moment Matching Posterior Sampling (MMPS), used here.
- Chung et al. (2023), *Diffusion Posterior Sampling for General Noisy Inverse Problems* (DPS). The simplest guidance method; a useful contrast.
- Ronneberger, Fischer & Brox (2015), *U-Net*. The network shape.
- Ho et al. (2022), *Video Diffusion Models*. Spatial layers plus temporal attention, the pattern used for frames here.
