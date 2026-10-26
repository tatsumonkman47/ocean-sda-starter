# 7. Literature

The papers behind this repository, what each contributes, and where it
shows up in the code. Grouped by topic; within a group, roughly in reading
order.

## Diffusion and score-based models

**Sohl-Dickstein, Weiss, Maheswaranathan & Ganguli (2015). *Deep
Unsupervised Learning using Nonequilibrium Thermodynamics.* ICML.**
[arXiv:1503.03585](https://arxiv.org/abs/1503.03585)
The original idea: learn to reverse a process that gradually destroys data
with noise. Historical background.

**Song & Ermon (2019). *Generative Modeling by Estimating Gradients of the
Data Distribution.* NeurIPS.** [arXiv:1907.05600](https://arxiv.org/abs/1907.05600)
Learns the score (gradient of the log-density) at many noise levels and
samples with Langevin dynamics. The "score-based" half of the field.

**Ho, Jain & Abbeel (2020). *Denoising Diffusion Probabilistic Models.*
NeurIPS.** [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
The paper that made diffusion models work well: a simple denoising
objective and a U-Net. The "diffusion" half of the field.

**Song, Sohl-Dickstein, Kingma, Kumar, Ermon & Poole (2021). *Score-Based
Generative Modeling through Stochastic Differential Equations.* ICLR.**
[arXiv:2011.13456](https://arxiv.org/abs/2011.13456)
Unifies the two views: noising is a stochastic differential equation, and
sampling solves its reverse. Introduces the variance-exploding (VE) and
variance-preserving (VP) processes and the DDPM++ / NCSN++ networks. This
repository uses the VE process and a DDPM++-style network (`SongUNet`).
Read this for the theory behind page 1.

**Song, Meng & Ermon (2021). *Denoising Diffusion Implicit Models.*
ICLR.** [arXiv:2010.02502](https://arxiv.org/abs/2010.02502)
DDIM: a deterministic sampler that needs far fewer steps. All sampling
here is DDIM (`inference/sda/sampling.py`).

**Efron (2011). *Tweedie's Formula and Selection Bias.* Journal of the
American Statistical Association 106(496).**
[doi:10.1198/jasa.2011.tm11181](https://doi.org/10.1198/jasa.2011.tm11181)
Tweedie's identity: for Gaussian corruption, the conditional mean
`E[x | x_t]` is the noisy input plus `σ²` times the score, and its
derivative is the posterior covariance. (That a squared-error denoiser
estimates `E[x | x_t]` is a separate fact: it is the minimizer of the
loss.) MMPS relies on both, with a learned denoiser in place of the exact
mean.

**Karras, Aittala, Aila & Laine (2022). *Elucidating the Design Space of
Diffusion-Based Generative Models.* NeurIPS.**
[arXiv:2206.00364](https://arxiv.org/abs/2206.00364)
"EDM". Separates design choices (noise schedule, network preconditioning,
loss weighting, sampler) and gives clean defaults. The preconditioning in
`EDMPrecond`, the loss weighting and the `SongUNet` configuration come from
here (page 1).

**Karras, Aittala, Lehtinen, Hellsten, Aila & Laine (2024). *Analyzing and
Improving the Training Dynamics of Diffusion Models.* CVPR.**
[arXiv:2312.02696](https://arxiv.org/abs/2312.02696)
"EDM2". Redesigns the network so activations and weights keep controlled
magnitudes, and learns the loss weighting per noise level. The parent
project's newer ocean models use an EDM2 adaptation; it may be added here
later (page 5).

**Karras, Aittala, Kynkäänniemi, Lehtinen, Aila & Laine (2024). *Guiding a
Diffusion Model with a Bad Version of Itself.* NeurIPS.**
[arXiv:2406.02507](https://arxiv.org/abs/2406.02507)
"Autoguidance": steer a model away from a weaker version of itself. Not
used here; the parent project has an untested implementation.

## Network building blocks

**Ronneberger, Fischer & Brox (2015). *U-Net: Convolutional Networks for
Biomedical Image Segmentation.* MICCAI.**
[arXiv:1505.04597](https://arxiv.org/abs/1505.04597)
The encoder-decoder with skip connections that every diffusion denoiser
here is built on.

**Wu & He (2018). *Group Normalization.* ECCV.**
[arXiv:1803.08494](https://arxiv.org/abs/1803.08494)
The normalization in each residual block, and the "GN" in AdaGN.

**Dhariwal & Nichol (2021). *Diffusion Models Beat GANs on Image
Synthesis.* NeurIPS.** [arXiv:2105.05233](https://arxiv.org/abs/2105.05233)
Introduces adaptive group normalization (AdaGN): the noise level scales
and shifts normalized features. Background for the per-frame AdaGN
discussion on pages 1 and 5.

**Ho, Salimans, Gritsenko, Chan, Norouzi & Fleet (2022). *Video Diffusion
Models.* NeurIPS.** [arXiv:2204.03458](https://arxiv.org/abs/2204.03458)
Spatial layers per frame plus attention across frames. The pattern this
repository's network uses for multi-frame ocean windows.

**Brenowitz et al. (2025). *Climate in a Bottle: Towards a Generative
Foundation Model for the Kilometer-Scale Global Atmosphere.*
[arXiv:2505.06474](https://arxiv.org/abs/2505.06474)**
NVIDIA's cBottle, a diffusion model of the global atmosphere. The network
code in `core/nn/` is a trimmed, ocean-adapted copy of cBottle's.

## Guiding a prior with observations (data assimilation)

**Meng et al. (2022). *SDEdit: Guided Image Synthesis and Editing with
Stochastic Differential Equations.* ICLR.**
[arXiv:2108.01073](https://arxiv.org/abs/2108.01073)
Start sampling from a noised version of a rough input instead of pure
noise. Suresh Babu et al.'s M_1; it produced unphysical fields there. Not
implemented here.

**Lugmayr et al. (2022). *RePaint: Inpainting using Denoising Diffusion
Probabilistic Models.* CVPR.** [arXiv:2201.09865](https://arxiv.org/abs/2201.09865)
Fill in missing regions by repeatedly replacing known pixels during
sampling. A simple inpainting baseline; not implemented here.

**Chung, Kim, McCann, Klasky & Ye (2023). *Diffusion Posterior Sampling for
General Noisy Inverse Problems.* ICLR.**
[arXiv:2209.14687](https://arxiv.org/abs/2209.14687)
DPS: add the gradient of the observation misfit, evaluated at the
denoiser's estimate, to each sampling step. Simple and widely used, but it
ignores the estimate's uncertainty and needs a hand-tuned step size.
Suresh Babu et al.'s M_2 is a DPS variant. A useful contrast to MMPS (page
4); not implemented here as such.

**Song, Vahdat, Mardani & Kautz (2023). *Pseudoinverse-Guided Diffusion
Models for Inverse Problems.* ICLR.**
[OpenReview](https://openreview.net/forum?id=9_gsMA8MRKQ)
ΠGDM: guidance through a pseudo-inverse of the observation operator, with
a variance term like Rozet & Louppe's. Background.

**Rozet & Louppe (2023). *Score-based Data Assimilation.* NeurIPS.**
[arXiv:2306.10574](https://arxiv.org/abs/2306.10574)
The origin of "SDA". Trains a prior on short trajectory windows and
assembles long trajectories from them, then conditions on observations at
inference only. Approximates the likelihood with variance
`σ_y² + γ σ_t²`, which is the `rozet` guidance option here
(`inference/sda/rozet.py`). The window-assembly idea underlies the parent
project's windowed SDA.

**Finzi, Boral, Wilson, Sha & Zepeda-Núñez (2023). *User-defined Event
Sampling and Uncertainty Quantification in Diffusion Models for Physical
Dynamical Systems.* ICML.**
[arXiv:2306.07526](https://arxiv.org/abs/2306.07526)
Conditions a diffusion model of chaotic dynamics on observations or
user-defined events with an approximation of the conditional score that
uses the covariance of the denoised estimate (from the denoiser's
Jacobian, by Tweedie's formula), and shows it converges to the true
conditional as the noise level falls. An early form of the second-order
likelihood that MMPS uses.

**Boys, Girolami, Pidstrigach, Reich, Mosca & Akyildiz (2024). *Tweedie
Moment Projected Diffusions for Inverse Problems.* TMLR.**
[arXiv:2310.06721](https://arxiv.org/abs/2310.06721)
TMPD: approximates each reverse step's conditional with a Gaussian whose
mean and covariance both come from Tweedie's formula, instead of DPS's
point estimate plus heuristics. Removes the hand-tuned step size, is
stable across noise levels, and comes with a guarantee for Gaussian
targets. Closely related to MMPS, which solves the same kind of system
iteratively.

**Rozet, Andry, Lanusse & Louppe (2024). *Learning Diffusion Priors from
Observations by Expectation Maximization.* NeurIPS.**
[arXiv:2405.13712](https://arxiv.org/abs/2405.13712)
Introduces Moment Matching Posterior Sampling (MMPS): use the denoiser's
Jacobian as the covariance in the likelihood and solve the resulting linear
system with a few conjugate-gradient iterations. This is the default
guidance here (`inference/sda/mmps.py`). The paper's main topic, training a
prior from observations only (DiEM), is not used here.

**Ho & Salimans (2022). *Classifier-Free Diffusion Guidance.* arXiv; a
shorter version appeared at a NeurIPS 2021 workshop.**
[arXiv:2207.12598](https://arxiv.org/abs/2207.12598)
Train one model with and without its condition, and at sampling
extrapolate between the two. The `cfg` option for conditional models
(page 6, `inference/sda/cfg.py`).

**Suresh Babu, Sadam & Lermusiaux (2026). *Guided Unconditional and
Conditional Generative Models for Super-Resolution and Inference of
Quasi-Geostrophic Turbulence.* Journal of Advances in Modeling Earth
Systems 18, e2025MS005324.**
[doi:10.1029/2025MS005324](https://doi.org/10.1029/2025MS005324),
[arXiv:2507.00719](https://arxiv.org/abs/2507.00719)
Compares SDEdit (M_1), DPS (M_2), a conditional model (M_3) and classifier-
free guidance (M_4) for reconstructing 2-D turbulence from coarse or
gappy observations. The conditional models were most accurate and best
calibrated; guided unconditional models struggled to spread information
into unobserved regions. The motivation for page 6. Note they did not test
MMPS.

## Applications to geophysical data assimilation

**Rozet & Louppe (2023). *Score-based Data Assimilation for a Two-Layer
Quasi-Geostrophic Model.* NeurIPS 2023 workshop (Machine Learning and the
Physical Sciences).** [arXiv:2310.01853](https://arxiv.org/abs/2310.01853)
SDA on an ocean-like quasi-geostrophic model, with network changes that cut
memory and run time. The closest ocean precedent for this repository's
approach.

**Huang, Gianinazzi, Yu, Dueben & Hoefler (2024). *DiffDA: a Diffusion
Model for Weather-scale Data Assimilation.* ICML.**
[arXiv:2401.05932](https://arxiv.org/abs/2401.05932)
Global atmospheric analysis with a diffusion model built on GraphCast,
conditioned on sparse observations by guidance and inpainting.

**Manshausen et al. (2024). *Generative Data Assimilation of Sparse
Weather Station Observations at Kilometer Scales.*
[arXiv:2406.16947](https://arxiv.org/abs/2406.16947)**
Score-based data assimilation of weather-station data into
kilometer-scale fields, with a diffusion prior trained on a regional
analysis. A close atmospheric analogue to this work.

**Martin, Manucharyan & Klein (2025). *Generative Data Assimilation for
Surface Ocean State Estimation From Multi-Modal Satellite Observations.*
Journal of Advances in Modeling Earth Systems 17(8), e2025MS005063.**
[doi:10.1029/2025MS005063](https://doi.org/10.1029/2025MS005063)
GenDA: a score-based diffusion model trained on a simulation serves as a
Bayesian prior, and sparse multi-modal satellite observations are
assimilated at inference to jointly reconstruct surface ocean variables at
eddy-resolving scales, including currents that satellites do not observe.
It reconstructs only the surface rather than the full interior state that
dynamical DA needs. The same two-stage design as this repository: train a
prior on simulation, add the observations only when sampling.

**Martin, Manucharyan & Klein (2026). *Generative Data Assimilation
Highlights Fronts as Key Regulators of Ocean Energy Cascade.* arXiv.**
[arXiv:2608.14955](https://arxiv.org/abs/2608.14955)
Uses generative data assimilation of several satellite products to
reconstruct gap-free, kilometre-scale surface currents over the Agulhas
Current, then diagnoses the energy cascade from them: upscale transfer
from submesoscales to mesoscale eddies above about 10 km, downscale below,
both concentrated in fronts. An example of using the reconstructions for
science, not only for mapping.

**Monkman, Balwada, Jones, Martin & Smith (2026). *Spatiotemporal Mapping
of Sea Surface Height and Temperature Fields from Satellite Observations
Using Score-based Data Assimilation.* Accepted in Earth and Space Science;
preprint on ESS Open Archive.**
[doi:10.22541/essoar.15003398/v3](https://doi.org/10.22541/essoar.15003398/v3)
Score-based data assimilation for joint mapping of SSH and SST in space
and time from satellite observations: the application this repository's
joint SSH + SST prior and SDA are aimed at.

## Linear solvers (page 4)

**Hestenes & Stiefel (1952). *Methods of Conjugate Gradients for Solving
Linear Systems.* J. Res. Natl. Bur. Stand. 49(6).**
[doi:10.6028/jres.049.044](https://doi.org/10.6028/jres.049.044) — CG.

**Paige & Saunders (1975). *Solution of Sparse Indefinite Systems of
Linear Equations.* SIAM J. Numer. Anal. 12(4).**
[doi:10.1137/0712047](https://doi.org/10.1137/0712047) — MINRES.

**Saad & Schultz (1986). *GMRES: A Generalized Minimal Residual Algorithm
for Solving Nonsymmetric Linear Systems.* SIAM J. Sci. Stat. Comput. 7(3).**
[doi:10.1137/0907058](https://doi.org/10.1137/0907058) — GMRES, the default.

## Ocean data and observations

**Rocha, Chereskin, Gille & Menemenlis (2016). *Mesoscale to Submesoscale
Wavenumber Spectra in Drake Passage.* Journal of Physical Oceanography
46(2).** [doi:10.1175/JPO-D-15-0087.1](https://doi.org/10.1175/JPO-D-15-0087.1)
A widely cited description of the MITgcm llc4320 simulation, the source of
all training data and OSSE truth here (page 3).

**Morrow et al. (2019). *Global Observations of Fine-Scale Ocean Surface
Topography With the Surface Water and Ocean Topography (SWOT) Mission.*
Frontiers in Marine Science 6.**
[doi:10.3389/fmars.2019.00232](https://doi.org/10.3389/fmars.2019.00232)
The SWOT mission whose swath coverage defines the SSH masks in the OSSE
tiles.

## Software

**azula** (Rozet et al.), [github.com/probabilists/azula](https://github.com/probabilists/azula).
A small PyTorch library of diffusion samplers and guidance methods. The
noise schedule, DDIM sampler, CG and GMRES here are vendored from it
(`inference/sda/_azula/`, MIT license), and `mmps.py` derives from its
`MMPSDenoiser`.
