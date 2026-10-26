# Reconstruct fields from observations

A trained diffusion model supplies plausible fields; observations guide those
fields toward measurements at sampling time. This lets us reuse one model
with different observation patterns. Here we use Moment Matching Posterior
Sampling ([Rozet et al., 2024](https://arxiv.org/abs/2405.13712)).

First [install the package](installation.md) and
[download the models and example data](downloads.md).
The examples below need a GPU and run from the repository root. Read
[Units and scaling](units.md) before adapting them to new variables.
For the reasoning behind the method, see the
[conceptual introduction](../introduction.md).

## Command line

Run from the repository root. Defaults are in `configs/sda.yaml`:

```bash
python scripts/run_sda.py ckpt=checkpoints/joint_ssh_sst_T5.pt \
    tile=osse_tiles/MidAtlantic_35N_50W.nc output=out/mid_atlantic.nc

# Append these overrides to the command above (replace the leading ...).
... observations.observe='[SSH]'                 # observe SSH only
... observations.mode=random                     # Bernoulli masks instead of instrument coverage
... observations.mode=none                       # prior samples
... observations.sigma_y.SSH=0.01 mmps.iterations=8 sampler.steps=128
... guidance.method=rozet rozet.gamma=0.1          # Rozet & Louppe (2023) heuristic
... window.t_start=6 window.origin='[50,180]'     # another window of the tile
```

**Output and diagnostics.** The output netCDF contains `samples` with dimensions
`(sample, channel, time, row, col)`, plus `truth`, `mask`, and `observations`
(NaN outside the mask). Fields use physical units and include latitude and
longitude. The script reports reconstruction errors, ensemble spread,
interval coverage, and observation misfit. Invalid options are rejected
before inference starts.

## Python

```python
import obottle_starter as obs
from obottle_starter.inference.osse import load_window

net, cfg = obs.load_checkpoint("checkpoints/joint_ssh_sst_T5.pt", "cuda")
names = obs.channel_names(cfg)                       # ['SSH', 'SST']
win = load_window("osse_tiles/MidAtlantic_35N_50W.nc", names, cfg.time_length)

post = obs.assimilate(
    net, cfg, times=win.times, center_lon=win.center_lon,
    truth=win.truth,                  # {channel: (T, H, W)} in m / degC
    masks=win.masks,                  # {channel: (T, H, W) bool}
    sigma_y={"SSH": 0.005, "SST": 0.1},
    n_members=8, steps=64, mmps_iterations=4,
)
post.samples.mean("sample")           # xarray, physical units
```

**Using real observations:** pass `y={channel: (T, H, W) values}` instead of
`truth`. Only values on `masks` are used. `examples/01_prior_and_sda.ipynb`
walks through generated fields, observation-based reconstructions, errors,
and spectra.

## Settings

| Option | Meaning |
|---|---|
| `sigma_y` | Required observation-error standard deviation (m, °C). Smaller values enforce closer agreement with observations. |
| `instrument_noise` | noise added to the synthetic observations (default: `sigma_y`) |
| `steps` | DDIM steps from σ = 200 down to 2e-4 |
| `mmps_iterations` | Krylov iterations per step, each requiring a network backward pass. Cost grows roughly linearly. |
| `mmps_solver` | `gmres` (default), `cg`, `minres` |
| `guidance`, `rozet_gamma` | `mmps` (default) or `rozet`: a fixed uncertainty `γ σ_t²` instead of the network's, no linear solve, one backward pass per step |
| `guidance="conditional"` / `"cfg"`, `cfg_weight` | for conditional models only (experimental; `docs/06_conditional_models.md`) |
| `n_members`, `n_batch` | ensemble size; members per GPU batch |

Python defaults match `configs/sda.yaml`. Check result stability by increasing
`steps` or `mmps_iterations` until results stabilize.

## Expected runtime

Measured on one L40S with 64 steps and 4 iterations, 4 members per batch:
2.2 s per step for the joint 5-frame model (about 2.5 min per batch) and
5.2 s per step for the SSH-only 11-frame model.

---

[Documentation index](../README.md) · [Project README](../../README.md)
