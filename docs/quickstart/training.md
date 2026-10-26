# Train a model

Training teaches a denoiser to recover clean fields from noisy inputs. We
start with generated fields so you can check the workflow before preparing
a scientific dataset. For your own application, the data, normalization,
and evaluation should reflect the fields you want to reconstruct.

First [install the package](installation.md). The synthetic examples need no
download; the llc4320 example uses the optional
[training patches](downloads.md). All commands run
from the repository root. See [Units and scaling](units.md)
when configuring new variables, and [Training](../02_training.md) for
the denoising objective and its connection to
[Karras et al. (2022)](https://arxiv.org/abs/2206.00364).

## Example commands

```bash
python scripts/train.py config=configs/train_smoke_test.yaml       # 1-min code check (white noise)
python scripts/train.py config=configs/train_structured.yaml       # small model, generated fields
python scripts/train.py data_dir=/path/to/patches                  # joint SSH+SST, as released
python scripts/train.py config=configs/train_ssh_only.yaml data_dir=training_patches \
    time_range_start=0 time_range_end=2000                         # the bundled subset
python scripts/train.py config=configs/train_conditional.yaml data_dir=... \
    init_from=checkpoints/joint_ssh_sst_T5.pt                      # conditional model (experimental)
python scripts/train.py ... start=resume num_steps=400000          # continue a run
torchrun --nproc_per_node=4 scripts/train.py data_dir=...          # multi-GPU (batch_size per GPU)
```

## Configure a training run

Training configuration is loaded in this order, with later values taking
precedence:

1. `configs/train_base.yaml`: all options, defaults, and descriptions.
2. An optional experiment file, such as `config=configs/train_ssh_only.yaml`.
3. Command-line `key=value` overrides.

Copy an experiment file to create your own configuration. Unknown options
and invalid settings are rejected before data loading. Each run saves its
merged config to `runs/<experiment_name>/config.yaml` and in every
checkpoint. Checkpoints are written to
`runs/<experiment_name>/checkpoints/checkpoint_{latest,stepN}.pt`.

## Start, resume, or initialize from a checkpoint

Use `start` to control run behavior:

- `fresh` (default): starts a new run; rejects an existing run directory.
- `resume`: continues a run; permits changes only to runtime settings such
  as `num_steps`.
- `auto`: resumes if possible, for requeued jobs.

Use `init_from` to start a new run from another checkpoint's weights.
Trained checkpoints work directly with `run_sda.py` and `assimilate`.
See [Training](../02_training.md) for details, notebook 02 for a small
training example, and notebook 04 for conditional training and CFG.

## Training cost

Training throughput on one L40S at the released model size: about 9.5
samples/s for 5 frames (batch 32) and 4.3 samples/s for 11 frames (batch 8).
The released joint model saw about 41 M samples (315k steps at batch 130),
which would take weeks on one L40S. To explore, use smaller models
(`model_channels`, `time_length`) or multiple GPUs.

## Training data format

```
{data_dir}/{field}/{patch_id}.zarr     one store per variable (its `field`) and patch
```

Each Zarr store must contain:

- A data variable with dimensions `(time, row, col)`, with rows ordered
  south to north and columns west to east. Select it with `source_variable`,
  or provide a store containing only one data variable.
- An hourly `time` coordinate and 2-D `latitude` and `longitude` coordinates.
- Finite values; training stops at the first NaN.

All fields must share the same patch IDs and times. Training samples random
patches and windows of `time_length` frames, with frame offsets controlled
by `time_sampling_mode`. To add a variable, add its field directory and an
entry in `variables`.

---

[Documentation index](../README.md) · [Project README](../../README.md)
