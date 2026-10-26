# Download models and example data

Download these assets to run the released ocean reconstruction examples.
The synthetic training examples do not need them. First follow
[installation](installation.md); after downloading, continue to
[reconstruction](inference.md).

The saved models (checkpoints) and simulated observation datasets (OSSE
tiles) are too large for the repository. They are
attached to the
[`data-v1` release](https://github.com/tatsumonkman47/ocean-sda-starter/releases/tag/data-v1).
From the repository root:

```bash
URL=https://github.com/tatsumonkman47/ocean-sda-starter/releases/download/data-v1
mkdir -p checkpoints osse_tiles
for f in joint_ssh_sst_T5.pt ssh_only_T11.pt; do curl -L -o checkpoints/$f $URL/$f; done
for f in MidAtlantic_35N_50W NorthAtlantic_48N_30W SubtropAtlantic_26N_44W; do
    curl -L -o osse_tiles/$f.nc $URL/$f.nc
done
# optional, 850 MB unpacked: real llc4320 SSH patches for training
curl -L -O $URL/training_patches.zip && unzip training_patches.zip && rm training_patches.zip
```

## Where to put the files

Place the downloaded directories in the repository root:

```
ocean-sda-starter/
  checkpoints/
    joint_ssh_sst_T5.pt      joint SSH + SST model, 5 frames      (95 MB)
    ssh_only_T11.pt          SSH-only model, 11 frames            (95 MB)
  osse_tiles/
    MidAtlantic_35N_50W.nc   336x336 cells (4 km), 24 frames, 12-hourly (16 MB each)
    NorthAtlantic_48N_30W.nc
    SubtropAtlantic_26N_44W.nc
  training_patches/          optional: 8 North Pacific SSH patches x 2000 hours (850 MB)
  src/ configs/ examples/ ...
```

The notebooks use `BUNDLE = Path("..")` to find these directories from
`examples/`. Git ignores all three directories. To store the downloads
elsewhere, create symbolic links in the repository root (for example,
`ln -s /path/to/bundle/checkpoints .`), or update `BUNDLE` in the notebooks
and pass full paths to the scripts.

## Verify the downloaded models

**These tests must be enabled explicitly.** To check that both checkpoints
reproduce the original training-code network, run:

```bash
OBOTTLE_CHECKPOINTS=checkpoints python -m pytest
```

Without this environment variable, the two checkpoint tests are skipped,
even if the models have been downloaded.

## Available models

| | `joint_ssh_sst_T5` | `ssh_only_T11` |
|---|---|---|
| channels | SSH, SST | SSH |
| frames per window | 5 | 11 |
| patch | 128 × 128 at 4 km (512 km) | same |
| training data | llc4320 North Pacific, 415 patches, Sep 2011 – Sep 2012, hourly | same |
| frame spacing in training | random, 1 h grid, total span ≤ 336 h | same |
| steps | 315k (batch 130) | 195k (batch 60) |
| parameters | 23.7 M | 23.7 M |

## Variables, units, and supported dates

- **SSH:** llc4320 sea-surface height with scales longer than 300 km removed,
  in metres, normalized by 0.051 m.
- **SST:** temperature in °C, with a North Pacific basin-mean daily climatology
  subtracted and the result divided by 5.5 °C.

**Unit conversion is automatic:** `assimilate`, the scripts, and the OSSE
helpers take and return physical units. The low-level classes in
`obottle_starter.inference.sda` use the units of their inputs, normally model
units. See [Units and scaling](units.md).

**Supported dates:** the released SST normalization looks up the climatology
by elapsed days from 2011-09-13. Dates outside **2011-09-13 through 2012-09-23**
raise an error unless you set `normalization.climatology.lookup=day_of_year`
or `normalization.climatology.out_of_range=clip`.

The OSSE tiles are in the Atlantic, so they also test transfer from the
Pacific training region.

---

[Documentation index](../README.md) · [Project README](../../README.md)
