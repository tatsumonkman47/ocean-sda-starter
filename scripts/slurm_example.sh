#!/bin/bash
# Template Slurm job for one GPU. Edit the #SBATCH lines and the environment
# activation for your cluster, then:  sbatch scripts/slurm_example.sh
#SBATCH --job-name=obottle-sda
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out

set -euo pipefail
# source /path/to/your/env/bin/activate   # or: conda activate obottle
cd "${SLURM_SUBMIT_DIR}"

python scripts/run_sda.py \
    ckpt=checkpoints/joint_ssh_sst_T5.pt \
    tile=osse_tiles/MidAtlantic_35N_50W.nc \
    output=out/mid_atlantic_${SLURM_JOB_ID}.nc

# Training instead. start=auto resumes if the run directory has a checkpoint
# and starts fresh otherwise, so the same job can be resubmitted:
# python scripts/train.py config=configs/train_ssh_only.yaml data_dir=training_patches \
#     time_range_start=0 time_range_end=2000 start=auto
