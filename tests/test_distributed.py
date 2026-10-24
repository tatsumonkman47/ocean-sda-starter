# SPDX-License-Identifier: Apache-2.0
"""Two-process DDP training on CPU (gloo).

Runs several optimizer steps (DDP reduction problems, such as parameters
that never receive a gradient, surface on the step after the first), checks
inside ``train`` that model and EMA weights agree across processes at every
checkpoint, then resumes the run. CUDA/NCCL needs a separate GPU run.
"""

import os
import socket

import pytest
import torch
import torch.multiprocessing as mp

from obottle_starter.training.train import train

from conftest import tiny_config


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank, world, port, cfg):
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        WORLD_SIZE=str(world),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
    )
    torch.set_num_threads(1)
    train(cfg)


@pytest.mark.skipif(torch.cuda.is_available(), reason="CPU/gloo test")
def test_two_process_training_and_resume(tmp_path):
    cfg = tiny_config(
        output_dir=str(tmp_path / "ddp"), num_steps=4, checkpoint_every=2, batch_size=2
    )
    mp.spawn(_worker, args=(2, _free_port(), cfg), nprocs=2, join=True)
    latest = tmp_path / "ddp" / "checkpoints" / "checkpoint_latest.pt"
    assert torch.load(latest)["step"] == 4

    cfg.start, cfg.num_steps = "resume", 6
    mp.spawn(_worker, args=(2, _free_port(), cfg), nprocs=2, join=True)
    assert torch.load(latest)["step"] == 6
