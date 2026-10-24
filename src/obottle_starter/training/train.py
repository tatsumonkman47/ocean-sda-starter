# SPDX-License-Identifier: Apache-2.0
"""A minimal EDM training loop for the SongUNet prior.

Single GPU by default; launch with ``torchrun --nproc_per_node=N`` for
data-parallel training (``batch_size`` is per GPU). Checkpoints hold
``model_state_dict``, ``ema_state_dict``, ``optimizer_state_dict``, ``step``
and the full ``config``, and load with
:func:`obottle_starter.core.model.load_checkpoint`.

How a run starts is explicit (``start`` in the config):

* ``fresh``: a new run. Fails if ``output_dir`` already holds checkpoints.
  ``init_from`` optionally copies another model's weights first.
* ``resume``: continue ``output_dir``'s run (or ``resume_from``). Fails if
  there is nothing to resume, or if a setting that defines what the run
  learns has changed (:data:`obottle_starter.core.config.RUNTIME_KEYS` lists
  the ones that may change).
* ``auto``: ``resume`` if a checkpoint exists, else ``fresh``; for jobs that
  are requeued.

``config.yaml`` in ``output_dir`` is the config the run started with; each
resume that changes a runtime setting appends to ``config_changes.yaml``.
"""

from __future__ import annotations

import copy
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from obottle_starter.core.config import (
    ConfigError,
    channel_names,
    describe_architecture,
    run_differences,
    upgrade,
    validate_training,
)
from obottle_starter.core.model import build_model, load_weights_padded
from obottle_starter.core.nn.loss import EDMLoss
from obottle_starter.core.nn.networks import TemporalAttention
from obottle_starter.core.normalization import Normalizer, file_sha256
from obottle_starter.core.obs_conditioning import is_conditional, prepend, settings
from obottle_starter.training.data import build_dataset, validation_samples
from obottle_starter.training.obs_simulation import training_condition
from obottle_starter.training.validation import (
    NET_KWARGS,
    denoising_loss_by_sigma,
    format_validation,
)


def learning_rate(step: int, cfg) -> float:
    """Linear warm-up from ``lr_min`` to ``lr`` over ``lr_rampup_steps``."""
    if cfg.lr_rampup_steps <= 0 or step >= cfg.lr_rampup_steps:
        return cfg.lr
    return cfg.lr_min + step / cfg.lr_rampup_steps * (cfg.lr - cfg.lr_min)


def _save(path: Path, step, net, ema, optimizer, cfg) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "step": step,
            "model_state_dict": net.state_dict(),
            "ema_state_dict": ema.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": OmegaConf.to_container(cfg, resolve=True),
        },
        tmp,
    )
    os.replace(tmp, path)


def _record_climatology_checksums(cfg) -> None:
    """Store each climatology's sha256 in ``cfg``, so checkpoints identify it."""
    for spec in cfg.variables:
        clim = (spec.get("normalization") or {}).get("climatology")
        if clim is not None and not clim.get("sha256"):
            clim.sha256 = file_sha256(Normalizer([spec]).climatology(0).path)


def _resume_source(cfg, latest: Path) -> Path | None:
    """The checkpoint to resume from, or ``None`` for a fresh run."""
    start = cfg.get("start", "fresh")
    if start == "fresh":
        if latest.exists():
            raise ConfigError(
                f"{latest.parent} already holds a run. Use start=resume to "
                "continue it, start=auto for requeued jobs, or a new "
                "output_dir / experiment_name for a new run"
            )
        return None
    path = Path(cfg.resume_from) if cfg.get("resume_from") else latest
    if path.exists():
        return path
    if start == "resume":
        raise ConfigError(f"start=resume, but there is no checkpoint at {path}")
    return None  # auto without a checkpoint: a fresh run


def _check_replicas(net, ema, world: int) -> None:
    """Fail if model or EMA weights differ between DDP processes."""
    with torch.no_grad():
        sums = torch.stack(
            [
                sum(p.double().sum() for p in module.parameters())
                for module in (net, ema)
            ]
        )
    gathered = [torch.zeros_like(sums) for _ in range(world)]
    dist.all_gather(gathered, sums)
    if not all(torch.equal(g, gathered[0]) for g in gathered):
        raise RuntimeError(
            "DDP replicas have diverged (model or EMA weights differ between "
            f"processes): {[g.tolist() for g in gathered]}"
        )


def train(cfg) -> Path:
    """Train until ``cfg.num_steps``; returns the output directory."""
    cfg = upgrade(cfg)
    warnings = validate_training(cfg)

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    rank = dist.get_rank() if distributed else 0
    world = dist.get_world_size() if distributed else 1
    if torch.cuda.is_available():
        local = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    else:
        device = torch.device("cpu")

    out_dir = Path(cfg.output_dir)
    ckpt_dir = out_dir / "checkpoints"
    latest = ckpt_dir / "checkpoint_latest.pt"
    resume = _resume_source(cfg, latest)
    _record_climatology_checksums(cfg)

    # Same initial weights on every process; then different data and noise.
    torch.manual_seed(int(cfg.seed))
    net = build_model(cfg, device)
    seed = int(cfg.seed) + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Temporal attention keeps a discrete positional table for calls without
    # frame times. Training always passes them, so the table gets no
    # gradient; freezing it keeps DDP from waiting for those gradients.
    for module in net.modules():
        if isinstance(module, TemporalAttention):
            module.positional_embedding.requires_grad_(False)
    trainable = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=cfg.lr)

    step, changes = 0, []
    if resume is not None:
        state = torch.load(resume, map_location=device, weights_only=False)
        saved = state["config"]
        diffs = run_differences(cfg, saved)
        if diffs:
            listed = "\n  ".join(diffs)
            raise ConfigError(
                f"Cannot resume {resume}: these settings differ from the run's "
                f"and would change what it learns:\n  {listed}\nStart a new "
                "run (new output_dir) instead; init_from=<that checkpoint> "
                "starts it from these weights."
            )
        for new, old in zip(cfg.variables, upgrade(saved).variables):
            a = (new.normalization.get("climatology") or {}).get("sha256")
            b = (old.get("normalization") or {}).get("climatology") or {}
            if a and b.get("sha256") and a != b.get("sha256"):
                raise ConfigError(
                    f"Cannot resume {resume}: the climatology file for "
                    f"{new.name} has changed since the run started (sha256)"
                )
        changes = run_differences(cfg, saved, ignore=())
        if cfg.get("init_from") and cfg.init_from != saved.get("init_from"):
            raise ConfigError(
                f"init_from={cfg.init_from} but the run being resumed started "
                f"from {saved.get('init_from')}; drop init_from or start a new run"
            )
        net.load_state_dict(state["model_state_dict"])
        ema = copy.deepcopy(net).eval().requires_grad_(False)
        ema.load_state_dict(state["ema_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        step = int(state["step"])
        message = f"Resumed from {resume} at step {step}"
    else:
        message = "Fresh run"
        if cfg.get("init_from"):
            # Start from another model's weights (e.g. a released prior),
            # with zero weights for any new observation inputs.
            state = torch.load(cfg.init_from, map_location=device, weights_only=False)
            weights = state.get("ema_state_dict", state.get("model_state_dict"))
            padded = load_weights_padded(net, weights, cfg, upgrade(state["config"]))
            pad = f", zero-padded for new observation inputs: {padded}"
            message = (
                f"Fresh run initialized from {cfg.init_from}: "
                f"{len(weights) - len(padded)} tensors copied unchanged"
                f"{pad if padded else ''}; none newly initialized"
            )
        ema = copy.deepcopy(net).eval().requires_grad_(False)

    # Validation passed: only now write run metadata.
    if rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        if resume is None:
            (out_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg))
        elif changes:
            with open(out_dir / "config_changes.yaml", "a") as f:
                f.write(OmegaConf.to_yaml({f"resume at step {step}": changes}))

    dataset = build_dataset(cfg)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=cfg.num_workers > 0,
        drop_last=True,
    )
    loss_fn = EDMLoss(
        distribution=cfg.get("noise_distribution", "log_uniform"),
        P_mean=cfg.get("p_mean", -1.2),
        P_std=cfg.get("p_std", 1.2),
        sigma_data=cfg.sigma_data,
        sigma_min=cfg.sigma_min,
        sigma_max=cfg.sigma_max,
    )

    model = net
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            net, device_ids=[device.index] if device.type == "cuda" else None
        )
    use_amp = device.type == "cuda"
    names = channel_names(cfg)
    normalizer = Normalizer.from_config(cfg)
    normalizer.source = "training config"
    if rank == 0:
        n_params = sum(p.numel() for p in net.parameters())
        print(message)
        for w in warnings:
            print(f"WARNING: {w}")
        print(describe_architecture(cfg))
        print(f"{n_params:,} parameters | {world} process(es) | device {device}")
        print(
            f"data: {cfg.data_source}"
            + (f" from {cfg.data_dir}" if cfg.data_source == "patches" else "")
            + f" | variables {names}"
        )
        print(normalizer.describe(sda=False))
        if is_conditional(cfg):
            s = settings(cfg)
            print(
                f"conditional model: observes {s['observe']}, "
                f"mask channel {s['mask_channel']}, training observations "
                f"{dict(cfg.conditioning.get('training_observations') or {})}"
            )
        horizon = cfg.batch_size * world / max(1e-12, 1.0 - cfg.ema_decay)
        print(
            f"EMA: decay {cfg.ema_decay} per step, about {horizon:,.0f} samples "
            f"({cfg.batch_size} x {world} per step)"
        )
    val = cfg.get("validation") or {}
    val_samples = validation_samples(cfg) if rank == 0 and val.get("every") else None
    obs_normalizer = normalizer if is_conditional(cfg) else None

    net.train()
    t0, seen = time.time(), 0
    batches = iter(loader)
    while step < cfg.num_steps:
        try:
            batch = next(batches)
        except StopIteration:
            batches = iter(loader)
            batch = next(batches)
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        kwargs = {k: batch[k] for k in NET_KWARGS if k in batch}
        if obs_normalizer is not None:
            obs = training_condition(cfg, batch["target"], obs_normalizer)
            kwargs["condition"] = prepend(obs, kwargs.get("condition"))

        def denoise(x, sigma):
            return model(x, sigma, **kwargs)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            loss = loss_fn(denoise, images=batch["target"]).total.mean()

        step += 1
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, cfg)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            for p_ema, p in zip(ema.parameters(), net.parameters()):
                p_ema.lerp_(p, 1.0 - cfg.ema_decay)
        seen += batch["target"].shape[0] * world

        if rank == 0 and step % cfg.log_every == 0:
            rate = seen / (time.time() - t0)
            print(
                f"step {step:7d} | loss {loss.item():.4f} | "
                f"lr {learning_rate(step, cfg):.2e} | {rate:.1f} samples/s",
                flush=True,
            )
        if val_samples is not None and step % val.every == 0:
            losses = denoising_loss_by_sigma(
                ema, val_samples, list(val.sigmas), cfg, normalizer=obs_normalizer
            )
            print(format_validation(step, losses, names), flush=True)
            with open(out_dir / "validation.csv", "a") as f:
                for sigma, values in losses.items():
                    f.write(f"{step},{sigma}," + ",".join(f"{v:.6g}" for v in values))
                    f.write("\n")
        if step % cfg.checkpoint_every == 0 or step == cfg.num_steps:
            if distributed:
                _check_replicas(net, ema, world)
            if rank == 0:
                _save(
                    ckpt_dir / f"checkpoint_step{step}.pt",
                    step,
                    net,
                    ema,
                    optimizer,
                    cfg,
                )
                _save(latest, step, net, ema, optimizer, cfg)

    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    return out_dir
