"""Training and evaluation loop functions."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from quantforge.training.metrics import compute_metrics

if TYPE_CHECKING:
    from torch.cuda.amp import GradScaler
    from torch.utils.data import DataLoader

    from quantforge.config.schema import TrainingConfig

logger = logging.getLogger("quantforge")


def train_one_epoch(
    model: nn.Module,
    loader: "DataLoader",
    optimizer: torch.optim.Optimizer,
    scheduler,
    loss_fn: nn.Module,
    device: str,
    cfg: "TrainingConfig",
    epoch: int,
    scaler: "GradScaler | None" = None,
    log_interval: int = 50,
) -> dict[str, float]:
    """Run one epoch of training.

    Returns:
        dict with 'loss' and 'lr'.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.autocast(device_type=device.split(":")[0]):
                output = model(images)
                loss = loss_fn(output, labels)
            scaler.scale(loss).backward()
            if cfg.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(images)
            loss = loss_fn(output, labels)
            loss.backward()
            if cfg.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        n_batches += 1

        if (batch_idx + 1) % log_interval == 0:
            logger.info(
                "  Epoch %d [%d/%d]  loss=%.4f",
                epoch + 1,
                batch_idx + 1,
                len(loader),
                loss.item(),
            )

    avg_loss = total_loss / max(n_batches, 1)
    current_lr = optimizer.param_groups[0]["lr"]
    return {"loss": avg_loss, "lr": current_lr}


@torch.no_grad()
def eval_one_epoch(
    model: nn.Module,
    loader: "DataLoader",
    loss_fn: nn.Module,
    device: str,
) -> dict[str, float]:
    """Run one epoch of evaluation.

    Returns:
        dict with 'loss', 'top1', 'top5'.
    """
    model.eval()
    total_loss = 0.0
    top1_sum = 0.0
    top5_sum = 0.0
    n_batches = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        output = model(images)
        loss = loss_fn(output, labels)

        metrics = compute_metrics(output, labels)
        total_loss += loss.item()
        top1_sum += metrics["top1"]
        top5_sum += metrics["top5"]
        n_batches += 1

    n = max(n_batches, 1)
    return {
        "loss": total_loss / n,
        "top1": top1_sum / n,
        "top5": top5_sum / n,
    }
