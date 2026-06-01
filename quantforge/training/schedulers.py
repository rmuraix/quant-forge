"""Learning rate scheduler builders."""

from __future__ import annotations

import torch
import torch.optim.lr_scheduler as lr_sched

from quantforge.config.schema import TrainingConfig


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: TrainingConfig,
    steps_per_epoch: int,
) -> lr_sched._LRScheduler | None:
    """Build a learning rate scheduler.

    Args:
        optimizer: The optimizer to schedule.
        cfg: Training config with scheduler settings.
        steps_per_epoch: Number of batches per epoch.

    Returns:
        A scheduler or None if 'none' is configured.
    """
    name = cfg.scheduler.lower()
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = cfg.warmup_epochs * steps_per_epoch

    if name == "none":
        return None

    if name == "cosine":
        if warmup_steps > 0:
            return _WarmupCosineScheduler(optimizer, warmup_steps, total_steps)
        return lr_sched.CosineAnnealingLR(optimizer, T_max=total_steps)

    if name == "step":
        return lr_sched.StepLR(
            optimizer, step_size=steps_per_epoch * max(1, cfg.epochs // 3)
        )

    # Fallback: cosine
    return lr_sched.CosineAnnealingLR(optimizer, T_max=max(1, total_steps))


class _WarmupCosineScheduler(lr_sched._LRScheduler):
    """Linear warmup then cosine annealing."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        import math

        step = self.last_epoch
        if step < self.warmup_steps:
            scale = (step + 1) / max(1, self.warmup_steps)
        else:
            progress = (step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [base_lr * scale for base_lr in self.base_lrs]
