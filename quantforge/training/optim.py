"""Optimizer builders."""

from __future__ import annotations

import torch
import torch.nn as nn

from quantforge.config.schema import TrainingConfig
from quantforge.utils.errors import ConfigError


def get_optimizer(model: nn.Module, cfg: TrainingConfig) -> torch.optim.Optimizer:
    """Build an optimizer from config.

    Args:
        model: The model whose parameters to optimize.
        cfg: Training config with optimizer/lr/weight_decay settings.

    Returns:
        Configured optimizer.
    """
    params = model.parameters()
    name = cfg.optimizer.lower()

    if name == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            params, lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay
        )
    raise ConfigError(
        f"Unknown optimizer '{cfg.optimizer}'. Supported: adamw, adam, sgd."
    )
