"""Loss function builders."""

from __future__ import annotations

import torch.nn as nn

from quantforge.config.schema import TrainingConfig


def get_loss_fn(cfg: TrainingConfig | None = None) -> nn.Module:
    """Return the loss function.

    Returns:
        CrossEntropyLoss for classification.
    """
    return nn.CrossEntropyLoss()
