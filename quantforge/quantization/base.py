"""Quantization strategy base class."""

from __future__ import annotations

import torch.nn as nn


class QuantizationStrategy:
    """Base class for quantization strategies."""

    def prepare_model(self, model: nn.Module) -> nn.Module:
        """Prepare model for quantization (called before training)."""
        return model

    def on_train_start(self, model: nn.Module) -> None:
        """Called at the start of training."""

    def on_epoch_end(self, model: nn.Module, epoch: int) -> None:
        """Called at the end of each training epoch."""

    def convert_model(self, model: nn.Module) -> nn.Module:
        """Convert model after training (e.g. QAT -> quantized)."""
        return model


class NoQuantizationStrategy(QuantizationStrategy):
    """Pass-through strategy: no quantization."""
