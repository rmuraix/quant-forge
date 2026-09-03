"""TorchAO Post-Training Quantization strategy."""

from __future__ import annotations

import logging

import torch.nn as nn

from quantforge.quantization.base import QuantizationStrategy
from quantforge.utils.errors import QuantizationError

logger = logging.getLogger("quantforge")


class TorchAOPTQStrategy(QuantizationStrategy):
    """PTQ using torchao (applied after training)."""

    def prepare_model(self, model: nn.Module) -> nn.Module:
        """PTQ does not modify the model before training."""
        return model

    def convert_model(self, model: nn.Module) -> nn.Module:
        """Apply post-training quantization."""
        try:
            import importlib.util

            if importlib.util.find_spec("torchao") is None:
                raise ImportError("torchao not installed")

            try:
                from torchao.quantization import Int8WeightOnlyConfig, quantize_

                logger.info(
                    "Applying torchao int8 weight-only PTQ with Int8WeightOnlyConfig..."
                )
                quantize_(model, Int8WeightOnlyConfig(version=2))
                return model
            except (ImportError, Exception) as e:
                logger.warning(
                    "torchao PTQ (Int8WeightOnlyConfig) failed: %s. Trying legacy fallback.",
                    e,
                )

            try:
                from torchao.quantization import int8_weight_only, quantize_  # type: ignore[attr-defined]

                logger.info("Applying torchao PTQ with int8_weight_only (legacy)...")
                quantize_(model, int8_weight_only())
                return model
            except (ImportError, Exception) as e:
                logger.warning(
                    "torchao PTQ fallback failed: %s. Returning original model.", e
                )
                return model

        except ImportError as e:
            raise QuantizationError(
                "torchao is not installed. Install it with: pip install torchao\n"
                "Or: uv pip install quantforge[quantize]"
            ) from e
