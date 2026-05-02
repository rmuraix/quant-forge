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
                from torchao.quantization import int8_weight_only, quantize_

                logger.info("Applying torchao int8 weight-only PTQ...")
                quantize_(model, int8_weight_only())
                return model
            except (ImportError, Exception) as e:
                logger.warning("torchao PTQ (int8_weight_only) failed: %s", e)

            try:
                from torchao.quantization.quant_api import (  # type: ignore[import]
                    Int8WeightOnlyQuantizedLinearWeight,
                    quantize_,
                )

                logger.info("Applying torchao PTQ (fallback)...")
                quantize_(model, Int8WeightOnlyQuantizedLinearWeight)
                return model
            except (ImportError, Exception) as e:
                logger.warning("torchao PTQ fallback failed: %s. Returning original model.", e)
                return model

        except ImportError as e:
            raise QuantizationError(
                "torchao is not installed. Install it with: pip install torchao\n"
                "Or: uv pip install quantforge[quantize]"
            ) from e
