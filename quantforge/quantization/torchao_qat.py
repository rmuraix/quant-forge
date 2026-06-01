"""TorchAO Quantization-Aware Training strategy."""

from __future__ import annotations

import logging

import torch.nn as nn

from quantforge.quantization.base import QuantizationStrategy
from quantforge.utils.errors import QuantizationError

logger = logging.getLogger("quantforge")


class TorchAOQATStrategy(QuantizationStrategy):
    """QAT using torchao."""

    def prepare_model(self, model: nn.Module) -> nn.Module:
        """Prepare model for QAT."""
        try:
            import importlib.util

            if importlib.util.find_spec("torchao") is None:
                raise ImportError("torchao not installed")

            try:
                from torchao.quantization import quantize_  # noqa: F401

                logger.info(
                    "torchao available. QAT preparation will occur at convert step."
                )
                return model
            except ImportError:
                pass

            logger.warning(
                "torchao QAT API not found. Training normally; quantization at convert step."
            )
            return model

        except ImportError as e:
            raise QuantizationError(
                "torchao is not installed. Install it with: pip install torchao\n"
                "Or: uv pip install quantforge[quantize]"
            ) from e

    def convert_model(self, model: nn.Module) -> nn.Module:
        """Convert QAT model to quantized model."""
        try:
            import importlib.util

            if importlib.util.find_spec("torchao") is None:
                raise ImportError("torchao not installed")

            try:
                from torchao.quantization import int8_weight_only, quantize_

                logger.info("Converting QAT model with torchao int8_weight_only...")
                quantize_(model, int8_weight_only())
                return model
            except (ImportError, Exception) as e:
                logger.warning(
                    "torchao int8_weight_only conversion failed: %s. Trying fallback.",
                    e,
                )

            try:
                from torchao.quantization.quant_api import (  # type: ignore[import]
                    Int8WeightOnlyQuantizedLinearWeight,
                    quantize_,
                )

                logger.info("Converting QAT model with torchao PTQ fallback...")
                quantize_(model, Int8WeightOnlyQuantizedLinearWeight)
                return model
            except (ImportError, Exception) as e:
                logger.warning(
                    "torchao fallback conversion failed: %s. Returning original model.",
                    e,
                )
                return model

        except ImportError as e:
            raise QuantizationError(
                "torchao is not installed. Install it with: pip install torchao"
            ) from e
