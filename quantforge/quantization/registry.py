"""Quantization strategy registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quantforge.quantization.base import NoQuantizationStrategy, QuantizationStrategy
from quantforge.utils.errors import QuantizationError

if TYPE_CHECKING:
    from quantforge.config.schema import QuantizationConfig

QUANTIZATION_REGISTRY: dict[tuple[str, str], type[QuantizationStrategy]] = {
    ("none", "none"): NoQuantizationStrategy,
}


# Register torchao strategies lazily
def _register_torchao() -> None:
    from quantforge.quantization.torchao_qat import TorchAOQATStrategy
    from quantforge.quantization.torchao_ptq import TorchAOPTQStrategy

    QUANTIZATION_REGISTRY[("torchao", "qat")] = TorchAOQATStrategy
    QUANTIZATION_REGISTRY[("torchao", "ptq")] = TorchAOPTQStrategy


_register_torchao()


def get_quantizer(cfg: "QuantizationConfig") -> QuantizationStrategy:
    """Return a QuantizationStrategy for the given config.

    Args:
        cfg: Quantization config section.

    Returns:
        QuantizationStrategy instance.
    """
    key = (cfg.backend, cfg.mode)
    cls = QUANTIZATION_REGISTRY.get(key)
    if cls is None:
        available = list(QUANTIZATION_REGISTRY.keys())
        raise QuantizationError(
            f"No quantization strategy found for backend='{cfg.backend}', mode='{cfg.mode}'.\n"
            f"Available: {available}"
        )
    return cls()
