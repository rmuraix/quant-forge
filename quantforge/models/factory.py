"""timm model factory."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quantforge.utils.errors import ModelError

if TYPE_CHECKING:
    import torch.nn as nn

    from quantforge.config.schema import ModelConfig


class ModelFactory:
    """Creates timm models from config."""

    def __init__(self, cfg: "ModelConfig", num_classes: int) -> None:
        self.cfg = cfg
        self.num_classes = num_classes

    def create(self) -> "nn.Module":
        """Create and return a timm model.

        Returns:
            Initialized torch.nn.Module.
        """
        try:
            import timm
        except ImportError as e:
            raise ModelError(
                "timm is not installed. Install it with: pip install timm"
            ) from e

        try:
            model = timm.create_model(
                self.cfg.name,
                pretrained=self.cfg.pretrained,
                num_classes=self.num_classes,
                in_chans=self.cfg.in_chans,
            )
        except Exception as e:
            raise ModelError(
                f"Failed to create model '{self.cfg.name}': {e}\n\n"
                f"Try:\n  model:\n    name: resnet18"
            ) from e

        return model
