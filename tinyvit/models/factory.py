from __future__ import annotations

import timm
import torch.nn as nn

from tinyvit.config import ModelConfig


def build_timm_model(model_cfg: ModelConfig) -> nn.Module:
    """Create a timm model configured for transfer learning."""

    checkpoint = str(model_cfg.checkpoint_path) if model_cfg.checkpoint_path else None
    model = timm.create_model(
        model_cfg.model_name,
        pretrained=model_cfg.pretrained,
        num_classes=model_cfg.num_classes,
        drop_rate=model_cfg.drop_rate,
        drop_path_rate=model_cfg.drop_path_rate,
        checkpoint_path=checkpoint,
    )

    if model_cfg.freeze_backbone:
        _freeze_backbone_except_classifier(model)

    return model


def _freeze_backbone_except_classifier(model: nn.Module) -> None:
    """Freeze all parameters except the classifier head."""

    classifier = model.get_classifier()

    # Freeze everything by default
    for param in model.parameters():
        param.requires_grad = False

    trainable_params = 0

    if isinstance(classifier, nn.Module):
        trainable_params += _unfreeze_module(classifier)
    elif isinstance(classifier, (list, tuple, set)):
        for module in classifier:
            if isinstance(module, nn.Module):
                trainable_params += _unfreeze_module(module)

    # If classifier could not be determined, keep model trainable to avoid deadlock
    if trainable_params == 0:
        for param in model.parameters():
            param.requires_grad = True


def _unfreeze_module(module: nn.Module) -> int:
    """Unfreeze parameters of a module and return count."""

    count = 0
    for param in module.parameters():
        param.requires_grad = True
        count += param.numel()
    return count
