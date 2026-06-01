"""Inference preprocessing from checkpoint metadata."""

from __future__ import annotations


def build_inference_transforms(preprocess_meta: dict):
    """Build eval transforms from checkpoint preprocess metadata.

    Args:
        preprocess_meta: dict with image_size, mean, std, interpolation.

    Returns:
        torchvision.transforms.Compose
    """
    from torchvision import transforms

    image_size = preprocess_meta.get("image_size", 224)
    mean = preprocess_meta.get("mean", [0.485, 0.456, 0.406])
    std = preprocess_meta.get("std", [0.229, 0.224, 0.225])

    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.143)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
