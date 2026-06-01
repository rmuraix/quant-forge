"""Image transform builders."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quantforge.config.schema import AugmentConfig, PreprocessConfig

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def build_train_transforms(
    cfg_preprocess: "PreprocessConfig", cfg_augment: "AugmentConfig"
):
    """Build training transforms."""
    from torchvision import transforms

    mean = tuple(cfg_preprocess.mean) if cfg_preprocess.mean else IMAGENET_DEFAULT_MEAN
    std = tuple(cfg_preprocess.std) if cfg_preprocess.std else IMAGENET_DEFAULT_STD
    size = cfg_preprocess.image_size

    ops = []
    if cfg_augment.random_resized_crop:
        ops.append(transforms.RandomResizedCrop(size))
    else:
        ops.extend([transforms.Resize(size), transforms.CenterCrop(size)])

    if cfg_augment.horizontal_flip:
        ops.append(transforms.RandomHorizontalFlip())

    if cfg_augment.randaugment:
        ops.append(transforms.RandAugment())

    ops.append(transforms.ToTensor())
    ops.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(ops)


def build_eval_transforms(cfg_preprocess: "PreprocessConfig"):
    """Build evaluation/inference transforms."""
    from torchvision import transforms

    mean = tuple(cfg_preprocess.mean) if cfg_preprocess.mean else IMAGENET_DEFAULT_MEAN
    std = tuple(cfg_preprocess.std) if cfg_preprocess.std else IMAGENET_DEFAULT_STD
    size = cfg_preprocess.image_size

    return transforms.Compose(
        [
            transforms.Resize(int(size * 1.143)),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
