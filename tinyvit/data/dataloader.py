from __future__ import annotations

from typing import Iterable, Iterator

import torch
from torch.utils.data import DataLoader

from tinyvit.config import Config, DataConfig
from tinyvit.data.hf_dataset import load_hf_dataset_splits
from tinyvit.data.transforms import build_transforms


def build_dataloaders(config: Config) -> tuple[DataLoader, DataLoader | None]:
    """Create train/val DataLoaders from Hugging Face datasets with transforms."""

    data_cfg = config.data
    splits = load_hf_dataset_splits(data_cfg)

    train_dl = _prepare_loader(
        splits.train,
        data_cfg,
        train=True,
        shuffle=data_cfg.shuffle and not data_cfg.streaming,
    )
    val_dl = (
        _prepare_loader(
            splits.val,
            data_cfg,
            train=False,
            shuffle=False,
        )
        if splits.val is not None
        else None
    )

    return train_dl, val_dl


def _apply_transforms(dataset, data_cfg: DataConfig, train: bool):
    transform = build_transforms(image_size=data_cfg.image_size, train=train)

    def convert(sample):
        return {
            "pixel_values": transform(sample["image"]),
            "label": sample["label"],
        }

    if data_cfg.streaming:
        return dataset.with_transform(convert)

    return dataset.map(convert, remove_columns=[c for c in dataset.column_names])


def _prepare_loader(dataset, data_cfg: DataConfig, train: bool, shuffle: bool):
    transformed = _apply_transforms(dataset, data_cfg, train=train)

    # IterableDataset supports shuffle=False only; map-style supports both.
    loader = DataLoader(
        transformed,
        batch_size=data_cfg.batch_size,
        shuffle=shuffle if not data_cfg.streaming else False,
        num_workers=data_cfg.num_workers,
        pin_memory=True,
        collate_fn=_collate,
    )
    return loader


def _collate(batch: Iterable[dict]) -> dict:
    """Collate batch into tensors."""

    if isinstance(batch, Iterator):
        batch = list(batch)

    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    return {"pixel_values": pixel_values, "labels": labels}
