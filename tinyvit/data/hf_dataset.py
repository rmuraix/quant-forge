from __future__ import annotations

from typing import NamedTuple

from datasets import IterableDataset, IterableDatasetDict, load_dataset

from tinyvit.config import DataConfig


class DatasetSplits(NamedTuple):
    train: IterableDataset
    val: IterableDataset | None
    test: IterableDataset | None


def load_hf_dataset_splits(cfg: DataConfig) -> DatasetSplits:
    """Load Hugging Face dataset splits with optional streaming for efficiency."""

    common_kwargs = {
        "name": cfg.dataset_config_name,
        "cache_dir": str(cfg.cache_dir) if cfg.cache_dir else None,
        "streaming": cfg.streaming,
    }

    train_ds = load_dataset(cfg.dataset_name, split=cfg.train_split, **common_kwargs)
    val_ds = (
        load_dataset(cfg.dataset_name, split=cfg.val_split, **common_kwargs)
        if cfg.val_split
        else None
    )
    test_ds = (
        load_dataset(cfg.dataset_name, split=cfg.test_split, **common_kwargs)
        if cfg.test_split
        else None
    )

    # Cast to iterable datasets for uniform downstream handling
    train_iter = _to_iterable(train_ds)
    val_iter = _to_iterable(val_ds) if val_ds is not None else None
    test_iter = _to_iterable(test_ds) if test_ds is not None else None

    return DatasetSplits(train=train_iter, val=val_iter, test=test_iter)


def _to_iterable(dataset) -> IterableDataset:
    """Ensure dataset is iterable to support streaming and low-memory usage."""

    if isinstance(dataset, IterableDataset):
        return dataset

    if isinstance(dataset, IterableDatasetDict):
        # Pick first available split if dict provided unexpectedly
        return next(iter(dataset.values()))

    # Fallback: convert map-style dataset to iterable to reduce memory footprint
    return dataset.to_iterable_dataset()
