"""Hugging Face dataset loading utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quantforge.utils.errors import DatasetColumnError, DatasetError

if TYPE_CHECKING:
    from quantforge.config.schema import DatasetConfig


def load_hf_dataset(cfg: "DatasetConfig"):
    """Load a Hugging Face dataset.

    Returns:
        datasets.DatasetDict with at least train and validation splits.
    """
    try:
        import datasets as ds
    except ImportError as e:
        raise DatasetError(
            "The 'datasets' package is not installed. Install it with: pip install datasets"
        ) from e

    kwargs: dict = {"path": cfg.name}
    if cfg.subset:
        kwargs["name"] = cfg.subset
    if cfg.cache_dir:
        kwargs["cache_dir"] = cfg.cache_dir

    try:
        dataset = ds.load_dataset(**kwargs)
    except Exception as e:
        raise DatasetError(f"Failed to load dataset '{cfg.name}': {e}") from e

    # Validate that required splits are present
    required_splits = [cfg.split.train]
    if cfg.split.validation:
        required_splits.append(cfg.split.validation)

    for split_name in required_splits:
        if split_name not in dataset:
            available = list(dataset.keys())
            raise DatasetError(
                f"Split '{split_name}' not found in dataset '{cfg.name}'.\n"
                f"Available splits: {available}\n\n"
                f"Try updating:\n  dataset:\n    split:\n      validation: {available[0] if available else 'test'}"
            )

    return dataset


def validate_columns(dataset_split, image_column: str, label_column: str) -> None:
    """Validate that required columns exist in the dataset split.

    Raises:
        DatasetColumnError with actionable message if a column is missing.
    """
    available = list(dataset_split.features.keys())
    for col, col_type in [
        (image_column, "image_column"),
        (label_column, "label_column"),
    ]:
        if col not in available:
            raise DatasetColumnError(
                f"Column '{col}' was not found in the dataset.\n"
                f"Available columns: {available}\n\n"
                f"Try:\n  dataset:\n    {col_type}: {available[0] if available else col}"
            )
