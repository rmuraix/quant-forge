"""HF Dataset DataModule: loads data, builds transforms, returns DataLoaders."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quantforge.data.hf_dataset import load_hf_dataset, validate_columns
from quantforge.data.labels import infer_label_mapping, infer_num_classes
from quantforge.data.transforms import build_eval_transforms, build_train_transforms
from quantforge.utils.errors import DatasetError

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from quantforge.config.schema import ExperimentConfig


class HFDataModule:
    """Data module for Hugging Face datasets."""

    def __init__(self, cfg: "ExperimentConfig") -> None:
        self.cfg = cfg
        self._dataset = None
        self._num_classes: int | None = None
        self._label_mapping: dict | None = None
        self._train_loader: "DataLoader | None" = None
        self._val_loader: "DataLoader | None" = None

    def setup(self) -> None:
        """Load dataset, validate columns, build transforms and loaders."""
        from torch.utils.data import DataLoader

        cfg_ds = self.cfg.dataset
        dataset = load_hf_dataset(cfg_ds)

        train_split = dataset[cfg_ds.split.train]
        val_split_name = cfg_ds.split.validation
        if val_split_name is None or val_split_name not in dataset:
            raise DatasetError(
                f"Validation split '{val_split_name}' not found.\n"
                "Please set dataset.split.validation in your config."
            )
        val_split = dataset[val_split_name]

        validate_columns(train_split, cfg_ds.image_column, cfg_ds.label_column)
        validate_columns(val_split, cfg_ds.image_column, cfg_ds.label_column)

        self._num_classes = infer_num_classes(train_split, cfg_ds.label_column)
        self._label_mapping = infer_label_mapping(train_split, cfg_ds.label_column)

        train_transform = build_train_transforms(self.cfg.preprocess, self.cfg.augment)
        eval_transform = build_eval_transforms(self.cfg.preprocess)

        def make_transform_fn(split_data, transform, img_col, lbl_col):
            def collate(batch):
                from PIL import Image

                import torch

                images, labels = [], []
                for item in batch:
                    img = item[img_col]
                    if not hasattr(img, "convert"):
                        img = Image.fromarray(img)
                    img = img.convert("RGB")
                    images.append(transform(img))
                    labels.append(item[lbl_col])
                return torch.stack(images), torch.tensor(labels, dtype=torch.long)

            return collate

        train_collate = make_transform_fn(
            train_split, train_transform, cfg_ds.image_column, cfg_ds.label_column
        )
        val_collate = make_transform_fn(
            val_split, eval_transform, cfg_ds.image_column, cfg_ds.label_column
        )

        self._train_loader = DataLoader(
            train_split,  # type: ignore[arg-type]
            batch_size=self.cfg.training.batch_size,
            shuffle=True,
            num_workers=self.cfg.training.num_workers,
            collate_fn=train_collate,
            pin_memory=True,
            drop_last=True,
        )
        self._val_loader = DataLoader(
            val_split,  # type: ignore[arg-type]
            batch_size=self.cfg.training.batch_size,
            shuffle=False,
            num_workers=self.cfg.training.num_workers,
            collate_fn=val_collate,
            pin_memory=True,
        )

    @property
    def num_classes(self) -> int:
        if self._num_classes is None:
            raise RuntimeError("DataModule not set up. Call setup() first.")
        return self._num_classes

    @property
    def label_mapping(self) -> dict:
        if self._label_mapping is None:
            raise RuntimeError("DataModule not set up. Call setup() first.")
        return self._label_mapping

    def train_dataloader(self) -> "DataLoader":
        if self._train_loader is None:
            raise RuntimeError("DataModule not set up. Call setup() first.")
        return self._train_loader

    def val_dataloader(self) -> "DataLoader":
        if self._val_loader is None:
            raise RuntimeError("DataModule not set up. Call setup() first.")
        return self._val_loader
