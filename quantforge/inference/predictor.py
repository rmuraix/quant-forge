"""Inference predictor."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from quantforge.inference.preprocessing import build_inference_transforms
from quantforge.models.checkpoint import load_checkpoint
from quantforge.models.factory import ModelFactory
from quantforge.utils.device import resolve_device

if TYPE_CHECKING:
    from quantforge.config.schema import ExperimentConfig

logger = logging.getLogger("quantforge")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


class Predictor:
    """Runs inference from a checkpoint."""

    def __init__(
        self,
        checkpoint_path: Path,
        cfg: "ExperimentConfig | None" = None,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.cfg = cfg
        self._model: torch.nn.Module | None = None
        self._transform = None
        self._label_mapping: dict | None = None
        self._device: str = "cpu"

    def _load(self) -> None:
        """Load model from checkpoint."""
        if self._model is not None:
            return

        ckpt = load_checkpoint(self.checkpoint_path)
        model_meta = ckpt.get("model", {})
        preprocess_meta = ckpt.get("preprocess", {})
        self._label_mapping = ckpt.get("label_mapping", {})

        # Resolve num_classes
        num_classes = model_meta.get("num_classes")
        if num_classes is None and self.cfg:
            num_classes = self.cfg.model.num_classes
        if num_classes is None:
            id_to_label = self._label_mapping.get("id_to_label", {}) if self._label_mapping else {}
            num_classes = len(id_to_label) or 1

        # Build model config from checkpoint metadata
        if self.cfg:
            model_cfg = self.cfg.model
        else:
            from quantforge.config.schema import ModelConfig

            model_cfg = ModelConfig(
                name=model_meta.get("name", "resnet18"),
                pretrained=False,
                num_classes=num_classes,
                in_chans=model_meta.get("in_chans", 3),
            )

        factory = ModelFactory(model_cfg, num_classes)
        model = factory.create()
        model.load_state_dict(ckpt["model_state_dict"])

        device = resolve_device(self.cfg.training.device if self.cfg else "auto")
        model = model.to(device)
        model.eval()

        self._model = model
        self._device = device
        self._transform = build_inference_transforms(preprocess_meta)

    def predict_image(self, image_path: Path) -> dict:
        """Run inference on a single image.

        Args:
            image_path: Path to image file.

        Returns:
            dict with 'image', 'label', 'confidence', 'label_id'.
        """
        from PIL import Image

        self._load()

        img = Image.open(image_path).convert("RGB")
        tensor = self._transform(img).unsqueeze(0).to(self._device)

        with torch.no_grad():
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=1)
            conf, pred_id = probs.max(1)

        pred_id_int = pred_id.item()
        label = str(pred_id_int)
        if self._label_mapping:
            id_to_label = self._label_mapping.get("id_to_label", {})
            label = id_to_label.get(pred_id_int) or id_to_label.get(str(pred_id_int)) or label

        return {
            "image": str(image_path),
            "label": label,
            "label_id": pred_id_int,
            "confidence": conf.item(),
        }

    def predict_dir(self, dir_path: Path) -> list[dict]:
        """Run inference on all images in a directory.

        Args:
            dir_path: Directory of images.

        Returns:
            List of prediction dicts.
        """
        images = [
            p for p in dir_path.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if not images:
            logger.warning("No images found in '%s'", dir_path)
            return []

        results = []
        for img_path in sorted(images):
            try:
                result = self.predict_image(img_path)
                results.append(result)
            except Exception as e:
                logger.warning("Failed to predict '%s': %s", img_path, e)
        return results
