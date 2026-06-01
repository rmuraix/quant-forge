"""Model evaluator."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING


from quantforge.data.datamodule import HFDataModule
from quantforge.evaluation.latency import measure_latency
from quantforge.evaluation.size import get_model_size_mb
from quantforge.models.checkpoint import load_checkpoint
from quantforge.models.factory import ModelFactory
from quantforge.training.loop import eval_one_epoch
from quantforge.training.losses import get_loss_fn

if TYPE_CHECKING:
    from quantforge.config.schema import ExperimentConfig

logger = logging.getLogger("quantforge")


class Evaluator:
    """Evaluates a checkpoint on the validation set."""

    def __init__(self, cfg: "ExperimentConfig", checkpoint_path: Path) -> None:
        self.cfg = cfg
        self.checkpoint_path = checkpoint_path

    def run(self) -> dict:
        """Run evaluation.

        Returns:
            dict with eval metrics.
        """
        from quantforge.utils.device import resolve_device

        device = resolve_device(self.cfg.training.device)

        # Load checkpoint
        logger.info("Loading checkpoint: %s", self.checkpoint_path)
        ckpt = load_checkpoint(self.checkpoint_path)
        model_meta = ckpt.get("model", {})
        num_classes = model_meta.get("num_classes") or self.cfg.model.num_classes

        # Load data
        logger.info("Loading dataset for evaluation...")
        dm = HFDataModule(self.cfg)
        dm.setup()
        if num_classes is None:
            num_classes = dm.num_classes

        # Create model and load weights
        factory = ModelFactory(self.cfg.model, num_classes)
        model = factory.create()
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device)
        model.eval()

        # Eval loop
        loss_fn = get_loss_fn()
        logger.info("Running evaluation...")
        val_metrics = eval_one_epoch(model, dm.val_dataloader(), loss_fn, device)

        # Model size
        size_mb = get_model_size_mb(model)

        # Latency
        image_size = self.cfg.preprocess.image_size
        latency_metrics = measure_latency(
            model,
            input_shape=(1, 3, image_size, image_size),
            device=device,
        )

        results = {
            "eval/loss": val_metrics["loss"],
            "eval/top1": val_metrics["top1"],
            "eval/top5": val_metrics["top5"],
            "eval/model_size_mb": size_mb,
            "eval/latency_ms": latency_metrics["latency_ms"],
            "eval/throughput_images_per_sec": latency_metrics[
                "throughput_images_per_sec"
            ],
        }
        logger.info(
            "Eval results: top1=%.4f  top5=%.4f  size=%.1f MB  latency=%.1f ms",
            results["eval/top1"],
            results["eval/top5"],
            results["eval/model_size_mb"],
            results["eval/latency_ms"],
        )
        return results
