"""Standard Trainer for fine-tuning."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from tqdm import tqdm

from quantforge.artifacts.manifest import build_manifest
from quantforge.data.datamodule import HFDataModule
from quantforge.models.checkpoint import save_checkpoint
from quantforge.models.factory import ModelFactory
from quantforge.training.loop import eval_one_epoch, train_one_epoch
from quantforge.training.losses import get_loss_fn
from quantforge.training.optim import get_optimizer
from quantforge.training.schedulers import get_scheduler
from quantforge.utils.seed import set_seed

if TYPE_CHECKING:
    from quantforge.artifacts.writer import ArtifactWriter
    from quantforge.config.resolve import ResolvedConfig
    from quantforge.config.schema import ExperimentConfig
    from quantforge.quantization.base import QuantizationStrategy
    from quantforge.tracking.base import Tracker

logger = logging.getLogger("quantforge")


class Trainer:
    """Standard fine-tuning trainer."""

    def __init__(
        self,
        cfg: "ExperimentConfig",
        rcfg: "ResolvedConfig",
        writer: "ArtifactWriter",
        tracker: "Tracker",
        quantizer: "QuantizationStrategy",
    ) -> None:
        self.cfg = cfg
        self.rcfg = rcfg
        self.writer = writer
        self.tracker = tracker
        self.quantizer = quantizer

    def run(self) -> None:
        """Execute the full training pipeline."""
        cfg = self.cfg
        rcfg = self.rcfg
        device = rcfg.device

        # 1. Seed
        set_seed(cfg.project.seed, cfg.training.deterministic)
        logger.info("Seed set to %d", cfg.project.seed)

        # 2. Set up artifact dirs
        self.writer.setup()
        self.writer.save_config(cfg)
        self.writer.save_resolved_config(rcfg)

        # 3. Load data
        logger.info("Loading dataset '%s'...", cfg.dataset.name)
        dm = HFDataModule(cfg)
        dm.setup()
        num_classes = dm.num_classes
        label_mapping = dm.label_mapping
        self.writer.save_label_mapping(label_mapping)
        logger.info("Classes: %d", num_classes)

        # Update resolved config num_classes
        rcfg.num_classes = num_classes
        rcfg.label_mapping = label_mapping
        self.writer.save_resolved_config(rcfg)

        # 4. Create model
        factory = ModelFactory(cfg.model, num_classes)
        model = factory.create()
        logger.info("Created model: %s", cfg.model.name)

        # 5. Prepare model for quantization
        model = self.quantizer.prepare_model(model)
        model = model.to(device)

        # 6. Build optimizer, scheduler, loss
        optimizer = get_optimizer(model, cfg.training)
        train_loader = dm.train_dataloader()
        steps_per_epoch = len(train_loader)
        scheduler = get_scheduler(optimizer, cfg.training, steps_per_epoch)
        loss_fn = get_loss_fn(cfg.training)

        # AMP scaler
        scaler = None
        if cfg.training.amp and device.startswith("cuda"):
            scaler = torch.cuda.amp.GradScaler()

        # 7. Start tracker
        self.tracker.start()
        if cfg.tracking.log_config:
            self.tracker.log_config(cfg.model_dump())

        # Notify quantizer
        self.quantizer.on_train_start(model)

        best_val_top1 = 0.0
        val_loader = dm.val_dataloader()

        epoch_bar = tqdm(range(cfg.training.epochs), desc="Epochs")
        for epoch in epoch_bar:
            train_metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                loss_fn,
                device,
                cfg.training,
                epoch,
                scaler,
                log_interval=cfg.logging.log_interval,
            )
            val_metrics = eval_one_epoch(model, val_loader, loss_fn, device)

            step = (epoch + 1) * steps_per_epoch
            metrics_row = {
                "epoch": epoch + 1,
                "train/loss": train_metrics["loss"],
                "train/lr": train_metrics["lr"],
                "val/loss": val_metrics["loss"],
                "val/top1": val_metrics["top1"],
                "val/top5": val_metrics["top5"],
            }
            self.writer.save_train_metrics(metrics_row)
            self.tracker.log_metrics(metrics_row, step=step)

            epoch_bar.set_postfix(
                train_loss=f"{train_metrics['loss']:.4f}",
                val_top1=f"{val_metrics['top1']:.4f}",
                val_top5=f"{val_metrics['top5']:.4f}",
            )

            # Notify quantizer
            self.quantizer.on_epoch_end(model, epoch)

            # Save last checkpoint
            save_checkpoint(
                self.writer.run_dir / "checkpoints" / "last.pt",
                model,
                cfg,
                label_mapping,
                metrics=metrics_row,
            )

            # Save best checkpoint
            if cfg.checkpoint.save_best and val_metrics["top1"] > best_val_top1:
                best_val_top1 = val_metrics["top1"]
                save_checkpoint(
                    self.writer.run_dir / "checkpoints" / "best.pt",
                    model,
                    cfg,
                    label_mapping,
                    metrics=metrics_row,
                )
                logger.info("  ✓ New best val/top1=%.4f — saved best.pt", best_val_top1)

        # 8. Convert model (quantization finalization)
        self.quantizer.convert_model(model)

        # 9. Summary
        self.tracker.log_summary(
            {
                "best_val_top1": best_val_top1,
                "local_run_dir": str(self.writer.run_dir),
                "best_checkpoint": "checkpoints/best.pt",
            }
        )

        # 10. Manifest
        manifest = build_manifest(
            run_id=rcfg.run_id,
            run_dir=rcfg.run_dir,
            command="train",
            extra={
                "best_val_top1": best_val_top1,
                "num_classes": num_classes,
                "epochs_completed": cfg.training.epochs,
            },
        )
        self.writer.save_manifest(manifest)

        self.tracker.finish()
        logger.info("Training complete. Run dir: %s", self.writer.run_dir)
