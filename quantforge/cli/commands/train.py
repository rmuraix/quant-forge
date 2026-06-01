"""quantforge train command."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from quantforge.utils.errors import QuantForgeError
from quantforge.utils.rich import print_error, print_info


def train_command(
    config: Annotated[Path, typer.Option("--config", "-c", help="Path to config YAML")],
    set_: Annotated[
        list[str], typer.Option("--set", help="Override config values (key.path=value)")
    ] = [],
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Validate config without training")
    ] = False,
    wandb_mode: Annotated[
        str | None,
        typer.Option("--wandb-mode", help="wandb mode: online | offline | disabled"),
    ] = None,
    debug: Annotated[
        bool, typer.Option("--debug", help="Show full traceback on error")
    ] = False,
) -> None:
    """Run standard fine-tuning."""
    try:
        from quantforge.artifacts.writer import ArtifactWriter
        from quantforge.config.loader import load_config
        from quantforge.config.resolve import resolve_config
        from quantforge.quantization.registry import get_quantizer
        from quantforge.tracking.registry import get_tracker
        from quantforge.training.trainer import Trainer
        from quantforge.utils.logging import setup_logging

        # Apply wandb_mode override
        overrides = list(set_)
        if wandb_mode is not None:
            overrides.append(f"tracking.mode={wandb_mode}")

        cfg = load_config(config, overrides or None)
        setup_logging(cfg.logging.level, cfg.logging.rich)

        if dry_run:
            print_info("Dry run: config is valid.")
            print_info(f"  Project: {cfg.project.name}")
            print_info(f"  Model:   {cfg.model.name}")
            print_info(f"  Dataset: {cfg.dataset.name}")
            print_info(f"  Epochs:  {cfg.training.epochs}")
            return

        rcfg = resolve_config(cfg, command="train")
        writer = ArtifactWriter(rcfg.run_dir)
        tracker = get_tracker(cfg.tracking, cfg.project.name, rcfg.run_id)
        quantizer = get_quantizer(cfg.quantization)

        trainer = Trainer(cfg, rcfg, writer, tracker, quantizer)
        trainer.run()

    except QuantForgeError as e:
        if debug:
            raise
        print_error(str(e))
        raise typer.Exit(1)
