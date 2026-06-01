"""quantforge qat command."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from quantforge.utils.errors import ConfigError, QuantForgeError
from quantforge.utils.rich import print_error, print_info


def qat_command(
    config: Annotated[Path, typer.Option("--config", "-c", help="Path to config YAML")],
    set_: Annotated[
        list[str], typer.Option("--set", help="Override config values (key.path=value)")
    ] = [],
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    wandb_mode: Annotated[str | None, typer.Option("--wandb-mode")] = None,
    debug: Annotated[bool, typer.Option("--debug")] = False,
) -> None:
    """Run quantization-aware training."""
    try:
        from quantforge.artifacts.writer import ArtifactWriter
        from quantforge.config.loader import load_config
        from quantforge.config.resolve import resolve_config
        from quantforge.quantization.registry import get_quantizer
        from quantforge.tracking.registry import get_tracker
        from quantforge.training.trainer import Trainer
        from quantforge.utils.logging import setup_logging

        overrides = list(set_)
        if wandb_mode is not None:
            overrides.append(f"tracking.mode={wandb_mode}")

        cfg = load_config(config, overrides or None)
        setup_logging(cfg.logging.level, cfg.logging.rich)

        # Validate QAT requirements
        q = cfg.quantization
        if not q.enabled or q.backend != "torchao" or q.mode != "qat":
            raise ConfigError(
                "The qat command requires the config to have:\n\n"
                "  quantization:\n"
                "    enabled: true\n"
                "    backend: torchao\n"
                "    mode: qat\n\n"
                "Current values:\n"
                f"  enabled={q.enabled}  backend={q.backend}  mode={q.mode}\n\n"
                "Fix your config or use overrides:\n"
                "  --set quantization.enabled=true\n"
                "  --set quantization.backend=torchao\n"
                "  --set quantization.mode=qat"
            )

        if dry_run:
            print_info("Dry run: QAT config is valid.")
            return

        rcfg = resolve_config(cfg, command="qat")
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
