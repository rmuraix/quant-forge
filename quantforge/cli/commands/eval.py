"""quantforge eval command."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from quantforge.utils.errors import QuantForgeError
from quantforge.utils.rich import print_error, print_info


def eval_command(
    config: Annotated[Path, typer.Option("--config", "-c", help="Path to config YAML")],
    checkpoint: Annotated[Path, typer.Option("--checkpoint", help="Path to checkpoint")],
    set_: Annotated[list[str], typer.Option("--set")] = [],
    debug: Annotated[bool, typer.Option("--debug")] = False,
) -> None:
    """Evaluate a checkpoint on the validation set."""
    try:
        from quantforge.config.loader import load_config
        from quantforge.evaluation.evaluator import Evaluator
        from quantforge.utils.logging import setup_logging

        cfg = load_config(config, list(set_) or None)
        setup_logging(cfg.logging.level, cfg.logging.rich)

        evaluator = Evaluator(cfg, checkpoint)
        results = evaluator.run()

        print_info("\nEvaluation Results:")
        for k, v in results.items():
            if isinstance(v, float):
                print_info(f"  {k}: {v:.4f}")
            else:
                print_info(f"  {k}: {v}")

    except QuantForgeError as e:
        if debug:
            raise
        print_error(str(e))
        raise typer.Exit(1)
