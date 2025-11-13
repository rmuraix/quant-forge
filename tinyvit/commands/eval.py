"""Evaluation command implementation."""

from pathlib import Path

import typer

from tinyvit.config_loader import load_config


def eval_command(
    config_path: Path = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to the configuration YAML file",
        exists=True,
        dir_okay=False,
    ),
) -> None:
    """Evaluate a trained model using the specified configuration.

    This is a dummy implementation that validates the config and prints the settings.
    """
    typer.echo("Starting evaluation...")

    # Load and validate configuration
    config = load_config(config_path)

    if config.eval is None:
        typer.echo("Error: eval configuration not found in config file", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Checkpoint: {config.eval.checkpoint_path}")
    typer.echo(f"Batch size: {config.eval.batch_size}")
    typer.echo(f"Device: {config.eval.device}")

    typer.echo("Evaluation completed successfully! (dummy)")
