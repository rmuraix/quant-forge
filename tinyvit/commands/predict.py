"""Prediction command implementation."""

from pathlib import Path

import typer

from tinyvit.config_loader import load_config


def predict_command(
    config_path: Path = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to the configuration YAML file",
        exists=True,
        dir_okay=False,
    ),
) -> None:
    """Make predictions using a trained model and the specified configuration.

    This is a dummy implementation that validates the config and prints the settings.
    """
    typer.echo("Starting prediction...")

    # Load and validate configuration
    config = load_config(config_path)

    if config.predict is None:
        typer.echo("Error: predict configuration not found in config file", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Checkpoint: {config.predict.checkpoint_path}")
    typer.echo(f"Input path: {config.predict.input_path}")
    typer.echo(f"Output path: {config.predict.output_path}")
    typer.echo(f"Device: {config.predict.device}")

    typer.echo("Prediction completed successfully! (dummy)")
