"""Training command implementation."""

from pathlib import Path

import typer

from tinyvit.config_loader import load_config
from tinyvit.models import build_timm_model


def train_command(
    config_path: Path = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to the configuration YAML file",
        exists=True,
        dir_okay=False,
    ),
) -> None:
    """Train a model using the specified configuration.

    This is a dummy implementation that validates the config and prints the settings.
    """
    typer.echo("Starting training...")

    # Load and validate configuration
    config = load_config(config_path)

    # Build model via timm
    model = build_timm_model(config.model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    typer.echo(f"Dataset: {config.data.dataset_name}")
    typer.echo(f"Batch size: {config.data.batch_size}")
    typer.echo(
        f"Model: {config.model.model_name} "
        f"(trainable params: {trainable_params:,}/{total_params:,})"
    )
    typer.echo(f"Epochs: {config.training.epochs}")
    typer.echo(f"Learning rate: {config.training.learning_rate}")
    typer.echo(f"Optimizer: {config.training.optimizer}")
    typer.echo(f"Device: {config.training.device}")

    typer.echo("Training completed successfully! (dummy)")
