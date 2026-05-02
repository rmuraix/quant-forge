"""quantforge quantize command."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from quantforge.utils.errors import QuantForgeError
from quantforge.utils.rich import print_error, print_info


def quantize_command(
    config: Annotated[Path, typer.Option("--config", "-c", help="Path to config YAML")],
    checkpoint: Annotated[Path, typer.Option("--checkpoint", help="Path to fine-tuned checkpoint")],
    output: Annotated[Path, typer.Option("--output", help="Output path for quantized model")],
    set_: Annotated[list[str], typer.Option("--set")] = [],
    debug: Annotated[bool, typer.Option("--debug")] = False,
) -> None:
    """Apply post-training quantization to a fine-tuned checkpoint."""
    try:

        from quantforge.config.loader import load_config
        from quantforge.models.checkpoint import load_checkpoint, save_checkpoint
        from quantforge.models.factory import ModelFactory
        from quantforge.quantization.torchao_ptq import TorchAOPTQStrategy
        from quantforge.utils.device import resolve_device
        from quantforge.utils.logging import setup_logging

        cfg = load_config(config, list(set_) or None)
        setup_logging(cfg.logging.level, cfg.logging.rich)
        device = resolve_device(cfg.training.device)

        print_info(f"Loading checkpoint: {checkpoint}")
        ckpt = load_checkpoint(checkpoint)
        model_meta = ckpt.get("model", {})
        num_classes = model_meta.get("num_classes") or cfg.model.num_classes or 1

        factory = ModelFactory(cfg.model, num_classes)
        model = factory.create()
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device)

        print_info("Applying PTQ quantization...")
        strategy = TorchAOPTQStrategy()
        quantized_model = strategy.convert_model(model)

        output.parent.mkdir(parents=True, exist_ok=True)
        label_mapping = ckpt.get("label_mapping", {})
        save_checkpoint(
            output,
            quantized_model,
            cfg,
            label_mapping,
            metrics=ckpt.get("metrics"),
            quantization_meta={
                "enabled": True,
                "backend": "torchao",
                "mode": "ptq",
                "dtype": cfg.quantization.dtype,
                "recipe": cfg.quantization.recipe,
            },
        )
        print_info(f"Quantized model saved to: {output}")

    except QuantForgeError as e:
        if debug:
            raise
        print_error(str(e))
        raise typer.Exit(1)
