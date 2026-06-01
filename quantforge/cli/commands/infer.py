"""quantforge infer command."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from quantforge.utils.errors import QuantForgeError
from quantforge.utils.rich import print_error, print_info


def infer_command(
    checkpoint: Annotated[
        Path, typer.Option("--checkpoint", help="Path to checkpoint")
    ],
    image: Annotated[
        Path | None, typer.Option("--image", help="Path to a single image")
    ] = None,
    input_dir: Annotated[
        Path | None, typer.Option("--input-dir", help="Directory of images")
    ] = None,
    config: Annotated[
        Path | None, typer.Option("--config", "-c", help="Optional config YAML")
    ] = None,
    output: Annotated[
        Path | None, typer.Option("--output", help="Optional output JSONL path")
    ] = None,
    debug: Annotated[bool, typer.Option("--debug")] = False,
) -> None:
    """Run inference on a single image or directory."""
    try:
        from quantforge.config.loader import load_config
        from quantforge.inference.output import format_predictions
        from quantforge.inference.predictor import Predictor

        if image is None and input_dir is None:
            raise QuantForgeError("Either --image or --input-dir must be provided.")

        cfg = load_config(config) if config else None
        predictor = Predictor(checkpoint, cfg)

        predictions: list[dict] = []
        if image is not None:
            result = predictor.predict_image(image)
            predictions.append(result)
        elif input_dir is not None:
            predictions = predictor.predict_dir(input_dir)

        formatted = format_predictions(predictions)
        print_info("\nPredictions:")
        typer.echo(formatted)

        if output is not None:
            import json

            output.parent.mkdir(parents=True, exist_ok=True)
            with open(output, "w") as f:
                for pred in predictions:
                    f.write(json.dumps(pred) + "\n")
            print_info(f"\nSaved to: {output}")

    except QuantForgeError as e:
        if debug:
            raise
        print_error(str(e))
        raise typer.Exit(1)
