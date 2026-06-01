"""quantforge inspect subcommands."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from quantforge.utils.errors import QuantForgeError
from quantforge.utils.rich import print_error, print_info


def inspect_config(
    config: Annotated[Path, typer.Option("--config", "-c", help="Path to config YAML")],
    debug: Annotated[bool, typer.Option("--debug")] = False,
) -> None:
    """Inspect and validate a config file."""
    try:
        import yaml

        from quantforge.config.loader import load_config

        cfg = load_config(config)
        print_info(f"Config: {config}  (valid)")
        typer.echo(
            yaml.dump(cfg.model_dump(), default_flow_style=False, sort_keys=False)
        )

    except QuantForgeError as e:
        if debug:
            raise
        print_error(str(e))
        raise typer.Exit(1)


def inspect_dataset(
    config: Annotated[Path, typer.Option("--config", "-c", help="Path to config YAML")],
    debug: Annotated[bool, typer.Option("--debug")] = False,
) -> None:
    """Inspect a dataset defined in the config."""
    try:
        from quantforge.config.loader import load_config
        from quantforge.data.hf_dataset import load_hf_dataset, validate_columns
        from quantforge.data.labels import infer_label_mapping, infer_num_classes

        cfg = load_config(config)
        print_info(f"Loading dataset '{cfg.dataset.name}'...")

        dataset = load_hf_dataset(cfg.dataset)
        train_split_name = cfg.dataset.split.train
        split = dataset[train_split_name]

        validate_columns(split, cfg.dataset.image_column, cfg.dataset.label_column)
        num_classes = infer_num_classes(split, cfg.dataset.label_column)
        label_mapping = infer_label_mapping(split, cfg.dataset.label_column)

        print_info(f"  Split '{train_split_name}': {len(split)} examples")
        print_info(f"  Columns: {list(split.features.keys())}")
        print_info(f"  Num classes: {num_classes}")
        print_info(
            f"  Labels: {list(label_mapping['id_to_label'].values())[:10]}{'...' if num_classes > 10 else ''}"
        )

    except QuantForgeError as e:
        if debug:
            raise
        print_error(str(e))
        raise typer.Exit(1)


def inspect_model(
    config: Annotated[Path, typer.Option("--config", "-c", help="Path to config YAML")],
    debug: Annotated[bool, typer.Option("--debug")] = False,
) -> None:
    """Inspect a model defined in the config."""
    try:
        from quantforge.config.loader import load_config
        from quantforge.models.factory import ModelFactory
        from quantforge.models.metadata import get_model_info

        cfg = load_config(config)
        num_classes = cfg.model.num_classes or 10  # fallback for inspection

        print_info(
            f"Creating model '{cfg.model.name}' (pretrained={cfg.model.pretrained})..."
        )
        factory = ModelFactory(cfg.model, num_classes)
        model = factory.create()

        info = get_model_info(model)
        print_info(f"  Total params:     {info['total_params']:,}")
        print_info(f"  Trainable params: {info['trainable_params']:,}")

    except QuantForgeError as e:
        if debug:
            raise
        print_error(str(e))
        raise typer.Exit(1)
