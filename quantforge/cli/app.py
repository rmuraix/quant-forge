"""QuantForge CLI application."""

from __future__ import annotations

import typer

from quantforge.cli.commands.eval import eval_command
from quantforge.cli.commands.infer import infer_command
from quantforge.cli.commands.init import init_command
from quantforge.cli.commands.inspect import inspect_config, inspect_dataset, inspect_model
from quantforge.cli.commands.qat import qat_command
from quantforge.cli.commands.quantize import quantize_command
from quantforge.cli.commands.train import train_command

app = typer.Typer(
    name="quantforge",
    help="QuantForge v0.1 — Fine-tune and quantize image classifiers.",
    no_args_is_help=True,
)

# Register main commands
app.command("init")(init_command)
app.command("train")(train_command)
app.command("qat")(qat_command)
app.command("quantize")(quantize_command)
app.command("eval")(eval_command)
app.command("infer")(infer_command)

# Inspect subcommands
inspect_app = typer.Typer(name="inspect", help="Inspection utilities.", no_args_is_help=True)
app.add_typer(inspect_app, name="inspect")
inspect_app.command("config")(inspect_config)
inspect_app.command("dataset")(inspect_dataset)
inspect_app.command("model")(inspect_model)
