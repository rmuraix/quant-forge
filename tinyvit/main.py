"""Main CLI entry point for TinyVit."""

import typer

from tinyvit.commands.eval import eval_command
from tinyvit.commands.predict import predict_command
from tinyvit.commands.train import train_command

app = typer.Typer(
    name="tinyvit",
    help="TinyVit - A modular Vision Transformer training framework",
    add_completion=False,
)

app.command(name="train", help="Train a model")(train_command)
app.command(name="eval", help="Evaluate a trained model")(eval_command)
app.command(name="predict", help="Make predictions with a trained model")(
    predict_command
)


if __name__ == "__main__":
    app()
