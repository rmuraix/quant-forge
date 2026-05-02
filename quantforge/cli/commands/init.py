"""quantforge init command."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from quantforge.config.defaults import TEMPLATES
from quantforge.utils.errors import QuantForgeError
from quantforge.utils.rich import print_error, print_info


def init_command(
    output: Annotated[Path, typer.Argument(help="Path to write the config YAML")],
    template: Annotated[
        str, typer.Option("--template", "-t", help="Template: basic | qat | ptq")
    ] = "basic",
    debug: Annotated[bool, typer.Option("--debug", help="Show full traceback on error")] = False,
) -> None:
    """Create a YAML config file from a template."""
    try:
        if template not in TEMPLATES:
            available = ", ".join(TEMPLATES.keys())
            raise QuantForgeError(
                f"Unknown template '{template}'. Available: {available}"
            )

        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            typer.confirm(
                f"File '{output}' already exists. Overwrite?", abort=True
            )

        output.write_text(TEMPLATES[template])
        print_info(f"Created config: {output}  (template: {template})")
        print_info(f"Edit it, then run:\n\n  quantforge train -c {output}\n")

    except QuantForgeError as e:
        if debug:
            raise
        print_error(str(e))
        raise typer.Exit(1)
