"""Rich console utilities."""

from __future__ import annotations


def get_console():
    """Return a Rich console instance."""
    try:
        from rich.console import Console

        return Console()
    except ImportError:
        return None


def print_info(msg: str) -> None:
    """Print an info message."""
    console = get_console()
    if console:
        console.print(f"[green]{msg}[/green]")
    else:
        print(msg)


def print_error(msg: str) -> None:
    """Print an error message."""
    console = get_console()
    if console:
        console.print(f"[red]{msg}[/red]")
    else:
        print(f"ERROR: {msg}")


def print_warning(msg: str) -> None:
    """Print a warning message."""
    console = get_console()
    if console:
        console.print(f"[yellow]{msg}[/yellow]")
    else:
        print(f"WARNING: {msg}")
