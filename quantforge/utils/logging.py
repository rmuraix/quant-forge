"""Logging setup utilities."""

from __future__ import annotations

import logging


def get_logger(name: str = "quantforge") -> logging.Logger:
    """Return a logger for the given name."""
    return logging.getLogger(name)


def setup_logging(level: str = "info", use_rich: bool = True) -> None:
    """Configure root logging."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    if use_rich:
        try:
            from rich.logging import RichHandler

            logging.basicConfig(
                level=numeric_level,
                format="%(message)s",
                handlers=[RichHandler(rich_tracebacks=True)],
            )
            return
        except ImportError:
            pass
    logging.basicConfig(
        level=numeric_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
