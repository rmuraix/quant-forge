"""Base tracker interface."""

from __future__ import annotations


class Tracker:
    """Abstract experiment tracker."""

    def start(self) -> None:
        """Initialize the tracker."""

    def log_config(self, config: dict) -> None:
        """Log experiment configuration."""

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        """Log metrics at a given step."""

    def log_summary(self, values: dict) -> None:
        """Log summary values."""

    def finish(self) -> None:
        """Finalize the tracker."""
