"""Tracker registry."""

from __future__ import annotations

from quantforge.config.schema import TrackingConfig
from quantforge.tracking.base import Tracker
from quantforge.tracking.wandb_tracker import WandbTracker
from quantforge.utils.errors import TrackingError


def get_tracker(
    cfg: TrackingConfig,
    project_name: str,
    run_name: str | None = None,
) -> Tracker:
    """Return a Tracker for the given config."""
    if cfg.backend == "wandb":
        return WandbTracker(cfg, project_name, run_name)
    raise TrackingError(
        f"Unsupported tracking backend: '{cfg.backend}'. Only 'wandb' is supported."
    )
