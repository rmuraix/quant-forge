"""Weights & Biases tracker implementation."""

from __future__ import annotations

from quantforge.config.schema import TrackingConfig
from quantforge.tracking.base import Tracker
from quantforge.utils.errors import TrackingError


class WandbTracker(Tracker):
    """Tracker that logs to Weights & Biases."""

    def __init__(
        self,
        cfg: TrackingConfig,
        project_name: str,
        run_name: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.project_name = project_name
        self.run_name = run_name or cfg.run_name
        self._run = None
        self._enabled = cfg.mode != "disabled"

    def start(self) -> None:
        """Initialize wandb run."""
        if not self._enabled:
            return
        try:
            import wandb
        except ImportError as e:
            raise TrackingError(
                "wandb is not installed. Install it with: pip install wandb"
            ) from e
        try:
            self._run = wandb.init(
                project=self.cfg.project,
                entity=self.cfg.entity,
                name=self.run_name,
                group=self.cfg.group,
                job_type=self.cfg.job_type,
                tags=self.cfg.tags or [],
                notes=self.cfg.notes,
                mode=self.cfg.mode,
                reinit=True,
            )
        except Exception as e:
            raise TrackingError(f"Failed to initialize wandb: {e}") from e

    def log_config(self, config: dict) -> None:
        """Log experiment config to wandb."""
        if not self._enabled or self._run is None:
            return
        if self.cfg.log_config:
            self._run.config.update(config, allow_val_change=True)

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        """Log metrics to wandb."""
        if not self._enabled or self._run is None:
            return
        if self.cfg.log_metrics:
            self._run.log(metrics, step=step)

    def log_summary(self, values: dict) -> None:
        """Log summary values to wandb."""
        if not self._enabled or self._run is None:
            return
        for k, v in values.items():
            self._run.summary[k] = v

    def finish(self) -> None:
        """Finish wandb run."""
        if not self._enabled or self._run is None:
            return
        self._run.finish()
