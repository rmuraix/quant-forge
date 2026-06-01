"""ArtifactWriter: local artifact saving."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from quantforge.config.resolve import ResolvedConfig
from quantforge.config.schema import ExperimentConfig
from quantforge.utils.paths import ensure_dir


class ArtifactWriter:
    """Manages local artifact saving for a single run."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir

    def setup(self) -> None:
        """Create the run directory structure."""
        for subdir in ("checkpoints", "quantized", "metrics", "predictions", "logs"):
            ensure_dir(self.run_dir / subdir)

    def save_config(self, cfg: ExperimentConfig) -> None:
        """Save raw experiment config as YAML."""
        path = self.run_dir / "config.yaml"
        with open(path, "w") as f:
            yaml.dump(cfg.model_dump(), f, default_flow_style=False, sort_keys=False)

    def save_resolved_config(self, rcfg: ResolvedConfig) -> None:
        """Save resolved runtime config as YAML."""
        path = self.run_dir / "config.resolved.yaml"
        with open(path, "w") as f:
            yaml.dump(rcfg.to_dict(), f, default_flow_style=False, sort_keys=False)

    def save_label_mapping(self, mapping: dict) -> None:
        """Save label mapping as JSON."""
        path = self.run_dir / "label_mapping.json"
        with open(path, "w") as f:
            json.dump(mapping, f, indent=2)

    def save_checkpoint(self, state: dict, name: str) -> Path:
        """Save a checkpoint dict as a .pt file. Returns the saved path."""
        import torch

        path = self.run_dir / "checkpoints" / name
        torch.save(state, path)
        return path

    def save_quantized_model(self, state: dict, name: str) -> Path:
        """Save a quantized model dict. Returns the saved path."""
        import torch

        path = self.run_dir / "quantized" / name
        torch.save(state, path)
        return path

    def save_train_metrics(self, metrics: dict) -> None:
        """Append a metrics dict to metrics/train.jsonl."""
        path = self.run_dir / "metrics" / "train.jsonl"
        with open(path, "a") as f:
            f.write(json.dumps(metrics) + "\n")

    def save_eval_metrics(self, metrics: dict) -> None:
        """Write evaluation metrics to metrics/eval.json."""
        path = self.run_dir / "metrics" / "eval.json"
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)

    def save_latency_metrics(self, metrics: dict) -> None:
        """Write latency metrics to metrics/latency.json."""
        path = self.run_dir / "metrics" / "latency.json"
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)

    def save_predictions(self, predictions: list[dict]) -> None:
        """Append predictions to predictions/predictions.jsonl."""
        path = self.run_dir / "predictions" / "predictions.jsonl"
        with open(path, "a") as f:
            for pred in predictions:
                f.write(json.dumps(pred) + "\n")

    def save_manifest(self, manifest: dict) -> None:
        """Write manifest.json."""
        path = self.run_dir / "manifest.json"
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2)
