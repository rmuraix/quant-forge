"""Resolved runtime config generation."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from quantforge.config.schema import ExperimentConfig
from quantforge.utils.device import resolve_device


def _get_versions() -> dict[str, str]:
    """Collect package versions safely."""
    versions: dict[str, str] = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    }
    for pkg in ("torch", "torchao", "timm", "datasets"):
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "not installed"
    try:
        import torch

        versions["cuda"] = torch.version.cuda or "none"
    except ImportError:
        versions["cuda"] = "none"
    return versions


@dataclass
class ResolvedConfig:
    """Runtime-resolved configuration."""

    run_id: str
    run_dir: Path
    device: str
    num_classes: int | None
    label_mapping: dict | None
    preprocessing: dict
    command: str
    versions: dict[str, str]
    config: ExperimentConfig
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to a plain dict for YAML saving."""
        return {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "device": self.device,
            "num_classes": self.num_classes,
            "label_mapping": self.label_mapping,
            "preprocessing": self.preprocessing,
            "command": self.command,
            "versions": self.versions,
            "config": self.config.model_dump(),
        }


def resolve_config(
    cfg: ExperimentConfig,
    command: str,
    num_classes: int | None = None,
    label_mapping: dict | None = None,
) -> ResolvedConfig:
    """Resolve runtime configuration from an ExperimentConfig.

    Args:
        cfg: Validated experiment config.
        command: CLI command name (e.g. 'train', 'eval').
        num_classes: Inferred from dataset if not provided in config.
        label_mapping: Label id<->name mapping from dataset.

    Returns:
        ResolvedConfig with computed runtime fields.
    """
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(cfg.project.output_dir) / cfg.project.name / run_id
    device = resolve_device(cfg.training.device)

    resolved_num_classes = cfg.model.num_classes or num_classes

    mean = cfg.preprocess.mean or [0.485, 0.456, 0.406]
    std = cfg.preprocess.std or [0.229, 0.224, 0.225]
    preprocessing = {
        "image_size": cfg.preprocess.image_size,
        "mean": mean,
        "std": std,
        "interpolation": cfg.preprocess.interpolation,
    }

    return ResolvedConfig(
        run_id=run_id,
        run_dir=run_dir,
        device=device,
        num_classes=resolved_num_classes,
        label_mapping=label_mapping,
        preprocessing=preprocessing,
        command=command,
        versions=_get_versions(),
        config=cfg,
    )
