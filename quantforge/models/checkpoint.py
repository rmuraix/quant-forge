"""Self-contained checkpoint save/load."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from quantforge.utils.errors import CheckpointError

if TYPE_CHECKING:
    import torch.nn as nn

    from quantforge.config.schema import ExperimentConfig

CHECKPOINT_FORMAT_VERSION = 1


def _get_environment() -> dict[str, str]:
    meta: dict[str, str] = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    }
    for pkg in ("torch", "torchao", "timm", "datasets"):
        try:
            mod = __import__(pkg)
            meta[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            meta[pkg] = "not installed"
    try:
        import torch

        meta["cuda"] = torch.version.cuda or "none"
    except ImportError:
        meta["cuda"] = "none"
    return meta


def save_checkpoint(
    path: Path,
    model: "nn.Module",
    cfg: "ExperimentConfig",
    label_mapping: dict,
    metrics: dict | None = None,
    quantization_meta: dict | None = None,
) -> None:
    """Save a self-contained checkpoint.

    Args:
        path: Destination file path.
        model: Model whose state_dict to save.
        cfg: Experiment config.
        label_mapping: Label id<->name mapping.
        metrics: Optional metrics to embed.
        quantization_meta: Optional quantization metadata override.
    """
    import torch

    quant_meta = quantization_meta or {
        "enabled": cfg.quantization.enabled,
        "backend": cfg.quantization.backend,
        "mode": cfg.quantization.mode,
        "dtype": cfg.quantization.dtype,
        "recipe": cfg.quantization.recipe,
    }

    state = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_state_dict": model.state_dict(),
        "model": {
            "name": cfg.model.name,
            "num_classes": cfg.model.num_classes,
            "in_chans": cfg.model.in_chans,
        },
        "preprocess": {
            "image_size": cfg.preprocess.image_size,
            "mean": cfg.preprocess.mean or [0.485, 0.456, 0.406],
            "std": cfg.preprocess.std or [0.229, 0.224, 0.225],
            "interpolation": cfg.preprocess.interpolation,
        },
        "label_mapping": label_mapping,
        "quantization": quant_meta,
        "config": cfg.model_dump(),
        "metrics": metrics or {},
        "environment": _get_environment(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: Path) -> dict:
    """Load a checkpoint and validate format version.

    Args:
        path: Path to checkpoint file.

    Returns:
        Checkpoint dict.
    """
    import torch

    if not path.exists():
        raise CheckpointError(
            f"Checkpoint not found: {path}\n\n"
            "Make sure to run training first to generate a checkpoint."
        )
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        raise CheckpointError(f"Failed to load checkpoint '{path}': {e}") from e

    if not isinstance(ckpt, dict) or "format_version" not in ckpt:
        raise CheckpointError(
            f"Checkpoint '{path}' is not a valid QuantForge checkpoint (missing format_version)."
        )
    if ckpt["format_version"] != CHECKPOINT_FORMAT_VERSION:
        raise CheckpointError(
            f"Checkpoint format version mismatch: expected {CHECKPOINT_FORMAT_VERSION}, "
            f"got {ckpt['format_version']}."
        )
    return ckpt
