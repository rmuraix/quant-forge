"""Artifact metadata helpers."""

from __future__ import annotations

import sys
from datetime import datetime


def build_environment_metadata() -> dict[str, str]:
    """Build environment metadata dict."""
    meta: dict[str, str] = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "timestamp": datetime.now().isoformat(),
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
