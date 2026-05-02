"""Device resolution utilities."""

from __future__ import annotations


def resolve_device(device: str) -> str:
    """Resolve 'auto' to 'cuda' or 'cpu' based on availability."""
    if device == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return device
