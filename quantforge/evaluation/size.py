"""Model size utilities."""

from __future__ import annotations

import io

import torch.nn as nn


def get_model_size_mb(model: nn.Module) -> float:
    """Return the model size in megabytes (via state_dict serialization)."""
    buf = io.BytesIO()
    import torch

    torch.save(model.state_dict(), buf)
    return buf.tell() / (1024 * 1024)
