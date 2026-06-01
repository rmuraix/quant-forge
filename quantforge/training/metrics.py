"""Training metrics utilities."""

from __future__ import annotations

import torch


def top_k_accuracy(output: torch.Tensor, target: torch.Tensor, k: int = 1) -> float:
    """Compute top-k accuracy.

    Args:
        output: Model logits of shape (N, C).
        target: Ground truth labels of shape (N,).
        k: Top-k value.

    Returns:
        Top-k accuracy as a float in [0, 1].
    """
    with torch.no_grad():
        batch_size = target.size(0)
        if batch_size == 0:
            return 0.0
        _, pred = output.topk(min(k, output.size(1)), dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[:k].reshape(-1).float().sum(0)
        return (correct_k / batch_size).item()


def compute_metrics(output: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Compute top-1 and top-5 accuracy.

    Returns:
        dict with 'top1' and 'top5' keys.
    """
    num_classes = output.size(1)
    return {
        "top1": top_k_accuracy(output, target, k=1),
        "top5": top_k_accuracy(output, target, k=min(5, num_classes)),
    }
