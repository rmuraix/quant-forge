"""Inference output formatting."""

from __future__ import annotations


def format_predictions(predictions: list[dict]) -> str:
    """Format a list of predictions for console output.

    Args:
        predictions: List of prediction dicts with 'image', 'label', 'confidence'.

    Returns:
        Formatted string.
    """
    lines = []
    for pred in predictions:
        image = pred.get("image", "unknown")
        label = pred.get("label", "unknown")
        conf = pred.get("confidence", 0.0)
        lines.append(f"  {image}: {label} ({conf:.1%})")
    return "\n".join(lines)
