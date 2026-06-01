"""Latency measurement utilities."""

from __future__ import annotations

import time

import torch
import torch.nn as nn


def measure_latency(
    model: nn.Module,
    input_shape: tuple[int, ...],
    device: str,
    warmup: int = 10,
    steps: int = 50,
) -> dict[str, float]:
    """Measure model inference latency.

    Args:
        model: Model to benchmark.
        input_shape: Input tensor shape (N, C, H, W).
        device: Device string.
        warmup: Number of warmup iterations.
        steps: Number of measured iterations.

    Returns:
        dict with 'latency_ms' and 'throughput_images_per_sec'.
    """
    model.eval()
    dummy = torch.randn(*input_shape, device=device)
    batch_size = input_shape[0]

    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(steps):
            model(dummy)

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    latency_ms = (elapsed / steps) * 1000
    throughput = (steps * batch_size) / elapsed

    return {
        "latency_ms": latency_ms,
        "throughput_images_per_sec": throughput,
        "batch_size": batch_size,
        "warmup_steps": warmup,
        "measured_steps": steps,
        "device": device,
    }
