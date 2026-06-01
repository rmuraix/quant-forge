"""Manifest building utilities."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def build_manifest(
    run_id: str,
    run_dir: Path,
    command: str,
    extra: dict | None = None,
) -> dict:
    """Build a manifest dict for the run."""
    manifest = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "command": command,
        "timestamp": datetime.now().isoformat(),
        "files": {},
    }
    if extra:
        manifest.update(extra)
    return manifest
