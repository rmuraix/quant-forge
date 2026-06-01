"""Config loading: read YAML, apply overrides, validate with Pydantic."""

from __future__ import annotations

from pathlib import Path

import yaml

from quantforge.config.overrides import apply_overrides
from quantforge.config.schema import ExperimentConfig
from quantforge.utils.errors import ConfigError


def load_config(path: Path, overrides: list[str] | None = None) -> ExperimentConfig:
    """Load an ExperimentConfig from a YAML file with optional overrides.

    Args:
        path: Path to the YAML config file.
        overrides: List of 'key.path=value' override strings.

    Returns:
        Validated ExperimentConfig instance.
    """
    try:
        with open(path) as f:
            data: dict = yaml.safe_load(f) or {}
    except FileNotFoundError:
        raise ConfigError(f"Config file not found: {path}")
    except yaml.YAMLError as e:
        raise ConfigError(f"Failed to parse YAML config '{path}': {e}")

    if overrides:
        data = apply_overrides(data, overrides)

    try:
        return ExperimentConfig(**data)
    except Exception as e:
        raise ConfigError(f"Config validation error: {e}") from e
