"""Configuration loader for YAML files with Pydantic validation."""

from pathlib import Path

import yaml

from tinyvit.config import Config


def load_config(config_path: Path) -> Config:
    """Load and validate configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Validated Config object.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        ValidationError: If the config doesn't match the schema.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
        if config_dict is None:
            raise ValueError(f"Config file is empty: {config_path}")

    return Config(**config_dict)
