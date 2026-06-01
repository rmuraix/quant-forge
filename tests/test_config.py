"""Tests for config loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from quantforge.config.loader import load_config
from quantforge.config.schema import ExperimentConfig
from quantforge.utils.errors import ConfigError


@pytest.fixture
def tmp_config(tmp_path: Path) -> Path:
    """Write a minimal valid config to a temp file."""
    cfg = {
        "version": 1,
        "project": {"name": "test-exp", "output_dir": "outputs", "seed": 0},
        "model": {"name": "resnet18"},
        "dataset": {
            "name": "cifar10",
            "split": {"train": "train", "validation": "test"},
        },
        "tracking": {"backend": "wandb", "mode": "disabled"},
    }
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(cfg))
    return p


def test_load_config_basic(tmp_config: Path) -> None:
    cfg = load_config(tmp_config)
    assert isinstance(cfg, ExperimentConfig)
    assert cfg.project.name == "test-exp"
    assert cfg.model.name == "resnet18"


def test_load_config_with_overrides(tmp_config: Path) -> None:
    cfg = load_config(tmp_config, overrides=["training.epochs=5", "training.lr=0.001"])
    assert cfg.training.epochs == 5
    assert abs(cfg.training.lr - 0.001) < 1e-9


def test_load_config_not_found(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nonexistent.yaml")


def test_quantization_enabled_requires_backend(tmp_config: Path) -> None:
    with pytest.raises(ConfigError):
        load_config(
            tmp_config,
            overrides=[
                "quantization.enabled=true",
                "quantization.backend=none",
                "quantization.mode=none",
            ],
        )


def test_quantization_disabled_requires_mode_none(tmp_config: Path) -> None:
    with pytest.raises(ConfigError):
        load_config(
            tmp_config,
            overrides=[
                "quantization.enabled=false",
                "quantization.mode=qat",
            ],
        )


def test_quantization_qat_requires_torchao(tmp_config: Path) -> None:
    with pytest.raises(ConfigError):
        load_config(
            tmp_config,
            overrides=[
                "quantization.enabled=true",
                "quantization.backend=other",
                "quantization.mode=qat",
            ],
        )


def test_tracking_backend_must_be_wandb(tmp_config: Path) -> None:
    with pytest.raises(ConfigError):
        load_config(tmp_config, overrides=["tracking.backend=mlflow"])


def test_default_values(tmp_config: Path) -> None:
    cfg = load_config(tmp_config)
    assert cfg.version == 1
    assert cfg.training.epochs == 10
    assert cfg.training.optimizer == "adamw"
    assert cfg.quantization.enabled is False
    assert cfg.quantization.backend == "none"
    assert cfg.quantization.mode == "none"


def test_override_bool_types(tmp_config: Path) -> None:
    cfg = load_config(
        tmp_config, overrides=["training.amp=false", "training.deterministic=true"]
    )
    assert cfg.training.amp is False
    assert cfg.training.deterministic is True


def test_override_null(tmp_config: Path) -> None:
    cfg = load_config(tmp_config, overrides=["model.num_classes=null"])
    assert cfg.model.num_classes is None


def test_valid_qat_config(tmp_config: Path) -> None:
    cfg = load_config(
        tmp_config,
        overrides=[
            "quantization.enabled=true",
            "quantization.backend=torchao",
            "quantization.mode=qat",
        ],
    )
    assert cfg.quantization.enabled is True
    assert cfg.quantization.backend == "torchao"
    assert cfg.quantization.mode == "qat"
