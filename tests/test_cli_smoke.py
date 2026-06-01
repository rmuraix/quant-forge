"""CLI smoke tests - test that commands are importable and help works."""

from __future__ import annotations

import subprocess
import sys


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "typer", "quantforge.cli.app:app", "run", *args],
        capture_output=True,
        text=True,
    )


def test_cli_importable() -> None:
    """Verify the CLI app can be imported without errors."""
    from quantforge.cli.app import app

    assert app is not None


def test_app_has_commands() -> None:
    """Verify expected commands are registered."""
    from quantforge.cli.app import app

    command_names = {cmd.name for cmd in app.registered_commands}
    assert "init" in command_names
    assert "train" in command_names
    assert "qat" in command_names
    assert "eval" in command_names
    assert "infer" in command_names
    assert "quantize" in command_names


def test_inspect_subapp_registered() -> None:
    """Verify inspect subapp is registered."""
    from quantforge.cli.app import inspect_app

    assert inspect_app is not None
    inspect_command_names = {cmd.name for cmd in inspect_app.registered_commands}
    assert "config" in inspect_command_names
    assert "dataset" in inspect_command_names
    assert "model" in inspect_command_names


def test_errors_importable() -> None:
    from quantforge.utils.errors import (
        ConfigError,
        DatasetColumnError,
        DatasetError,
        QuantForgeError,
    )

    assert issubclass(ConfigError, QuantForgeError)
    assert issubclass(DatasetColumnError, DatasetError)
    assert issubclass(DatasetError, QuantForgeError)


def test_schema_importable() -> None:
    from quantforge.config.schema import ExperimentConfig

    cfg = ExperimentConfig()
    assert cfg.version == 1
    assert cfg.project.seed == 42


def test_quantization_registry() -> None:
    from quantforge.quantization.registry import get_quantizer
    from quantforge.config.schema import QuantizationConfig

    cfg = QuantizationConfig(enabled=False, backend="none", mode="none")
    q = get_quantizer(cfg)
    assert q is not None


def test_artifact_writer_setup(tmp_path) -> None:
    from quantforge.artifacts.writer import ArtifactWriter

    writer = ArtifactWriter(tmp_path / "run")
    writer.setup()
    assert (tmp_path / "run" / "checkpoints").is_dir()
    assert (tmp_path / "run" / "quantized").is_dir()
    assert (tmp_path / "run" / "metrics").is_dir()
    assert (tmp_path / "run" / "predictions").is_dir()
    assert (tmp_path / "run" / "logs").is_dir()
