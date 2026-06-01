"""Tests for config override parsing."""

from __future__ import annotations

import pytest

from quantforge.config.overrides import apply_overrides, parse_override, parse_value
from quantforge.utils.errors import ConfigError


def test_parse_bool_true() -> None:
    assert parse_value("true") is True
    assert parse_value("True") is True


def test_parse_bool_false() -> None:
    assert parse_value("false") is False
    assert parse_value("False") is False


def test_parse_null() -> None:
    assert parse_value("null") is None
    assert parse_value("none") is None
    assert parse_value("None") is None
    assert parse_value("~") is None


def test_parse_int() -> None:
    assert parse_value("42") == 42
    assert isinstance(parse_value("42"), int)


def test_parse_float() -> None:
    val = parse_value("3.14")
    assert abs(val - 3.14) < 1e-9
    assert isinstance(val, float)


def test_parse_string() -> None:
    assert parse_value("resnet18") == "resnet18"
    assert isinstance(parse_value("resnet18"), str)


def test_parse_override_basic() -> None:
    key, val = parse_override("training.epochs=10")
    assert key == "training.epochs"
    assert val == 10


def test_parse_override_invalid() -> None:
    with pytest.raises(ConfigError):
        parse_override("no_equals_sign")


def test_apply_overrides_nested() -> None:
    data: dict = {}
    result = apply_overrides(data, ["training.epochs=5", "model.name=resnet50"])
    assert result["training"]["epochs"] == 5
    assert result["model"]["name"] == "resnet50"


def test_apply_overrides_existing() -> None:
    data = {"training": {"epochs": 10, "lr": 0.001}}
    result = apply_overrides(data, ["training.epochs=20"])
    assert result["training"]["epochs"] == 20
    assert abs(result["training"]["lr"] - 0.001) < 1e-9


def test_apply_overrides_deep() -> None:
    data: dict = {}
    result = apply_overrides(data, ["a.b.c=42"])
    assert result["a"]["b"]["c"] == 42
