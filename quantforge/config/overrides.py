"""Config override parsing: --set key.path=value."""

from __future__ import annotations

from quantforge.utils.errors import ConfigError


def parse_value(raw: str) -> object:
    """Parse a string value into a Python scalar."""
    if raw.lower() in ("true",):
        return True
    if raw.lower() in ("false",):
        return False
    if raw.lower() in ("null", "none", "~"):
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def parse_override(kv: str) -> tuple[str, object]:
    """Parse 'key.path=value' into (dot_path, typed_value)."""
    if "=" not in kv:
        raise ConfigError(f"Invalid override '{kv}': must be in 'key=value' form.")
    key, _, raw = kv.partition("=")
    return key.strip(), parse_value(raw.strip())


def apply_overrides(data: dict, overrides: list[str]) -> dict:
    """Apply list of 'key.path=value' overrides to a nested dict."""
    for kv in overrides:
        path, value = parse_override(kv)
        keys = path.split(".")
        target = data
        for k in keys[:-1]:
            if k not in target or not isinstance(target[k], dict):
                target[k] = {}
            target = target[k]
        target[keys[-1]] = value
    return data
