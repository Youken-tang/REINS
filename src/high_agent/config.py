"""Configuration and secret storage for high-agent."""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal envs
    yaml = None

CONFIG_ENV_PREFIX = "HIGH_AGENT_"


@dataclass(frozen=True)
class ConfigPaths:
    home: Path
    config_path: Path
    secrets_path: Path


def get_config_paths(env: dict[str, str] | None = None) -> ConfigPaths:
    values = env or os.environ
    home_raw = values.get("HIGH_AGENT_HOME", "").strip()
    home = Path(home_raw).expanduser() if home_raw else Path.home() / ".config" / "high-agent"
    return ConfigPaths(
        home=home,
        config_path=home / "config.yaml",
        secrets_path=home / "secrets.yaml",
    )


def load_config(paths: ConfigPaths | None = None) -> dict[str, Any]:
    paths = paths or get_config_paths()
    data = _load_yaml(paths.config_path)
    return data if isinstance(data, dict) else {}


def save_config(config: dict[str, Any], paths: ConfigPaths | None = None) -> None:
    paths = paths or get_config_paths()
    _save_yaml(paths.config_path, config, mode=0o644)


def load_secrets(paths: ConfigPaths | None = None) -> dict[str, Any]:
    paths = paths or get_config_paths()
    data = _load_yaml(paths.secrets_path)
    return data if isinstance(data, dict) else {}


def save_secrets(secrets: dict[str, Any], paths: ConfigPaths | None = None) -> None:
    paths = paths or get_config_paths()
    _save_yaml(paths.secrets_path, secrets, mode=0o600)


def get_nested(data: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def set_nested(data: dict[str, Any], keys: tuple[str, ...], value: Any) -> None:
    current = data
    for key in keys[:-1]:
        child = current.get(key)
        if not isinstance(child, dict):
            child = {}
            current[key] = child
        current = child
    current[keys[-1]] = value


def _load_yaml(path: Path) -> Any:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        text = handle.read()
    if yaml is not None:
        return yaml.safe_load(text) or {}
    return json.loads(text or "{}")


def _save_yaml(path: Path, data: dict[str, Any], *, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        if yaml is not None:
            yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False)
        else:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    os.chmod(path, mode)
