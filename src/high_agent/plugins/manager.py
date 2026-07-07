"""Minimal high-agent native plugin manager."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None

from high_agent.config import get_config_paths
from high_agent.runtime.resource_access import ResourceAccess
from high_agent.tools.registry import ToolRegistry


@dataclass(frozen=True)
class PluginManifest:
    name: str
    path: Path
    enabled: bool = False
    tools: list[dict[str, Any]] = field(default_factory=list)
    commands: list[dict[str, Any]] = field(default_factory=list)
    hooks: list[dict[str, Any]] = field(default_factory=list)


class PluginManager:
    def __init__(self, plugin_root: str | Path | None = None, *, allow_disabled: bool = False) -> None:
        self.plugin_root = Path(plugin_root) if plugin_root is not None else get_config_paths().home / "plugins"
        self.allow_disabled = allow_disabled
        self.manifests: list[PluginManifest] = []

    def discover(self) -> list[Path]:
        if not self.plugin_root.exists():
            return []
        return sorted(path for path in self.plugin_root.glob("*/plugin.yaml") if path.is_file())

    def load(self) -> list[PluginManifest]:
        manifests: list[PluginManifest] = []
        for path in self.discover():
            data = _load_yaml(path)
            name = str(data.get("name") or path.parent.name)
            enabled = bool(data.get("enabled", False))
            if not enabled and not self.allow_disabled:
                continue
            manifests.append(
                PluginManifest(
                    name=name,
                    path=path,
                    enabled=enabled,
                    tools=list(data.get("tools") or []),
                    commands=list(data.get("commands") or []),
                    hooks=list(data.get("hooks") or []),
                )
            )
        self.manifests = manifests
        return list(self.manifests)

    def register_tools(self, registry: ToolRegistry) -> list[str]:
        names: list[str] = []
        for manifest in self.manifests:
            for tool in manifest.tools:
                name = str(tool.get("name") or "").strip()
                if not name:
                    continue
                description = str(tool.get("description") or f"Plugin tool from {manifest.name}")
                response = tool.get("response", {"ok": True, "plugin": manifest.name, "tool": name})
                access = _resource_access(str(tool.get("resource_access") or "external"))
                registry.register(
                    name=name,
                    schema={"description": description, "parameters": {"type": "object", "properties": {}}},
                    handler=lambda args, value=response: value,
                    resource_access=lambda args, root, value=access: value,
                )
                names.append(name)
        return names

    def register_commands(self, command_registry: Any, console: Any | None = None) -> list[str]:
        names: list[str] = []
        for manifest in self.manifests:
            for command in manifest.commands:
                name = str(command.get("name") or "").strip()
                if not name:
                    continue
                help_text = str(command.get("help") or f"Plugin command from {manifest.name}")
                response = str(command.get("response") or f"{manifest.name}:{name}")

                def _handler(args: list[str], text: str = response) -> None:
                    if console is not None:
                        console.print(text)
                    else:
                        print(text)
                    return None

                command_registry.register(name, _handler, help_text)
                names.append(name if name.startswith("/") else f"/{name}")
        return names

    def run_hooks(self, hook_name: str, payload: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for manifest in self.manifests:
            for hook in manifest.hooks:
                if str(hook.get("name") or "") == hook_name:
                    results.append({"plugin": manifest.name, "hook": hook_name, "payload": payload or {}, "ok": True})
        return results


def _resource_access(kind: str) -> ResourceAccess:
    if kind == "none":
        return ResourceAccess.empty()
    if kind == "read":
        return ResourceAccess.read("plugin:read")
    if kind == "write":
        return ResourceAccess.write("plugin:write")
    return ResourceAccess(unknown=True, side_effect_level="external")


def _load_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        data = yaml.safe_load(text) or {}
    else:
        import json

        data = json.loads(text or "{}")
    return data if isinstance(data, dict) else {}
