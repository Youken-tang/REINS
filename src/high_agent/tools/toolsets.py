"""Toolset registry for project-building workflows."""

from __future__ import annotations

from dataclasses import dataclass, field

from high_agent.approval import ApprovalManager
from high_agent.tools.core import create_core_registry
from high_agent.tools.registry import ToolRegistry
from high_agent.tools.result_store import ToolResultStore


TOOLSETS: dict[str, set[str]] = {
    "file": {"read_file", "read_many_files", "write_file", "write_many_files", "append_file", "list_dir", "list_tree", "mkdir", "move_path", "delete_path"},
    "edit": {"replace_in_file", "patch_file"},
    "search": {"search_files", "http_fetch"},
    "terminal": {"terminal"},
    "todo": {"todo_write", "todo_read"},
    "code": {"run_python", "run_tests"},
    "browser-lite": {"http_fetch"},
    "mcp": {"mcp_call"},
    "base": {"noop", "sleep"},
    "delegate": {"delegate_task"},
}

DEFAULT_TOOLSETS = {"base", "file", "edit", "search", "terminal", "todo", "code", "browser-lite", "delegate"}


@dataclass
class ToolsetRegistry:
    enabled: set[str] = field(default_factory=lambda: set(DEFAULT_TOOLSETS))
    allow_terminal: bool = False
    allow_outside_workspace: bool = False
    approval_manager: ApprovalManager | None = None
    result_store: ToolResultStore | None = None

    def enable(self, name: str) -> None:
        if name not in TOOLSETS:
            raise KeyError(f"unknown toolset: {name}")
        self.enabled.add(name)

    def disable(self, name: str) -> None:
        self.enabled.discard(name)

    def names(self) -> list[str]:
        return sorted(TOOLSETS)

    def tool_names(self) -> set[str]:
        out: set[str] = set()
        for name in self.enabled:
            out.update(TOOLSETS.get(name, set()))
        return out

    def registry(self) -> ToolRegistry:
        full = create_core_registry(
            allow_terminal=self.allow_terminal,
            allow_outside_workspace=self.allow_outside_workspace,
            approval_manager=self.approval_manager,
            result_store=self.result_store,
        )
        return full.filtered(self.tool_names())

    def definitions(self) -> list[dict]:
        return self.registry().definitions()
