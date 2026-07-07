"""Tool registry with required resource metadata."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from high_agent.runtime.resource_access import ResourceAccess
from high_agent.runtime.types import AgentTaskSpec, TaskContext, TaskResult
from high_agent.tools.result_store import ToolResultStore

ToolHandler = Callable[[dict[str, Any]], Any]
ResourceFn = Callable[[dict[str, Any], str | None], ResourceAccess]


@dataclass(frozen=True)
class ToolEntry:
    name: str
    schema: dict[str, Any]
    handler: ToolHandler
    resource_access: ResourceFn
    barrier: str = "none"
    max_result_size_chars: int = 100_000


class ToolRegistry:
    def __init__(self, *, result_store: ToolResultStore | None = None) -> None:
        self._tools: dict[str, ToolEntry] = {}
        self.result_store = result_store

    def register(self, *, name: str, schema: dict[str, Any], handler: ToolHandler,
                 resource_access: ResourceFn, barrier: str = "none",
                 max_result_size_chars: int = 100_000,
                 override: bool = False) -> None:
        if resource_access is None:
            raise ValueError(f"tool {name} must declare resource_access")
        # previously a duplicate register() call silently overwrote
        # the existing entry. Two plugins / extensions registering the same
        # name (or a typo collision) replaced one binding with the other,
        # making the failure mode invisible until the wrong handler ran.
        # Default to refusing the overwrite; callers that legitimately want
        # to swap a binding (test fixtures, hot-reload) must pass
        # override=True to opt in.
        if name in self._tools and not override:
            raise ValueError(
                f"tool {name!r} already registered; pass override=True to replace it"
            )
        self._tools[name] = ToolEntry(
            name=name,
            schema={**schema, "name": name},
            handler=handler,
            resource_access=resource_access,
            barrier=barrier,
            max_result_size_chars=max_result_size_chars,
        )

    def get(self, name: str) -> ToolEntry:
        return self._tools[name]

    def definitions(self) -> list[dict[str, Any]]:
        return [{"type": "function", "function": entry.schema} for entry in self._tools.values()]

    def names(self) -> list[str]:
        return sorted(self._tools)

    def filtered(self, names: set[str]) -> "ToolRegistry":
        registry = ToolRegistry(result_store=self.result_store)
        registry._tools = {name: entry for name, entry in self._tools.items() if name in names}
        return registry

    def task_from_call(self, name: str, args: dict[str, Any], *,
                       workspace_root: str | None = None,
                       task_id: str | None = None) -> AgentTaskSpec:
        entry = self.get(name)
        access = entry.resource_access(args, workspace_root).normalized(workspace_root)

        def _handler(ctx: TaskContext) -> TaskResult:
            try:
                handler_args = dict(args)
                if workspace_root is not None:
                    handler_args["_workspace_root"] = workspace_root
                raw = entry.handler(handler_args)
                if isinstance(raw, TaskResult):
                    return raw
                text = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
                if len(text) > entry.max_result_size_chars:
                    if self.result_store is not None:
                        result_id = self.result_store.put(text, summary=f"{entry.name} result stored outside context")
                        text = json.dumps(
                            {
                                "summary": f"{entry.name} output exceeded context limit",
                                "result_id": result_id,
                                "preview": text[:2000],
                                "truncated": True,
                            },
                            ensure_ascii=False,
                        )
                    else:
                        text = text[:entry.max_result_size_chars] + "\n[truncated]"
                return TaskResult.completed(text)
            except Exception as exc:
                # surface the active workspace_root in the error
                # summary. Tool errors like ``FileNotFoundError`` /
                # ``PermissionError`` only show the offending path; without
                # the root the planner cannot tell a typo (`/home/Youken`
                # vs `/home/youken`) from a genuinely missing target. See
                message = str(exc)
                if workspace_root is not None and "workspace_root=" not in message:
                    message = f"{message} (workspace_root={workspace_root})"
                return TaskResult.failed(message, error_type=type(exc).__name__)

        return AgentTaskSpec(
            kind="tool",
            goal=f"{name}({', '.join(sorted(args))})",
            input={"name": name, "args": dict(args)},
            reads=set(access.reads),
            writes=set(access.writes),
            resource_access=access,
            barrier=entry.barrier,  # type: ignore[arg-type]
            task_id=task_id or f"tool-{name}",
            handler=_handler,
        )
