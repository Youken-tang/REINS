"""Adapter package for wiring benchmark systems onto the Reins runtime.

Public surface
==============

* :data:`HERMES_TOOL_RESOURCE_TABLE` — static mapping of tool names to
  :class:`reins.runtime.resource_access.ResourceAccess` factories.

* :func:`resource_access_for` — dispatches a hermes-flavoured tool call
  (``name`` + ``args``) to a ``ResourceAccess``. Handles the three special
  tiers: path-scoped, terminal/shell, and default unknown.

* :func:`dispatch_tool_calls` — lowers a hermes-shaped ``tool_calls`` batch
  to Reins :class:`AgentTaskSpec` jobs and waits for the next delivery
  batch, returning per-call results in original order.

* :class:`TaskTrace` / :class:`ToolCall` / :class:`ToolResult` — execution
  trace shape consumed by every adapter and evaluator in the benchmark.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str = ""


@dataclass
class ToolResult:
    call_id: str = ""
    output: str = ""
    success: bool = False


@dataclass
class TaskTrace:
    """Execution trace produced by an agent adapter, consumed by evaluators."""

    task_id: str = ""
    agent_name: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    wall_time: float = 0.0
    task_seconds: float = 0.0
    success: bool = False
    error: str = ""
    final_answer: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    step_count: int = 0
    batch_count: int = 0
    conflict_count: int = 0
    peak_parallelism: int = 0
    parallelism_timeline: list[dict[str, Any]] = field(default_factory=list)
    planning_stall_seconds: float = 0.0
    streaming_dispatch_count: int = 0
    total_dispatch_count: int = 0
    model_calls: int = 0
    total_tokens: int = 0
    jsonl: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


from benchmark.adapters.hermes_reins_runtime import (  # noqa: E402,F401
    HERMES_TOOL_RESOURCE_TABLE,
    HermesToolCall,
    dispatch_tool_calls,
    resource_access_for,
)
from benchmark.adapters.hermes_reins_loop import (  # noqa: F401
    ENV_FLAG,
    InvokeTool,
    execute_tool_calls,
)
from benchmark.adapters.runtime_host import (  # noqa: F401
    HermesReinsHost,
    get_default_host,
    reset_default_host,
)

__all__ = [
    "ENV_FLAG",
    "HERMES_TOOL_RESOURCE_TABLE",
    "HermesReinsHost",
    "HermesToolCall",
    "InvokeTool",
    "TaskTrace",
    "ToolCall",
    "ToolResult",
    "dispatch_tool_calls",
    "execute_tool_calls",
    "get_default_host",
    "reset_default_host",
    "resource_access_for",
]
