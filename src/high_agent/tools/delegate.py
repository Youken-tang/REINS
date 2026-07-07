"""delegate_task tool — lowers goal+tasks into worker AgentTaskSpecs."""

from __future__ import annotations

from typing import Any

from high_agent.runtime.resource_access import ResourceAccess, normalize_component
from high_agent.runtime.types import AgentTaskSpec, TaskResult, new_id
from high_agent.tools.registry import ToolRegistry


DELEGATE_TASK_SCHEMA = {
    "description": (
        "Delegate a goal to one or more worker agents. Each task runs independently "
        "and in parallel (subject to resource conflicts). "
        "Use mode='worker' for one-shot subtasks: the worker runs a single model turn "
        "and may issue tool_calls (e.g. write_file, read_file), but it does not iterate "
        "or react to those tool results. Batch every tool call you need into that one turn. "
        "Use mode='sub_agent' for multi-step subtasks that need to plan, observe tool results, "
        "and then call more tools — sub_agent runs an inner agent loop until it returns a final summary."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": "High-level goal for this delegation batch.",
            },
            "mode": {
                "type": "string",
                "enum": ["worker", "sub_agent"],
                "description": (
                    "Execution mode. 'worker' = single model turn that may emit tool_calls "
                    "(no iteration on tool results). 'sub_agent' = inner agent loop that plans, "
                    "observes tool results, and calls more tools across multiple turns."
                ),
                "default": "worker",
            },
            "tasks": {
                "type": "array",
                "description": "Individual worker tasks to execute in parallel.",
                "items": {
                    "type": "object",
                    "properties": {
                        "goal": {"type": "string", "description": "Specific goal for this worker."},
                        "input": {"type": "string", "description": "Input data or instructions."},
                        "reads": {"type": "array", "items": {"type": "string"}, "description": "File paths this worker reads."},
                        "writes": {"type": "array", "items": {"type": "string"}, "description": "File paths this worker writes."},
                    },
                    "required": ["goal"],
                },
            },
        },
        "required": ["goal", "tasks"],
    },
}


def delegate_task_resource_access(args: dict[str, Any], workspace_root: str | None) -> ResourceAccess:
    """delegate_task itself completes immediately (emits discovered_tasks).
    Resource claims live on individual worker tasks, so the parent is empty."""
    return ResourceAccess.empty()


def delegate_task_handler(args: dict[str, Any]) -> TaskResult:
    """Lower delegate_task into multiple kind='worker' or 'agent_loop' AgentTaskSpecs.

    Returns TaskResult.completed with discovered_tasks so the runtime
    schedules workers. The delegate_task itself delivers immediately.

 mode='sub_agent' now lowers to kind='agent_loop' (the async
    suspend/resume variant introduced in v11-C7) instead of the legacy
    kind='sub_agent'. The schema-level ``mode='sub_agent'`` name is kept
    so the model-facing tool contract is unchanged; only the runtime kind
    routes through the new handler. v11-C10 retires the legacy kind.
    """
    batch_goal = str(args.get("goal") or "delegated work")
    tasks_spec = args.get("tasks") or []
    workspace_root = args.get("_workspace_root")
    mode = str(args.get("mode") or "worker")
    child_depth = int(args.get("_depth", 0))
    if mode not in ("worker", "sub_agent"):
        mode = "worker"
    task_kind = "agent_loop" if mode == "sub_agent" else "worker"

    if not isinstance(tasks_spec, list) or not tasks_spec:
        return TaskResult.failed(
            "delegate_task requires at least one task in the tasks array",
            error_type="validation_error",
        )

    discovered: list[AgentTaskSpec] = []
    for i, task_def in enumerate(tasks_spec):
        if not isinstance(task_def, dict):
            continue
        worker_goal = str(task_def.get("goal") or f"subtask-{i}")
        worker_input: dict[str, Any] = {
            "input": str(task_def.get("input") or ""),
            "parent_goal": batch_goal,
        }

        reads: frozenset[str] = frozenset()
        writes: frozenset[str] = frozenset()
        if workspace_root:
            if task_def.get("reads"):
                reads = frozenset(
                    normalize_component(f"file:{r}", workspace_root)
                    for r in task_def["reads"]
                )
            if task_def.get("writes"):
                writes = frozenset(
                    normalize_component(f"file:{w}", workspace_root)
                    for w in task_def["writes"]
                )

        resource_access = ResourceAccess(
            reads=reads,
            writes=writes,
            side_effect_level="local" if writes else "none",
        ) if reads or writes else ResourceAccess.empty()

        discovered.append(AgentTaskSpec(
            kind=task_kind,
            goal=worker_goal,
            input=worker_input,
            resource_access=resource_access,
            task_id=new_id("worker" if task_kind == "worker" else "agentloop"),
            deliverable=True,
            metadata={"depth": child_depth},
        ))

    return TaskResult.completed(
        f"Delegated {len(discovered)} worker tasks for: {batch_goal}",
        discovered_tasks=discovered,
    )


def register_delegate_task(registry: ToolRegistry) -> None:
    """Register delegate_task into an existing ToolRegistry."""
    registry.register(
        name="delegate_task",
        schema=DELEGATE_TASK_SCHEMA,
        handler=lambda args: delegate_task_handler(args),
        resource_access=delegate_task_resource_access,
    )
