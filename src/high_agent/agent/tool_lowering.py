"""Shared tool_call → AgentTaskSpec lowering used by AgentLoop and Worker.

Both ``agent/loop.py`` (multi-step agent_loop_step resume path) and
``agent/worker.py`` (single-turn delegated worker) need to take a model's
``response.tool_calls`` and turn each into an ``AgentTaskSpec`` that the
runtime can schedule. The logic is identical:

1. Normalize via ``ToolCallNormalizer`` so ``functions.foo`` /
   ``multi_tool_use.parallel`` / ``write_many_files`` collapse into a flat
   list of single tool calls.
2. Validate each name against the registry; emit a synthetic failure
   ``AgentTaskSpec`` for unknown tools or unparseable arguments rather than
   silently dropping them, so the model still sees a tool result.
3. For ``delegate_task``, inject ``_depth = depth + 1`` so the recursion
   guard in ``agent_loop`` survives across handoffs.
4. Lower via ``ToolRegistry.task_from_call``, then attach metadata
   (``tool_call_id``, ``tool_name``, ``discovered_by``).

This module deliberately does NOT know about ``MainAgent._tool_result_groups``
or the ``_active_action_index`` dedupe map — those are MainAgent-specific
concerns that live above the shared lowering layer.
"""

from __future__ import annotations

from typing import Any

from high_agent.agent.tool_calls import ToolCallNormalizer
from high_agent.llm.types import ToolCall
from high_agent.runtime.types import AgentTaskSpec, TaskContext, TaskResult, new_id
from high_agent.tools.registry import ToolRegistry


def _failed_task(call_id: str, tool_name: str, message: str, *, error_type: str = "tool_lowering_error") -> AgentTaskSpec:
    def _handle(ctx: TaskContext) -> TaskResult:
        return TaskResult.failed(message, error_type=error_type)

    return AgentTaskSpec(
        kind="tool",
        goal=f"{tool_name} failed: {message}",
        task_id=new_id(f"tool-error-{call_id}"),
        handler=_handle,
        metadata={"tool_call_id": call_id, "tool_name": tool_name},
    )


def lower_tool_calls_to_specs(
    tool_calls: list[ToolCall],
    *,
    tools: ToolRegistry,
    workspace_root: str | None,
    parent_task_id: str,
    depth: int,
    task_id_prefix: str = "loop",
) -> list[AgentTaskSpec]:
    """Lower a model's tool_calls into runtime-ready AgentTaskSpecs.

    ``parent_task_id`` is recorded on each spec's ``metadata.discovered_by``
    so the ledger can link children back to the dispatching task.
    ``task_id_prefix`` controls the synthetic task_id namespace
    ("loop-<call_id>" for agent_loop, "worker-<call_id>" for worker).
    """
    normalizer = ToolCallNormalizer()
    normalized = normalizer.normalize(tool_calls)
    out: list[AgentTaskSpec] = []
    valid_tools = set(tools.names())

    for item in normalized:
        call = item.call
        call_id = call.id or new_id("call")
        try:
            args = call.args_dict()
        except Exception:
            out.append(_failed_task(call_id, call.name, "invalid arguments"))
            continue
        if call.name not in valid_tools:
            out.append(_failed_task(call_id, call.name, f"unknown tool: {call.name}"))
            continue
        if call.name == "delegate_task":
            args["_depth"] = depth + 1
        try:
            task = tools.task_from_call(
                call.name,
                args,
                workspace_root=workspace_root,
                task_id=f"{task_id_prefix}-{call_id}",
            )
        except Exception as exc:
            out.append(_failed_task(call_id, call.name, f"lowering failed: {exc}"))
            continue
        task.metadata.update(
            {
                "tool_call_id": call_id,
                "tool_name": call.name,
                "discovered_by": parent_task_id,
            }
        )
        out.append(task)
    return out
