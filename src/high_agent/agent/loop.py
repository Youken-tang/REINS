""" agent_loop / agent_loop_step kinds with suspend-on-LLM.

v11-C10 retired the legacy SubAgent in-thread loop and the batched
sub_agent / sub_agent_step kinds entirely. AgentLoop is the canonical
multi-step delegate executor: a continuation-passing form built on the
suspend/resume protocol introduced in C3-C5 and the non-blocking
``ModelClient.complete_async`` introduced in C6.

Each agent_loop_step iteration breaks into three phases:

1. **Aggregate prev results** — read previously dispatched children's
   ``TaskResult`` via ``ctx.runtime.collect(...)`` and roll their summaries
   back into the message history. The roll-up logic operates on the new
   ``AgentLoopState`` envelope.
2. **Suspend on LLM** — call ``model_client.complete_async(...)`` to obtain
   a ``concurrent.futures.Future``, register it via
   ``ctx.runtime.register_future(future)`` to obtain a token, and return
   ``TaskResult.suspended(awaiting=[future_done(token)], snapshot=state)``.
   The worker thread is released; the IO loop completes the future and
   the scheduler resumes the task on the next ready slot.
3. **Resume → lower → next step** — the resume_handler reads the response
   from the future, lowers tool_calls into ``AgentTaskSpec`` children via
   ``ctx.runtime.add_discovered``, and emits the next ``agent_loop_step``
   task whose ``dependencies=[task_completed(c) for c in children]``.

Design properties (vs. the retired sub_agent_step):

- The LLM HTTP RTT no longer occupies a worker thread (was the residual
  v10 P3 gap). The worker pool can run other tools / steps in parallel.
- Context isolation is explicit: ``AgentLoopState`` serializes messages,
  submitted_ids, depth, iteration, deadline, and system prompt through
  ``AgentTaskSpec.input["loop_state"]``. No ``AgentContext`` is shared
  between parent and child agent_loops.
- invariants (timeout / max_iterations → ``TaskResult.failed``
  with ``error_type`` set; never reported as completed) are preserved.

 for the public
surface and for the
runtime semantics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any

from high_agent.agent.prompt_policy import workspace_context_snippet
from high_agent.agent.tool_calls import sanitize_tool_protocol_messages
from high_agent.agent.tool_lowering import lower_tool_calls_to_specs
from high_agent.llm.client import ModelClient
from high_agent.runtime.types import (
    AgentTaskSpec,
    DependencyPredicate,
    TaskContext,
    TaskHandler,
    TaskResult,
    new_id,
)
from high_agent.tools.registry import ToolRegistry


_AGENT_LOOP_SYSTEM_PROMPT = (
    "You are a focused sub-agent executing a specific goal within a larger task. "
    "Use the provided tools to complete your objective. "
    "Return multiple tool calls in one response when they have no dependencies. "
    "When all work is done, provide a concise summary of what was accomplished."
)


@dataclass
class AgentLoopState:
    """Serializable state carried across agent_loop_step iterations.

    Travels through ``AgentTaskSpec.input["loop_state"]`` between steps so
    the scheduler can drop and resume each iteration on different worker
    threads without retaining any python-level instance.
    """

    objective: str
    parent_task_id: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    submitted_ids: list[str] = field(default_factory=list)
    last_round_call_meta: list[dict[str, Any]] = field(default_factory=list)
    iteration: int = 0
    deadline_monotonic: float = 0.0
    # defaults bumped 8→16 iterations / 120s→240s. The previous
    # ceiling routinely tripped on multi-file scaffolds (~138 of 187
    # failures in the 2026-05 e2e run hit ``agent_loop_max_iterations``
    # before the sub-agent could finish lowering its plan). Operators can
    # still override per-call via ``AgentLoopState`` or globally via
    # ``agent.agent_loop_max_iterations`` / ``agent.agent_loop_timeout_seconds``
    # in config.yaml.
    max_iterations: int = 16
    timeout_seconds: float = 240.0
    depth: int = 0
    system_prompt: str = _AGENT_LOOP_SYSTEM_PROMPT

    def to_input(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "parent_task_id": self.parent_task_id,
            "messages": self.messages,
            "submitted_ids": list(self.submitted_ids),
            "last_round_call_meta": list(self.last_round_call_meta),
            "iteration": self.iteration,
            "deadline_monotonic": self.deadline_monotonic,
            "max_iterations": self.max_iterations,
            "timeout_seconds": self.timeout_seconds,
            "depth": self.depth,
            "system_prompt": self.system_prompt,
        }

    @staticmethod
    def from_input(payload: dict[str, Any]) -> "AgentLoopState":
        return AgentLoopState(
            objective=str(payload.get("objective") or ""),
            parent_task_id=str(payload.get("parent_task_id") or ""),
            messages=list(payload.get("messages") or []),
            submitted_ids=list(payload.get("submitted_ids") or []),
            last_round_call_meta=list(payload.get("last_round_call_meta") or []),
            iteration=int(payload.get("iteration") or 0),
            deadline_monotonic=float(payload.get("deadline_monotonic") or 0.0),
            max_iterations=int(payload.get("max_iterations") or 16),
            timeout_seconds=float(payload.get("timeout_seconds") or 240.0),
            depth=int(payload.get("depth") or 0),
            system_prompt=str(
                payload.get("system_prompt") or _AGENT_LOOP_SYSTEM_PROMPT
            ),
        )


def _build_initial_prompt(objective: str) -> str:
    return f"Goal: {objective}"


def _build_assistant_message(
    response: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    assistant_msg: dict[str, Any] = {
        "role": "assistant",
        "content": response.content or "",
    }
    call_meta: list[dict[str, Any]] = []
    if response.tool_calls:
        rendered_calls: list[dict[str, Any]] = []
        for tc in response.tool_calls:
            call_id = tc.id or new_id("call")
            rendered_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
            )
            call_meta.append({"tool_call_id": call_id, "tool_name": tc.name})
        assistant_msg["tool_calls"] = rendered_calls
    return assistant_msg, call_meta


def _failed_task(call_id: str, tool_name: str, message: str) -> AgentTaskSpec:
    def _handle(ctx: TaskContext) -> TaskResult:
        return TaskResult.failed(message, error_type="agent_loop_tool_error")

    return AgentTaskSpec(
        kind="tool",
        goal=f"{tool_name} failed: {message}",
        task_id=f"loop-err-{call_id}",
        handler=_handle,
        metadata={"tool_call_id": call_id, "tool_name": tool_name},
    )


def _append_results_to_messages(
    messages: list[dict[str, Any]],
    last_round_task_ids: list[str],
    call_meta: list[dict[str, Any]],
    results: dict[str, TaskResult],
) -> None:
    """Roll prev round's tool results into the message history.

    ``call_meta`` mirrors the assistant message's tool_calls in submission
    order; ``last_round_task_ids`` is the same length and order. Pairing
    is positional, so the caller must keep both arrays aligned.
    """
    pairs = list(zip(last_round_task_ids, call_meta))
    for tid, meta in pairs:
        out_call_id = meta.get("tool_call_id", tid)
        out_tool_name = meta.get("tool_name", "tool")
        result = results.get(tid)
        if result is None or not isinstance(result, TaskResult):
            content = "no result (timeout)"
        elif result.status == "completed":
            content = result.summary or "completed"
        else:
            content = f"[{result.status}] {result.summary}"
        messages.append(
            {
                "role": "tool",
                "tool_call_id": out_call_id,
                "name": out_tool_name,
                "content": content[:4000],
            }
        )


# ---------------------------------------------------------------------------
# kind="agent_loop" entry handler — release the worker thread immediately.
# ---------------------------------------------------------------------------


def create_agent_loop_handler(
    model_client: ModelClient,
    tool_registry: ToolRegistry,
    *,
    max_iterations: int = 16,
    timeout: float = 240.0,
    exclude_tools: set[str] | None = None,
    max_depth: int = 2,
) -> TaskHandler:
    """Entry handler for ``kind="agent_loop"``.

    Builds the initial ``AgentLoopState`` and emits the first
    ``agent_loop_step`` as a discovered task. Returns immediately; no LLM
    call happens here. The step handler does the real work.
    """

    def handler(ctx: TaskContext) -> TaskResult:
        current_depth = int(ctx.task.metadata.get("depth", 0))
        objective = ctx.task.goal
        deadline = time.monotonic() + timeout

        workspace_root = getattr(ctx.runtime, "workspace_root", None)
        ws_snippet = workspace_context_snippet(workspace_root)
        system_content = (
            f"{_AGENT_LOOP_SYSTEM_PROMPT}\n\n{ws_snippet}"
            if ws_snippet
            else _AGENT_LOOP_SYSTEM_PROMPT
        )

        state = AgentLoopState(
            objective=objective,
            parent_task_id=ctx.task.task_id,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": _build_initial_prompt(objective)},
            ],
            submitted_ids=[],
            last_round_call_meta=[],
            iteration=0,
            deadline_monotonic=deadline,
            max_iterations=max_iterations,
            timeout_seconds=timeout,
            depth=current_depth,
            system_prompt=system_content,
        )

        first_step = AgentTaskSpec(
            kind="agent_loop_step",
            goal=f"step 0: {objective}",
            input={
                "loop_state": state.to_input(),
                "exclude_tools": sorted(exclude_tools or set()),
                "max_depth": max_depth,
                "last_round_task_ids": [],
            },
            task_id=new_id("loop-step"),
            metadata={
                "depth": current_depth,
                "discovered_by": ctx.task.task_id,
                "agent_loop_root": ctx.task.task_id,
                "step_index": 0,
            },
            deliverable=False,
        )

        return TaskResult.completed(
            f"AgentLoop delegated: {objective}",
            discovered_tasks=[first_step],
            deliverable=False,
        )

    return handler


# ---------------------------------------------------------------------------
# kind="agent_loop_step" handler — Phase A: aggregate + LLM suspend.
# ---------------------------------------------------------------------------


def _filtered_registry(
    tool_registry: ToolRegistry,
    state: AgentLoopState,
    exclude_tools: set[str],
    max_depth: int,
) -> ToolRegistry:
    if state.depth >= max_depth:
        effective_exclude = exclude_tools | {"delegate_task"}
    else:
        effective_exclude = set(exclude_tools)
    available = set(tool_registry.names()) - effective_exclude
    return tool_registry.filtered(available)


def create_agent_loop_step_handler(
    model_client: ModelClient,
    tool_registry: ToolRegistry,
) -> TaskHandler:
    """Step handler for ``kind="agent_loop_step"``.

    A single step is split into a Phase A (aggregate + suspend on the LLM
    future) and a Phase B (resume → lower tool_calls → next step). Phase A
    runs synchronously inside the worker thread up to the moment we hold a
    future from ``model_client.complete_async``; from there we return
    ``TaskResult.suspended`` and the scheduler reschedules Phase B onto a
    fresh worker thread when the future completes.
    """

    def handler(ctx: TaskContext) -> TaskResult:
        payload = dict(ctx.task.input or {})
        state = AgentLoopState.from_input(payload.get("loop_state") or {})
        exclude_tools_raw = payload.get("exclude_tools") or []
        exclude_tools: set[str] = set(exclude_tools_raw)
        max_depth = int(payload.get("max_depth") or 2)
        last_round_task_ids: list[str] = list(
            payload.get("last_round_task_ids") or []
        )

        filtered_tools = _filtered_registry(
            tool_registry, state, exclude_tools, max_depth
        )

        # Aggregate previous round's tool results into messages.
        if last_round_task_ids and state.last_round_call_meta:
            collected = ctx.runtime.collect(last_round_task_ids)
            normalized: dict[str, TaskResult] = {
                tid: r for tid, r in collected.items() if isinstance(r, TaskResult)
            }
            _append_results_to_messages(
                state.messages,
                last_round_task_ids,
                state.last_round_call_meta,
                normalized,
            )

        # timeout / max_iterations always surface as failed.
        if time.monotonic() > state.deadline_monotonic:
            return TaskResult.failed(
                f"AgentLoop timeout after {state.timeout_seconds:.0f}s. "
                f"Completed {state.iteration} iterations for: {state.objective}",
                error_type="agent_loop_timeout",
            )
        if state.iteration >= state.max_iterations:
            return TaskResult.failed(
                f"AgentLoop reached max iterations ({state.max_iterations}) "
                f"for: {state.objective}",
                error_type="agent_loop_max_iterations",
            )

        api_messages = sanitize_tool_protocol_messages(state.messages)

        # Suspend on the LLM future. complete_streaming_async preserves
        # feat-S1 incremental on_tool_call dispatch, but for agent_loop we
        # rely on resume-time lowering for clarity (early-dispatch can be
        # added later without breaking the contract).
        future = model_client.complete_async(
            api_messages,
            tools=filtered_tools.definitions(),
        )
        model_name = str(
            getattr(getattr(model_client, "settings", None), "model", "") or ""
        )
        token = ctx.runtime.register_future(
            future,
            task_id=ctx.task.task_id,
            model=model_name,
        )

        # The future itself rides along on the snapshot — snapshot is an
        # opaque in-memory payload (the scheduler never serializes it), so
        # this avoids reaching into the runtime's private _future_registry
        # at resume time.
        snapshot: dict[str, Any] = {
            "loop_state": state.to_input(),
            "exclude_tools": sorted(exclude_tools),
            "max_depth": max_depth,
            "future_token": token,
            "future": future,
        }

        return TaskResult.suspended(
            resume_handler=_resume_after_model(model_client, tool_registry),
            awaiting=[DependencyPredicate.future_done(token)],
            suspend_token=token,
            snapshot=snapshot,
            summary=f"step {state.iteration}: awaiting model",
        )

    return handler


def _resume_after_model(
    model_client: ModelClient,
    tool_registry: ToolRegistry,
):
    """Phase B continuation invoked by the scheduler after the LLM future
    resolves. Reads the response off the future stored in the runtime
    registry, lowers tool_calls into children, and emits the next
    ``agent_loop_step`` task gated on those children.
    """

    def resume(prev: TaskResult, ctx: TaskContext) -> TaskResult:
        snapshot = prev.snapshot or {}
        state = AgentLoopState.from_input(snapshot.get("loop_state") or {})
        exclude_tools: set[str] = set(snapshot.get("exclude_tools") or [])
        max_depth = int(snapshot.get("max_depth") or 2)
        future = snapshot.get("future")
        if future is None:
            token = str(snapshot.get("future_token") or prev.suspend_token)
            future = ctx.runtime._future_registry.get(token)
        if future is None:
            return TaskResult.failed(
                f"AgentLoop lost LLM future",
                error_type="agent_loop_future_missing",
            )
        try:
            response = future.result()
        except Exception as exc:
            return TaskResult.failed(
                f"AgentLoop model call failed: {exc}",
                error_type="agent_loop_model_error",
            )

        filtered_tools = _filtered_registry(
            tool_registry, state, exclude_tools, max_depth
        )

        assistant_msg, call_meta = _build_assistant_message(response)
        state.messages.append(assistant_msg)

        if not response.tool_calls:
            return TaskResult.completed(
                response.content
                or f"AgentLoop completed: {state.objective}",
                deliverable=True,
            )

        tasks = lower_tool_calls_to_specs(
            response.tool_calls,
            tools=filtered_tools,
            workspace_root=getattr(ctx.runtime, "workspace_root", None),
            parent_task_id=state.parent_task_id,
            depth=state.depth,
            task_id_prefix="loop",
        )
        if not tasks:
            return TaskResult.completed(
                response.content or state.objective,
                deliverable=True,
            )

        new_ids = ctx.runtime.add_discovered(state.parent_task_id, tasks)
        all_child_ids = list(new_ids)
        if not all_child_ids:
            return TaskResult.completed(
                response.content or state.objective,
                deliverable=True,
            )

        next_state = replace(
            state,
            submitted_ids=list(state.submitted_ids) + list(all_child_ids),
            last_round_call_meta=call_meta,
            iteration=state.iteration + 1,
        )

        next_step = AgentTaskSpec(
            kind="agent_loop_step",
            goal=f"step {next_state.iteration}: {state.objective}",
            input={
                "loop_state": next_state.to_input(),
                "exclude_tools": sorted(exclude_tools),
                "max_depth": max_depth,
                "last_round_task_ids": list(all_child_ids),
            },
            dependencies=[
                DependencyPredicate.task_completed(cid) for cid in all_child_ids
            ],
            task_id=new_id("loop-step"),
            metadata={
                "depth": state.depth,
                "discovered_by": ctx.task.task_id,
                "agent_loop_root": ctx.task.metadata.get(
                    "agent_loop_root", state.parent_task_id
                ),
                "step_index": next_state.iteration,
            },
            deliverable=False,
        )

        return TaskResult.completed(
            f"step {state.iteration}: dispatched {len(all_child_ids)} children",
            discovered_tasks=[next_step],
            deliverable=False,
        )

    return resume
