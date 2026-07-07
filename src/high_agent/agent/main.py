"""Main agent loop for model/tool/runtime orchestration."""

from __future__ import annotations

import threading
import json
from dataclasses import dataclass, field
from collections.abc import Callable
from typing import Any

from high_agent.agent.context import TotalContext
from high_agent.agent.controller import AgentRunController, RunUsage, delivery_content
from high_agent.agent.prompt_policy import PromptPolicy
from high_agent.agent.protocol import AgentTurnResult
from high_agent.agent.tool_calls import (
    NormalizedToolCall,
    ToolCallNormalizer,
    assistant_tool_calls_from_provider,
    repair_current_tool_calls,
)
from high_agent.llm.client import ModelClient
from high_agent.llm.types import NormalizedResponse, ToolCall
from high_agent.runtime.scheduler import CausalRuntime
from high_agent.runtime.types import AgentTaskSpec, DeliveryBatch, DeliveryEvent, TaskResult, new_id
from high_agent.tools.registry import ToolRegistry


@dataclass
class ToolResultGroup:
    parent_call_id: str
    original_tool_name: str
    child_call_ids: set[str] = field(default_factory=set)
    delivered_call_ids: set[str] = field(default_factory=set)
    events: list[DeliveryEvent] = field(default_factory=list)


@dataclass
class MainAgent:
    objective: str
    runtime: CausalRuntime
    model_client: ModelClient | None = None
    tools: ToolRegistry | None = None
    # the prior prompt
    # gated completion on "ledger empty" via the cue "Do not claim
    # completion while runtime status says tasks are pending". Under
    # multi-planner pumping the ledger is NEVER empty (other planners'
    # tasks are always running), so the model never emitted final.
    # Per the right framing is: ledger is RUNTIME BOOKKEEPING,
    # not the goal-completion oracle. The model should decide based on
    # whether the OBJECTIVE is met (files exist, tests pass, etc.), not
    # on whether sibling planners happen to have tasks running.
    # Trace evidence pre-fix: 12/12 W_fix high_agent-REINS cells in v2 had
    # planner.final_candidate=0; smoke v6 with max_planner=1 produced 5
    # final_candidate events on the same prompt — proving the cue, not
    # the multi-planner mechanism itself, was the blocker.
    system_prompt: str = (
        "You are high-agent, a parallel task execution system operating in a fresh workspace. "
        "Tasks may reference files that do not exist yet — that is expected. Treat the prompt "
        "as a specification: scaffold the file layout the prompt implies, write a minimal "
        "implementation, add the regression test the prompt asks for, then run it. "
        "Use tools to inspect or change the workspace. "
        "When independent tool calls have no resource conflict, the runtime executes them in parallel. "
        "IMPORTANT: Prefer batching multiple tool calls in a single response. "
        "For example, create all files in one response rather than one file per response. "
        "Use write_many_files or multiple write_file calls together when creating project structures. "
        "Decide completion based on whether the OBJECTIVE is met — the requested files exist with "
        "correct content and the test you were asked to add passes. The runtime ledger ('Tasks "
        "running', 'Tasks waiting') is internal bookkeeping for parallel scheduling; running "
        "tasks do NOT mean the goal is incomplete. When the objective is met, reply with a final "
        "summary and NO tool calls — that ends the conversation. Do not keep exploring once the "
        "goal is met just because the ledger shows other in-flight scheduling activity."
    )
    tool_use_enforcement: Any = "auto"
    max_planner_requests: int = 4
    # v11-D2: caller decides; cli/main.py threads model_timeout in here so the
    # controller's stale cap matches the underlying httpx wait budget.
    planner_stale_seconds: float = 600.0
    # v11-D3: caller-provided breaker threshold. Trips after N planner
    # timeouts on the same snapshot; controller refuses to keep dispatching
    # against a wedged snapshot rather than spinning planner_seq forever.
    planner_stuck_threshold: int = 3
    context: TotalContext = field(init=False)
    last_usage: RunUsage = field(init=False, default_factory=RunUsage)
    last_messages: list[dict[str, Any]] = field(init=False, default_factory=list)
    _tool_result_groups: dict[str, ToolResultGroup] = field(init=False, default_factory=dict)
    _active_action_index: dict[tuple[str, str], set[str]] = field(init=False, default_factory=dict)
    _active_action_index_lock: threading.Lock = field(init=False, default_factory=threading.Lock)

    def __post_init__(self) -> None:
        self.context = TotalContext(self.objective)

    def submit_tasks(self, tasks: list[AgentTaskSpec]) -> list[str]:
        self.runtime.start()
        return self.runtime.submit(tasks)

    def wait_delivery(self, timeout: float | None = None) -> DeliveryBatch | None:
        batch = self.runtime.wait_next_delivery(timeout=timeout)
        if batch:
            for summary in batch.summaries():
                self.context.add_delivery(summary, batch.digest)
        return batch

    def finalize_when_idle(self, timeout: float | None = None) -> str:
        self.runtime.wait_all(timeout=timeout)
        while True:
            batch = self.wait_delivery(timeout=0)
            if not batch:
                break
        return self.context.render()

    def run(
        self,
        prompt: str | None = None,
        *,
        max_iterations: int = 200,
        model_params: dict[str, Any] | None = None,
        delivery_timeout: float = 30.0,
        conversation_history: list[dict[str, Any]] | None = None,
        on_delivery: Callable[[DeliveryBatch], None] | None = None,
    ) -> str:
        if self.model_client is None:
            raise RuntimeError("MainAgent.run requires model_client")
        if self.tools is None:
            raise RuntimeError("MainAgent.run requires tools")

        objective = prompt or self.objective
        self.context = TotalContext(objective)
        self._tool_result_groups = {}
        self.runtime.start()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt_for_run()},
        ]
        messages.extend(_compact_history(conversation_history or []))
        messages.append({"role": "user", "content": objective})
        params = dict(model_params or {})
        controller = AgentRunController(
            agent=self,
            objective=objective,
            messages=messages,
            model_params=params,
            delivery_timeout=delivery_timeout,
            max_iterations=max_iterations,
            max_planner_requests=self.max_planner_requests,
            planner_stale_seconds=self.planner_stale_seconds,
            planner_stuck_threshold=self.planner_stuck_threshold,
            on_delivery=on_delivery,
        )
        answer = controller.run()
        self.last_usage = controller.usage
        self.last_messages = messages
        return answer

    def _system_prompt_for_run(self) -> str:
        settings = getattr(self.model_client, "settings", None)
        model_name = str(getattr(settings, "model", "") or "")
        return PromptPolicy(self.tool_use_enforcement).build(
            base_prompt=self.system_prompt,
            model_name=model_name,
            has_tools=bool(self.tools and self.tools.names()),
            workspace_root=getattr(self.runtime, "workspace_root", None),
        )

    def _record_assistant_message(self, messages: list[dict[str, Any]], response: NormalizedResponse) -> list[NormalizedToolCall]:
        message: dict[str, Any] = {"role": "assistant", "content": response.content or ""}
        normalized_calls: list[NormalizedToolCall] = []
        if response.tool_calls:
            repair_current_tool_calls(response.tool_calls)
            message["tool_calls"] = assistant_tool_calls_from_provider(response.tool_calls)
            normalized_calls = ToolCallNormalizer().normalize(response.tool_calls)
        messages.append(message)
        return normalized_calls

    def _normalize_tool_calls(self, tool_calls: list[ToolCall]) -> list[NormalizedToolCall]:
        return ToolCallNormalizer().normalize(tool_calls)

    def _submit_tool_calls(self, tool_calls: list[ToolCall]) -> list[str]:
        return self._submit_normalized_tool_calls(self._normalize_tool_calls(tool_calls))

    def _submit_normalized_tool_calls(self, tool_calls: list[NormalizedToolCall]) -> list[str]:
        tasks: list[AgentTaskSpec] = []
        valid_tools = set(self.tools.names()) if self.tools is not None else set()
        for index, item in enumerate(tool_calls):
            call = item.call
            call_id = call.id or new_id("call")
            parent_call_id = item.parent_call_id or ""
            if parent_call_id:
                group = self._tool_result_groups.setdefault(
                    parent_call_id,
                    ToolResultGroup(
                        parent_call_id=parent_call_id,
                        original_tool_name=_group_tool_name(item),
                    ),
                )
                group.child_call_ids.add(call_id)

            def add_task(task: AgentTaskSpec) -> None:
                if parent_call_id:
                    task.metadata["parent_tool_call_id"] = parent_call_id
                tasks.append(task)

            try:
                args = call.args_dict()
            except Exception as exc:
                add_task(_failed_tool_task(call, call_id, f"invalid tool arguments: {exc}"))
                continue

            if call.name not in valid_tools:
                add_task(_failed_tool_task(call, call_id, f"unknown_tool: {item.original_name}", error_type="unknown_tool"))
                continue

            try:
                task = self.tools.task_from_call(
                    call.name,
                    args,
                    workspace_root=self.runtime.workspace_root,
                    task_id=f"tool-{call_id or index}",
                )
            except Exception as exc:
                add_task(_failed_tool_task(call, call_id, f"tool lowering failed: {exc}"))
                continue
            task.metadata.update(
                {
                    "tool_call_id": call_id,
                    "tool_name": call.name,
                    "original_tool_name": item.original_name,
                    "parent_tool_call_id": parent_call_id,
                }
            )
            add_task(task)
        task_ids = self.runtime.submit(tasks)
        self._index_active_actions(tasks)
        return task_ids

    def _append_delivery_messages(self, messages: list[dict[str, Any]], batch: DeliveryBatch) -> int:
        appended = 0
        for event in batch.events:
            content = delivery_content(event, batch.digest)
            tool_call_id = event.metadata.get("tool_call_id") or event.task_id
            parent_call_id = str(event.metadata.get("parent_tool_call_id") or "")
            if parent_call_id and parent_call_id in self._tool_result_groups:
                group = self._tool_result_groups[parent_call_id]
                group.events.append(event)
                group.delivered_call_ids.add(str(tool_call_id))
                if group.delivered_call_ids >= group.child_call_ids:
                    messages.append(
                        {
                            "role": "tool",
                            "name": group.original_tool_name,
                            "tool_call_id": parent_call_id,
                            "content": f"tool_call_id={parent_call_id}\n{_grouped_delivery_content(group, batch.digest)}",
                        }
                    )
                    del self._tool_result_groups[parent_call_id]
                    appended += 1
                continue
            messages.append(
                {
                    "role": "tool",
                    "name": event.metadata.get("tool_name") or event.kind,
                    "tool_call_id": tool_call_id,
                    "content": f"tool_call_id={tool_call_id}\n{content}",
                }
            )
            appended += 1
        return appended

    def run_one_turn(self, ledger_digest: str) -> AgentTurnResult:
        """Protocol-compatible single-turn entry point.

        For MainAgent this delegates to the full run() loop with max_iterations=1.
        Primarily exists so MainAgent satisfies the Agent protocol structurally.
        """
        if self.model_client is None or self.tools is None:
            return AgentTurnResult.final_answer(self.context.render())
        answer = self.run(max_iterations=1)
        return AgentTurnResult.final_answer(answer)

    def _index_active_actions(self, tasks: list[AgentTaskSpec]) -> None:
        with self._active_action_index_lock:
            for task in tasks:
                key = _action_index_key(task)
                if key:
                    self._active_action_index.setdefault(key, set()).add(task.task_id)

    def _unindex_active_action(self, task_id: str, task: AgentTaskSpec) -> None:
        key = _action_index_key(task)
        if not key:
            return
        with self._active_action_index_lock:
            s = self._active_action_index.get(key)
            if s:
                s.discard(task_id)
                if not s:
                    del self._active_action_index[key]


def _compact_history(history: list[dict[str, Any]], *, max_messages: int = 12) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in history[-max_messages:]:
        role = item.get("role")
        if role not in {"user", "assistant"}:
            continue
        content = str(item.get("content") or "")
        if not content:
            continue
        out.append({"role": role, "content": content[:12_000]})
    return out


def _failed_tool_task(call: ToolCall, call_id: str, message: str, *, error_type: str = "tool_lowering_error") -> AgentTaskSpec:
    def _handle(ctx: Any) -> TaskResult:
        return TaskResult.failed(message, error_type=error_type)

    return AgentTaskSpec(
        kind="tool",
        goal=f"{call.name} failed to lower",
        # previously the suffix was int(time.time() * 1000), so two
        # failed lowerings within the same millisecond (e.g. a batch with
        # multiple unknown_tool entries) collided and runtime.submit's
        # `self._tasks[task.task_id] = task` overwrote the first task. Use
        # a process-wide monotonic id (new_id) instead.
        task_id=new_id(f"tool-error-{call_id}"),
        handler=_handle,
        metadata={"tool_call_id": call_id, "tool_name": call.name},
    )


def _group_tool_name(item: NormalizedToolCall) -> str:
    data = item.call.provider_data or {}
    if isinstance(data, dict):
        if data.get("lowered_from"):
            return str(data["lowered_from"])
        if data.get("wrapper"):
            return str(data["wrapper"])
    return item.original_name or item.call.name


def _grouped_delivery_content(group: ToolResultGroup, digest: str) -> str:
    statuses = [event.result.status for event in group.events]
    if any(status == "failed" for status in statuses):
        status = "failed"
    elif any(status == "blocked" for status in statuses):
        status = "blocked"
    elif any(status == "cancelled" for status in statuses):
        status = "cancelled"
    else:
        status = "completed"
    payload = {
        "tool": group.original_tool_name,
        "tool_call_id": group.parent_call_id,
        "status": status,
        "children": [
            {
                "task_id": event.task_id,
                "tool_call_id": event.metadata.get("tool_call_id") or event.task_id,
                "tool": event.metadata.get("tool_name") or event.kind,
                "status": event.result.status,
                "summary": event.summary,
                "duration_seconds": event.metadata.get("duration_seconds"),
            }
            for event in group.events
        ],
        "ledger": digest,
    }
    return json.dumps(payload, ensure_ascii=False)


def _action_index_key(task: AgentTaskSpec) -> tuple[str, str] | None:
    tool_name = task.metadata.get("tool_name", "")
    if not tool_name:
        return None
    # canonicalise over the full task.input. Tools like
    # delegate_task put their payload at the top level (e.g. {"objective":
    # ..., "tasks": [...]}); the previous `(task.input or {}).get("args")`
    # always returned {} for them, collapsing every concurrent delegate
    # to the same key and incorrectly tagging the second / third / ...
    # parallel delegate_task as a duplicate of the first.
    payload = dict(task.input or {})
    payload = _strip_internal_keys(payload)
    args = payload.get("args")
    if isinstance(args, dict):
        # AgentLoop (formerly SubAgent) injects `_depth` into
        # delegate_task args during `_lower_tool_calls` so the runtime can
        # track recursive depth, and ToolRegistry.task_from_call adds
        # `_workspace_root`. Both are dispatcher-internal and must not affect
        # dedupe; otherwise legitimate parallel delegates at different depths
        # look distinct from each other and a model echoing back `_depth`
        # could reset the recursion guard.
        payload["args"] = _strip_internal_keys(args)
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return (tool_name, canonical)


def _strip_internal_keys(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if not (isinstance(k, str) and k.startswith("_"))}
