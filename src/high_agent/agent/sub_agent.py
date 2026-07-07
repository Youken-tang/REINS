"""SubAgent — multi-turn executor with local planning and tool execution."""

from __future__ import annotations

import time
import json
from dataclasses import dataclass, field
from typing import Any

from high_agent.agent.protocol import AgentContext, AgentTurnResult
from high_agent.agent.tool_calls import ToolCallNormalizer, sanitize_tool_protocol_messages
from high_agent.llm.client import ModelClient
from high_agent.runtime.types import AgentTaskSpec, TaskContext, TaskHandler, TaskResult, new_id
from high_agent.tools.registry import ToolRegistry


_SUB_AGENT_SYSTEM_PROMPT = (
    "You are a focused sub-agent executing a specific goal within a larger task. "
    "Use the provided tools to complete your objective. "
    "Return multiple tool calls in one response when they have no dependencies. "
    "When all work is done, provide a concise summary of what was accomplished."
)


@dataclass
class SubAgent:
    """Multi-turn agent that runs inside a task handler thread.

    Uses TaskContext.runtime to submit sub-tasks and waits for their
    completion, forming a local plan-execute loop. Satisfies the Agent protocol.
    """

    objective: str
    model_client: ModelClient
    tools: ToolRegistry
    runtime: Any
    parent_task_id: str
    max_iterations: int = 8
    timeout_seconds: float = 120.0
    depth: int = 0
    system_prompt: str = _SUB_AGENT_SYSTEM_PROMPT
    context: AgentContext = field(init=False)
    _messages: list[dict[str, Any]] = field(init=False, default_factory=list)
    _submitted_ids: set[str] = field(init=False, default_factory=set)
    _iteration: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.context = AgentContext(objective=self.objective)

    def run(self) -> TaskResult:
        """Execute the local plan-execute loop until done or max_iterations."""
        deadline = time.monotonic() + self.timeout_seconds
        self._messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self._build_initial_prompt()},
        ]

        for self._iteration in range(self.max_iterations):
            if time.monotonic() > deadline:
                return TaskResult.completed(
                    f"SubAgent timeout after {self.timeout_seconds:.0f}s. "
                    f"Completed {self._iteration} iterations for: {self.objective}"
                )

            api_messages = sanitize_tool_protocol_messages(self._messages)
            early_ids: list[str] = []
            response = self._call_model(api_messages, early_ids)

            assistant_msg: dict[str, Any] = {"role": "assistant", "content": response.content or ""}
            if response.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id or new_id("call"),
                        "type": "function",
                        "function": {"name": tc.name, "arguments": tc.arguments},
                    }
                    for tc in response.tool_calls
                ]
            self._messages.append(assistant_msg)

            if not response.tool_calls:
                return TaskResult.completed(response.content or f"SubAgent completed: {self.objective}")

            tasks = self._lower_tool_calls(response.tool_calls)
            if not tasks:
                return TaskResult.completed(response.content or self.objective)

            early_set = set(early_ids)
            remaining_tasks = [t for t in tasks if t.task_id not in early_set]
            if remaining_tasks:
                new_ids = self.runtime.add_discovered(self.parent_task_id, remaining_tasks)
                self._submitted_ids.update(new_ids)

            all_ids = early_ids + [t.task_id for t in remaining_tasks]
            remaining = max(0.1, deadline - time.monotonic())
            results = self._wait_for_tasks(all_ids, timeout=min(remaining, 60.0))

            self._append_results_to_messages(tasks, results)

        return TaskResult.completed(
            f"SubAgent reached max iterations ({self.max_iterations}) for: {self.objective}"
        )

    def run_one_turn(self, ledger_digest: str) -> AgentTurnResult:
        """Protocol-compatible entry point."""
        result = self.run()
        return AgentTurnResult.from_task_result(result)

    def _build_initial_prompt(self) -> str:
        parts = [f"Goal: {self.objective}"]
        digest = self._scoped_digest()
        if digest:
            parts.append(f"Runtime state:\n{digest}")
        return "\n\n".join(parts)

    def _call_model(self, api_messages: list[dict[str, Any]], early_ids: list[str]) -> Any:
        """Call model with streaming early dispatch when available, fallback to sync."""
        complete_streaming = getattr(self.model_client, "complete_streaming", None)
        if complete_streaming is not None and callable(complete_streaming) and not isinstance(complete_streaming, property):
            def _on_tool_call(call: Any) -> None:
                try:
                    tasks = self._lower_tool_calls([call])
                    if tasks:
                        ids = self.runtime.add_discovered(self.parent_task_id, tasks)
                        self._submitted_ids.update(ids)
                        early_ids.extend(ids)
                except Exception as exc:
                    try:
                        self.runtime.trace.emit(
                            "subagent.dispatch_failed",
                            parent_task_id=self.parent_task_id,
                            error=str(exc),
                            error_type=type(exc).__name__,
                            tool=str(getattr(call, "name", "") or ""),
                        )
                    except Exception:
                        pass

            try:
                return complete_streaming(
                    api_messages,
                    tools=self.tools.definitions(),
                    on_tool_call=_on_tool_call,
                )
            except (TypeError, AttributeError):
                pass
        return self.model_client.complete(
            api_messages,
            tools=self.tools.definitions(),
        )

    def _scoped_digest(self) -> str:
        if not self._submitted_ids:
            return ""
        records = self.runtime.ledger.records_snapshot()
        digest = self.runtime.ledger.digest()
        lines: list[str] = []

        for tid in sorted(self._submitted_ids):
            rec = records.get(tid)
            if rec:
                lines.append(f"  {rec.state}: {rec.goal[:80]}")

        causal_lines: list[str] = []
        for src, targets in (digest.causal_chains or {}).items():
            if src in self._submitted_ids:
                scoped = [t for t in targets if t in self._submitted_ids]
                if scoped:
                    causal_lines.append(f"  {src} → {', '.join(scoped[:4])}")
        if causal_lines:
            lines.append("Completions and what they unblocked:")
            lines.extend(causal_lines)

        scope = self._submitted_ids | {self.parent_task_id}
        disc_lines: list[str] = []
        for parent, children in (digest.discovery_chains or {}).items():
            if parent in scope:
                scoped = [c for c in children if c in self._submitted_ids]
                if scoped:
                    disc_lines.append(f"  {parent} ⇒ {', '.join(scoped[:4])}")
        if disc_lines:
            lines.append("Discovered tasks:")
            lines.extend(disc_lines)

        return "\n".join(lines) if lines else ""

    def _lower_tool_calls(self, tool_calls: list[Any]) -> list[AgentTaskSpec]:
        normalizer = ToolCallNormalizer()
        normalized = normalizer.normalize(tool_calls)
        tasks: list[AgentTaskSpec] = []
        valid_tools = set(self.tools.names())
        workspace_root = getattr(self.runtime, "workspace_root", None)

        for item in normalized:
            call = item.call
            call_id = call.id or new_id("call")
            try:
                args = call.args_dict()
            except Exception:
                tasks.append(self._failed_task(call_id, call.name, "invalid arguments"))
                continue
            if call.name not in valid_tools:
                tasks.append(self._failed_task(call_id, call.name, f"unknown tool: {call.name}"))
                continue
            if call.name == "delegate_task":
                args["_depth"] = self.depth + 1
            try:
                task = self.tools.task_from_call(
                    call.name, args,
                    workspace_root=workspace_root,
                    task_id=f"sub-{call_id}",
                )
            except Exception as exc:
                tasks.append(self._failed_task(call_id, call.name, f"lowering failed: {exc}"))
                continue
            task.metadata.update({
                "tool_call_id": call_id,
                "tool_name": call.name,
                "discovered_by": self.parent_task_id,
            })
            tasks.append(task)
        return tasks

    def _failed_task(self, call_id: str, tool_name: str, message: str) -> AgentTaskSpec:
        def _handle(ctx: TaskContext) -> TaskResult:
            return TaskResult.failed(message, error_type="sub_agent_tool_error")

        return AgentTaskSpec(
            kind="tool",
            goal=f"{tool_name} failed: {message}",
            task_id=f"sub-err-{call_id}",
            handler=_handle,
            metadata={"tool_call_id": call_id, "tool_name": tool_name},
        )

    def _wait_for_tasks(self, task_ids: list[str], timeout: float = 60.0) -> dict[str, TaskResult]:
        return self.runtime.wait_tasks(task_ids, timeout=timeout)

    def _append_results_to_messages(self, tasks: list[AgentTaskSpec], results: dict[str, TaskResult]) -> None:
        for task in tasks:
            call_id = task.metadata.get("tool_call_id", task.task_id)
            tool_name = task.metadata.get("tool_name", task.kind)
            result = results.get(task.task_id)
            if result is None:
                content = "no result (timeout)"
            elif result.status == "completed":
                content = result.summary or "completed"
            else:
                content = f"[{result.status}] {result.summary}"
            self._messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "name": tool_name,
                "content": content[:4000],
            })


def create_sub_agent_handler(
    model_client: ModelClient,
    tool_registry: ToolRegistry,
    *,
    max_iterations: int = 8,
    timeout: float = 120.0,
    exclude_tools: set[str] | None = None,
    max_depth: int = 2,
) -> TaskHandler:
    """Factory that returns a TaskHandler for kind='sub_agent' tasks.

    Register with CausalRuntime via executors={'sub_agent': create_sub_agent_handler(...)}.
    Supports recursive delegation up to max_depth levels.
    """

    def handler(ctx: TaskContext) -> TaskResult:
        current_depth = int(ctx.task.metadata.get("depth", 0))
        if current_depth >= max_depth:
            excluded = (exclude_tools or set()) | {"delegate_task"}
        else:
            excluded = exclude_tools or set()
        available = set(tool_registry.names()) - excluded
        filtered_tools = tool_registry.filtered(available)

        sub = SubAgent(
            objective=ctx.task.goal,
            model_client=model_client,
            tools=filtered_tools,
            runtime=ctx.runtime,
            parent_task_id=ctx.task.task_id,
            max_iterations=max_iterations,
            timeout_seconds=timeout,
            depth=current_depth,
        )
        return sub.run()

    return handler
