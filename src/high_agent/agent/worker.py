"""WorkerAgent — single-turn delegated executor.

 Workers can now produce tool_calls and dispatch them as
``discovered_tasks``. Previously the worker handler called
``run_one_turn(...).content`` and threw away ``response.tool_calls``,
silently dropping any filesystem mutation the model attempted from inside a
worker turn (the trace symptom: empty ``schemas/`` / ``services/`` /
``templates/`` directories under a delegated scaffold).

Workers stay single-turn. Unlike ``agent_loop`` they do not iterate. The
contract now is: one model call → either a final summary OR a flat batch
of tool_calls dispatched as runtime children. The model-facing ``mode=
'sub_agent'`` keeps the multi-step path; ``mode='worker'`` is for
one-shot computation that may need to issue tool calls but does not need
to react to their results.

To keep recursion bounded, the worker's view of the tool registry has
``delegate_task`` filtered out — workers cannot fan out further worker
batches. Use ``mode='sub_agent'`` if recursive delegation is required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from high_agent.agent.protocol import AgentContext, AgentTurnResult
from high_agent.agent.prompt_policy import workspace_context_snippet
from high_agent.agent.tool_calls import (
    assistant_tool_calls_from_provider,
    repair_current_tool_calls,
    sanitize_tool_protocol_messages,
)
from high_agent.agent.tool_lowering import lower_tool_calls_to_specs
from high_agent.llm.client import ModelClient
from high_agent.llm.types import NormalizedResponse, ToolCall
from high_agent.runtime.types import AgentTaskSpec, TaskContext, TaskHandler, TaskResult
from high_agent.tools.registry import ToolRegistry


_WORKER_EXCLUDED_TOOLS = frozenset({"delegate_task"})


@dataclass
class WorkerAgent:
    """Executes a single delegated goal without recursive scheduling.

    Satisfies the Agent protocol. Does NOT hold a runtime reference.
    With a ``model_client`` the worker calls the model once and either
    returns a summary or surfaces tool_calls. With a ``tool_registry`` the
    handler can lower those tool_calls into runtime children; without one
    they are reported in text form only.
    """

    objective: str
    input_data: dict[str, Any] = field(default_factory=dict)
    model_client: ModelClient | None = None
    tool_registry: ToolRegistry | None = None
    workspace_root: str | None = None
    system_prompt: str = (
        "You are a focused worker agent. Complete the given goal using the provided context. "
        "When the goal requires filesystem changes or external side effects, call the appropriate "
        "tools in a single response — workers run one turn, so batch every tool call you need now. "
        "Otherwise, return a concise result summary."
    )
    context: AgentContext = field(init=False)

    def __post_init__(self) -> None:
        input_summaries = self.input_data.get("component_summaries") or []
        if isinstance(input_summaries, str):
            input_summaries = [input_summaries]
        ledger_digest = str(self.input_data.get("ledger_digest") or "")
        self.context = AgentContext.for_worker(
            self.objective,
            input_summaries=input_summaries,
            ledger_digest=ledger_digest,
        )

    def _effective_system_prompt(self) -> str:
        ws = workspace_context_snippet(self.workspace_root)
        return f"{self.system_prompt}\n\n{ws}" if ws else self.system_prompt

    def call_model(self, ledger_digest: str) -> NormalizedResponse | None:
        """Single model invocation; returns the raw provider response.

        Used by the handler so it can lower ``response.tool_calls`` directly
        into ``AgentTaskSpec`` without round-tripping through the rendered
        assistant-message shape. Returns ``None`` when no model_client.
        """
        if self.model_client is None:
            return None
        messages = [
            {"role": "system", "content": self._effective_system_prompt()},
            {"role": "user", "content": self._build_prompt(ledger_digest)},
        ]
        api_messages = sanitize_tool_protocol_messages(messages)
        tools_def = self.effective_registry().definitions() if self.tool_registry else None
        return self.model_client.complete(api_messages, tools=tools_def)

    def run_one_turn(self, ledger_digest: str) -> AgentTurnResult:
        response = self.call_model(ledger_digest)
        if response is None:
            return AgentTurnResult.final_answer(f"Worker completed: {self.objective}")
        if response.tool_calls:
            repair_current_tool_calls(response.tool_calls)
            rendered = assistant_tool_calls_from_provider(response.tool_calls)
            return AgentTurnResult.with_tool_calls(rendered, response.content or "")
        return AgentTurnResult.final_answer(response.content or f"Worker completed: {self.objective}")

    def _build_prompt(self, ledger_digest: str) -> str:
        parts = [f"Goal: {self.objective}"]
        user_input = self.input_data.get("input")
        if user_input:
            parts.append(f"Input: {user_input}")
        context_render = self.context.render(max_chars=4_000)
        if context_render:
            parts.append(f"Context:\n{context_render}")
        if ledger_digest:
            parts.append(f"Ledger:\n{ledger_digest}")
        return "\n\n".join(parts)

    def effective_registry(self) -> ToolRegistry:
        """Worker-scoped tool registry with delegate_task filtered out."""
        assert self.tool_registry is not None
        available = set(self.tool_registry.names()) - _WORKER_EXCLUDED_TOOLS
        return self.tool_registry.filtered(available)


def create_worker_handler(
    model_client: ModelClient | None = None,
    tool_registry: ToolRegistry | None = None,
    *,
    timeout: float = 60.0,
) -> TaskHandler:
    """Factory that returns a TaskHandler for kind='worker' tasks.

    Register with CausalRuntime via
    ``executors={'worker': create_worker_handler(client, registry)}``.
    When ``tool_registry`` is provided, the handler lowers worker tool_calls
    into ``discovered_tasks`` so the runtime schedules them with full
    resource accounting; ``delegate_task`` is filtered out at the worker
    layer to bound recursion. Without a registry, tool_calls are surfaced in
    the summary as text only.
    """

    def handler(ctx: TaskContext) -> TaskResult:
        task = ctx.task
        input_data = dict(task.input or {})
        input_data.setdefault("ledger_digest", ctx.ledger_digest)

        worker = WorkerAgent(
            objective=task.goal,
            input_data=input_data,
            model_client=model_client,
            tool_registry=tool_registry,
            workspace_root=getattr(ctx.runtime, "workspace_root", None),
        )
        response = worker.call_model(ctx.ledger_digest)
        if response is None:
            return TaskResult.completed(f"Worker completed: {task.goal}")

        raw_calls: list[ToolCall] = list(response.tool_calls or [])
        if not raw_calls:
            return TaskResult.completed(response.content or f"Worker completed: {task.goal}")

        if tool_registry is None:
            preview = ", ".join(call.name for call in raw_calls)
            summary = (
                f"Worker produced {len(raw_calls)} tool_call(s) ({preview}) "
                "but no tool_registry was wired; treating as final."
            )
            content = (response.content or "").strip()
            if content:
                summary = f"{summary}\n{content[:1000]}"
            return TaskResult.completed(summary)

        repair_current_tool_calls(raw_calls)
        specs: list[AgentTaskSpec] = lower_tool_calls_to_specs(
            raw_calls,
            tools=worker.effective_registry(),
            workspace_root=getattr(ctx.runtime, "workspace_root", None),
            parent_task_id=task.task_id,
            depth=int(task.metadata.get("depth", 0)),
            task_id_prefix="worker",
        )
        names = ", ".join(spec.metadata.get("tool_name", "?") for spec in specs)
        summary = f"Worker dispatched {len(specs)} tool call(s): {names}"
        narrative = (response.content or "").strip()
        if narrative:
            summary = f"{summary}\n{narrative[:800]}"
        return TaskResult.completed(summary, discovered_tasks=specs)

    return handler
