"""Agent Protocol — minimal interface for MainAgent, WorkerAgent, PlannerAgent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from high_agent.agent.context import TotalContext
from high_agent.runtime.types import AgentTaskSpec, TaskResult


@dataclass
class AgentContext:
    """Lightweight context view for all agent types.

    MainAgent uses full TotalContext; WorkerAgent uses a scoped subset.
    """

    objective: str
    completed_summaries: list[str] = field(default_factory=list)
    runtime_digests: list[str] = field(default_factory=list)

    @staticmethod
    def for_worker(goal: str, *, input_summaries: list[str] | None = None,
                   ledger_digest: str = "") -> "AgentContext":
        return AgentContext(
            objective=goal,
            completed_summaries=list(input_summaries or []),
            runtime_digests=[ledger_digest] if ledger_digest else [],
        )

    @staticmethod
    def from_total(ctx: TotalContext) -> "AgentContext":
        return AgentContext(
            objective=ctx.objective,
            completed_summaries=list(ctx.completed_summaries),
            runtime_digests=list(ctx.runtime_digests),
        )

    def render(self, *, max_chars: int = 12_000) -> str:
        parts = [f"Objective: {self.objective}"]
        if self.completed_summaries:
            completed = "\n".join(f"- {s}" for s in self.completed_summaries[-24:])
            parts.append("Context:\n" + completed)
        if self.runtime_digests:
            parts.append("Runtime:\n" + self.runtime_digests[-1])
        rendered = "\n\n".join(parts)
        if len(rendered) <= max_chars:
            return rendered
        return rendered[:max_chars]


@dataclass
class AgentTurnResult:
    """Return value from a single agent turn."""

    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    task_result: TaskResult | None = None
    discovered_tasks: list[AgentTaskSpec] = field(default_factory=list)
    is_final: bool = False

    @staticmethod
    def final_answer(content: str) -> "AgentTurnResult":
        return AgentTurnResult(content=content, is_final=True)

    @staticmethod
    def with_tool_calls(tool_calls: list[dict[str, Any]], content: str = "") -> "AgentTurnResult":
        return AgentTurnResult(content=content, tool_calls=tool_calls, is_final=False)

    @staticmethod
    def from_task_result(result: TaskResult) -> "AgentTurnResult":
        return AgentTurnResult(
            content=result.summary,
            task_result=result,
            discovered_tasks=list(result.discovered_tasks),
            is_final=True,
        )


@runtime_checkable
class Agent(Protocol):
    """Minimal protocol that MainAgent, WorkerAgent, PlannerAgent all satisfy."""

    objective: str

    def run_one_turn(self, ledger_digest: str) -> AgentTurnResult: ...
