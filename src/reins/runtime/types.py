"""Shared runtime dataclasses."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from reins.runtime.resource_access import ResourceAccess

TaskKind = Literal["tool", "worker", "agent_loop", "agent_loop_step", "planner", "reducer", "validator", "recovery", "commit"]
TaskState = Literal["waiting", "ready", "running", "suspended", "completed", "failed", "blocked", "cancelled"]
TaskStatus = Literal["completed", "failed", "blocked", "cancelled", "timeout", "partial", "suspended"]
BarrierKind = Literal["none", "delivery", "interactive", "external_commit"]
WriteMode = Literal["replace", "append", "proposal"]


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


@dataclass(frozen=True, slots=True)
class ComponentWrite:
    component_id: str
    value: Any
    mode: WriteMode = "replace"


@dataclass
class FailureEvent:
    error_type: str
    message: str
    retryable: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DependencyPredicate:
    """Small predicate DSL for runtime readiness."""

    kind: str
    component: str | None = None
    min_version: int | None = None
    task_id: str | None = None
    event_type: str | None = None
    children: list["DependencyPredicate"] = field(default_factory=list)
    token: str | None = None

    @staticmethod
    def exists(component: str) -> "DependencyPredicate":
        return DependencyPredicate(kind="exists", component=component)

    @staticmethod
    def version_at_least(component: str, version: int) -> "DependencyPredicate":
        return DependencyPredicate(kind="version_at_least", component=component, min_version=version)

    @staticmethod
    def task_completed(task_id: str) -> "DependencyPredicate":
        return DependencyPredicate(kind="task_completed", task_id=task_id)

    @staticmethod
    def task_terminal(task_id: str) -> "DependencyPredicate":
        return DependencyPredicate(kind="task_terminal", task_id=task_id)

    @staticmethod
    def future_done(token: str) -> "DependencyPredicate":
        return DependencyPredicate(kind="future_done", token=token)

    @staticmethod
    def all(children: list["DependencyPredicate"]) -> "DependencyPredicate":
        return DependencyPredicate(kind="all", children=children)

    @staticmethod
    def any(children: list["DependencyPredicate"]) -> "DependencyPredicate":
        return DependencyPredicate(kind="any", children=children)

    def key(self) -> tuple:
        """Hashable identity for awaiting_index reverse lookup."""
        return (
            self.kind,
            self.component,
            self.min_version,
            self.task_id,
            self.event_type,
            self.token,
        )


TaskHandler = Callable[["TaskContext"], "TaskResult"]
# A resume_handler is invoked with the prior suspended TaskResult (so the
# continuation can read its snapshot / suspend_token) and a fresh TaskContext.
ResumeHandler = Callable[["TaskResult", "TaskContext"], "TaskResult"]


@dataclass
class AgentTaskSpec:
    kind: TaskKind
    goal: str
    input: dict[str, Any] = field(default_factory=dict)
    dependencies: list[DependencyPredicate] = field(default_factory=list)
    reads: set[str] = field(default_factory=set)
    writes: set[str] = field(default_factory=set)
    resource_access: ResourceAccess | None = None
    barrier: BarrierKind = "none"
    timeout_seconds: float | None = None
    priority: int = 0
    retry_policy: dict[str, Any] = field(default_factory=dict)
    task_id: str = field(default_factory=lambda: new_id("task"))
    handler: TaskHandler | None = None
    deliverable: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def with_id(self, task_id: str) -> "AgentTaskSpec":
        self.task_id = task_id
        return self


@dataclass
class TaskContext:
    task: AgentTaskSpec
    runtime: Any
    ledger_digest: str
    submitted_at: float = field(default_factory=time.monotonic)


@dataclass
class TaskResult:
    status: TaskStatus
    summary: str = ""
    component_writes: list[ComponentWrite] = field(default_factory=list)
    discovered_tasks: list[AgentTaskSpec] = field(default_factory=list)
    blocked_on: list[DependencyPredicate | str] = field(default_factory=list)
    failure_event: FailureEvent | None = None
    deliverable: bool = True
    value: Any = None
    # Suspend/resume protocol fields ( / C4 / C5). Only populated when
    # status == "suspended". The scheduler reads these in
    # ``_drain_completions`` Phase 2 to register the task in ``_suspended`` and
    # rebind ``resume_handler`` when the awaiting predicates fire.
    resume_handler: "ResumeHandler | None" = None
    awaiting: list[DependencyPredicate] = field(default_factory=list)
    suspend_token: str = ""
    snapshot: dict[str, Any] | None = None

    @staticmethod
    def completed(summary: str = "", *, writes: list[ComponentWrite] | None = None,
                  discovered_tasks: list[AgentTaskSpec] | None = None,
                  deliverable: bool = True, value: Any = None) -> "TaskResult":
        return TaskResult(
            status="completed",
            summary=summary,
            component_writes=list(writes or []),
            discovered_tasks=list(discovered_tasks or []),
            deliverable=deliverable,
            value=value,
        )

    @staticmethod
    def failed(message: str, *, error_type: str = "task_error", retryable: bool = False,
               deliverable: bool = True) -> "TaskResult":
        return TaskResult(
            status="failed",
            summary=message,
            failure_event=FailureEvent(error_type=error_type, message=message, retryable=retryable),
            deliverable=deliverable,
        )

    @staticmethod
    def blocked(message: str, *, blocked_on: list[DependencyPredicate | str] | None = None,
                deliverable: bool = True) -> "TaskResult":
        return TaskResult(
            status="blocked",
            summary=message,
            blocked_on=list(blocked_on or []),
            deliverable=deliverable,
        )

    @staticmethod
    def suspended(*, resume_handler: "ResumeHandler",
                  awaiting: list[DependencyPredicate],
                  suspend_token: str,
                  snapshot: dict[str, Any] | None = None,
                  summary: str = "") -> "TaskResult":
        """Suspend the task pending all awaiting predicates.


        ``resume_handler`` is called with ``(prev_suspended_result, new_ctx)``
        when every predicate in ``awaiting`` is satisfied. The scheduler
        accumulates suspend wall-time in ``_suspended_total_seconds`` so
        long-idling tasks do not get harvested by ``cancel_stale_tasks``.
        """
        if not awaiting:
            raise ValueError("TaskResult.suspended requires at least one awaiting predicate")
        if not suspend_token:
            raise ValueError("TaskResult.suspended requires a non-empty suspend_token")
        return TaskResult(
            status="suspended",
            summary=summary,
            resume_handler=resume_handler,
            awaiting=list(awaiting),
            suspend_token=suspend_token,
            snapshot=snapshot,
            deliverable=False,
        )


@dataclass(frozen=True, slots=True)
class DeliveryEvent:
    seq: int
    task_id: str
    kind: TaskKind
    summary: str
    result: TaskResult
    barrier: BarrierKind = "none"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.monotonic)


@dataclass(slots=True)
class DeliveryBatch:
    events: list[DeliveryEvent]
    digest: str
    batch_seq: int

    def summaries(self) -> list[str]:
        return [event.summary for event in self.events]
