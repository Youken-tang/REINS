"""Internal task ledger and compact digest generation."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import RLock

from reins.time_utils import format_duration_compact
from reins.runtime.context_store import ContextStore
from reins.runtime.types import AgentTaskSpec, FailureEvent, TaskState

TERMINAL_STATES = {"completed", "failed", "blocked", "cancelled"}


@dataclass
class TaskRecord:
    task_id: str
    kind: str
    goal: str
    state: TaskState
    created_at: float = field(default_factory=time.monotonic)
    updated_at: float = field(default_factory=time.monotonic)
    started_at: float | None = None
    finished_at: float | None = None
    summary: str = ""
    waiting_on: list[str] = field(default_factory=list)
    failure: FailureEvent | None = None
    triggered_by: str | None = None
    triggered_tasks: list[str] = field(default_factory=list)
    discovered_by: str | None = None
    discovered_tasks: list[str] = field(default_factory=list)

    def run_seconds(self, now: float | None = None) -> float:
        if self.started_at is None:
            return 0.0
        end = self.finished_at if self.finished_at is not None else (now if now is not None else time.monotonic())
        return max(0.0, end - self.started_at)


@dataclass
class LedgerDigest:
    text: str
    seq: int
    counts: dict[str, int]
    causal_chains: dict[str, list[str]] = field(default_factory=dict)
    recent_completions: list[tuple[str, list[str]]] = field(default_factory=list)
    discovery_chains: dict[str, list[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskTimingStats:
    wall_seconds: float = 0.0
    task_seconds: float = 0.0
    completed_task_seconds: float = 0.0
    running_task_seconds: float = 0.0
    completed_tasks: int = 0
    running_tasks: int = 0
    max_task_id: str = ""
    max_task_seconds: float = 0.0
    running: tuple[tuple[str, float], ...] = ()

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "wall_seconds": self.wall_seconds,
            "task_seconds": self.task_seconds,
            "completed_task_seconds": self.completed_task_seconds,
            "running_task_seconds": self.running_task_seconds,
            "completed_tasks": self.completed_tasks,
            "running_tasks": self.running_tasks,
            "max_task_id": self.max_task_id,
            "max_task_seconds": self.max_task_seconds,
        }

    def summary(self) -> str:
        parts = [
            f"wall={format_duration_compact(self.wall_seconds)}",
            f"task={format_duration_compact(self.task_seconds)}",
        ]
        if self.running_tasks:
            running = ", ".join(f"{task_id} {format_duration_compact(seconds)}" for task_id, seconds in self.running[:3])
            parts.append(f"running={running}")
        if self.max_task_id:
            parts.append(f"max={self.max_task_id} {format_duration_compact(self.max_task_seconds)}")
        return "time: " + ", ".join(parts)


class RuntimeLedger:
    """Run-scoped accounting system, not model context storage.

    Internally delegates versioned state to a ContextStore for lock-free
    reads and per-block parallel writes. The public API remains unchanged.
    """

    def __init__(self, store: ContextStore | None = None) -> None:
        self._lock = RLock()
        self._records: dict[str, TaskRecord] = {}
        self._components: dict[str, int] = {}
        self._conflicts: list[str] = []
        self._delivery_counts = {"ready": 0, "delivered": 0, "buffered": 0}
        self._causal_edges: dict[str, list[str]] = {}
        self._discovery_edges: dict[str, list[str]] = {}
        self._recent_triggers: list[tuple[str, str]] = []
        self.store = store or ContextStore()

    @property
    def seq(self) -> int:
        return self.store.clock.value

    def add_task(self, task: AgentTaskSpec, state: TaskState) -> None:
        with self._lock:
            self._records[task.task_id] = TaskRecord(
                task_id=task.task_id,
                kind=task.kind,
                goal=task.goal,
                state=state,
            )
        self.store.write(f"task:{task.task_id}", {"state": state, "kind": task.kind, "goal": task.goal})

    def set_state(self, task_id: str, state: TaskState, *, summary: str = "",
                  waiting_on: list[str] | None = None,
                  failure: FailureEvent | None = None) -> None:
        with self._lock:
            rec = self._records.get(task_id)
            if not rec:
                return
            now = time.monotonic()
            rec.state = state
            rec.updated_at = now
            if state == "running" and rec.started_at is None:
                rec.started_at = now
            if state in TERMINAL_STATES and rec.finished_at is None:
                rec.finished_at = now
            if summary:
                rec.summary = summary
            if waiting_on is not None:
                rec.waiting_on = waiting_on
            if failure:
                rec.failure = failure
        self.store.write(f"task:{task_id}", {"state": state, "summary": summary})

    def note_component(self, component_id: str, version: int) -> None:
        with self._lock:
            self._components[component_id] = version
        self.store.write("stats", {"component": component_id, "version": version})

    def note_conflict(self, task_id: str, reason: str) -> None:
        with self._lock:
            self._conflicts.append(f"{task_id}:{reason}")
            self._conflicts = self._conflicts[-20:]
        self.store.write("stats", {"conflict": f"{task_id}:{reason}"})

    def note_delivery(self, state: str, count: int = 1) -> None:
        with self._lock:
            self._delivery_counts[state] = self._delivery_counts.get(state, 0) + count
        self.store.write("stats", {state: count})

    def record_trigger(self, source_id: str, target_task_id: str) -> None:
        """Record that source_id (task_id or component_id) caused target_task_id to wake."""
        with self._lock:
            edges = self._causal_edges.setdefault(source_id, [])
            if target_task_id not in edges:
                edges.append(target_task_id)
            target_rec = self._records.get(target_task_id)
            if target_rec and target_rec.triggered_by is None:
                target_rec.triggered_by = source_id
            source_rec = self._records.get(source_id)
            if source_rec and target_task_id not in source_rec.triggered_tasks:
                source_rec.triggered_tasks.append(target_task_id)
            self._recent_triggers.append((source_id, target_task_id))
            self._recent_triggers = self._recent_triggers[-32:]
        self.store.write(f"causal:{source_id}", self._causal_edges.get(source_id, []))

    def record_discovery(self, parent_task_id: str, discovered_task_id: str) -> None:
        """Record that parent_task_id dynamically produced discovered_task_id."""
        with self._lock:
            edges = self._discovery_edges.setdefault(parent_task_id, [])
            if discovered_task_id not in edges:
                edges.append(discovered_task_id)
            child_rec = self._records.get(discovered_task_id)
            if child_rec and child_rec.discovered_by is None:
                child_rec.discovered_by = parent_task_id
            parent_rec = self._records.get(parent_task_id)
            if parent_rec and discovered_task_id not in parent_rec.discovered_tasks:
                parent_rec.discovered_tasks.append(discovered_task_id)
        self.store.write(f"discovery:{parent_task_id}", self._discovery_edges.get(parent_task_id, []))

    def task_state(self, task_id: str) -> TaskState | None:
        with self._lock:
            rec = self._records.get(task_id)
            return rec.state if rec else None

    def task_started_at(self, task_id: str) -> float | None:
        """Return the monotonic start time of a task, or None if not started."""
        with self._lock:
            rec = self._records.get(task_id)
            return rec.started_at if rec else None

    def task_created_at(self, task_id: str) -> float | None:
        """Return the monotonic creation time of a task, or None if unknown.

        Used by trace emit sites that need wait_seconds = ready_at - created_at
.
        """
        with self._lock:
            rec = self._records.get(task_id)
            return rec.created_at if rec else None

    def counts(self) -> dict[str, int]:
        with self._lock:
            out: dict[str, int] = {}
            for rec in self._records.values():
                out[rec.state] = out.get(rec.state, 0) + 1
            return out

    def task_duration(self, task_id: str, *, now: float | None = None) -> float | None:
        with self._lock:
            rec = self._records.get(task_id)
            if rec is None:
                return None
            return rec.run_seconds(now)

    def timing(self, *, now: float | None = None) -> TaskTimingStats:
        with self._lock:
            if not self._records:
                return TaskTimingStats()
            current = time.monotonic() if now is None else now
            records = list(self._records.values())
            wall_seconds = max(0.0, current - min(rec.created_at for rec in records))
            task_seconds = 0.0
            completed_task_seconds = 0.0
            running_task_seconds = 0.0
            completed_tasks = 0
            running: list[tuple[str, float]] = []
            max_task_id = ""
            max_task_seconds = 0.0
            for rec in records:
                duration = rec.run_seconds(current)
                task_seconds += duration
                if rec.state == "running":
                    running_task_seconds += duration
                    running.append((rec.task_id, duration))
                if rec.state in TERMINAL_STATES:
                    completed_tasks += 1
                    completed_task_seconds += duration
                if duration > max_task_seconds:
                    max_task_id = rec.task_id
                    max_task_seconds = duration
            running.sort(key=lambda item: item[1], reverse=True)
            return TaskTimingStats(
                wall_seconds=wall_seconds,
                task_seconds=task_seconds,
                completed_task_seconds=completed_task_seconds,
                running_task_seconds=running_task_seconds,
                completed_tasks=completed_tasks,
                running_tasks=len(running),
                max_task_id=max_task_id,
                max_task_seconds=max_task_seconds,
                running=tuple(running),
            )

    def digest(self, *, since_seq: int | None = None) -> LedgerDigest:
        with self._lock:
            #: the digest is bookkeeping
            # for the planner, not a transcript of every task ever run.
            # Showing 4 labels per state is fine for {running,waiting,…},
            # but `completed` grew unbounded across a cell (26+ tasks by
            # planner #28 in capped runs), priming the LLM to "keep going"
            # because the visible "completed" list looked huge. We now:
            #   - report `completed` as a count + the 2 most recent labels
            #     (LLM only needs to know what just finished, not every
            #     prior tool that ever ran);
            #   - keep other states at 4 labels (those are bounded by
            #     in-flight concurrency anyway).
            groups: dict[str, list[str]] = {}
            for rec in self._records.values():
                groups.setdefault(rec.state, []).append(_label(rec))
            parts: list[str] = []
            for state in ("completed", "running", "suspended", "waiting", "ready", "failed", "blocked", "cancelled"):
                labels = groups.get(state) or []
                if not labels:
                    continue
                if state == "completed":
                    # Most recent first (records dict insertion order ≈ creation;
                    # we can't reliably sort by completion time without changing
                    # TaskRecord, but the most-recent slice is a strict improvement).
                    recent = labels[-2:]
                    parts.append(f"completed: {len(labels)} total, recent: {', '.join(recent)}")
                else:
                    parts.append(f"{state}: {', '.join(labels[:4])}")
            if self._conflicts:
                parts.append(f"conflicts: {'; '.join(self._conflicts[-3:])}")
            if self._delivery_counts:
                parts.append(
                    "delivery: "
                    + ", ".join(f"{k}={v}" for k, v in sorted(self._delivery_counts.items()) if v)
                )
            if self._records:
                parts.append(self.timing().summary())

            causal_chains = self._build_causal_chains_locked()
            recent_completions = self._build_recent_completions_locked()
            discovery_chains = {
                parent: list(children) for parent, children in self._discovery_edges.items()
            }
            if causal_chains:
                rendered = "; ".join(
                    f"{src}→{{{','.join(targets[:4])}}}"
                    for src, targets in list(causal_chains.items())[:4]
                )
                parts.append(f"chains: {rendered}")

            text = "; ".join(parts) if parts else "runtime idle"
            return LedgerDigest(
                text=text,
                seq=self.store.clock.value,
                counts=self.counts(),
                causal_chains=causal_chains,
                recent_completions=recent_completions,
                discovery_chains=discovery_chains,
            )

    def records_snapshot(self) -> dict[str, TaskRecord]:
        with self._lock:
            return dict(self._records)

    def _build_causal_chains_locked(self) -> dict[str, list[str]]:
        """Return source_id → list[target_id] for sources that are completed tasks
        or known components. Only completed sources are surfaced because the
        digest is meant to describe «因为 X 完成而启动了 Y»."""
        chains: dict[str, list[str]] = {}
        for src, targets in self._causal_edges.items():
            rec = self._records.get(src)
            if rec is not None and rec.state != "completed":
                continue
            if not targets:
                continue
            chains[src] = list(targets)
        return chains

    def _build_recent_completions_locked(self) -> list[tuple[str, list[str]]]:
        """Latest completed task_ids paired with the tasks they unblocked.
        Most recent first, capped at 6 entries."""
        completions: list[tuple[float, str, list[str]]] = []
        for rec in self._records.values():
            if rec.state != "completed" or rec.finished_at is None:
                continue
            triggered = self._causal_edges.get(rec.task_id, [])
            completions.append((rec.finished_at, rec.task_id, list(triggered)))
        completions.sort(key=lambda item: item[0], reverse=True)
        return [(task_id, targets) for _, task_id, targets in completions[:6]]

    def _bump(self) -> None:
        pass


def _label(rec: TaskRecord) -> str:
    if rec.waiting_on:
        return f"{rec.task_id} on {'+'.join(rec.waiting_on[:2])}"
    if rec.summary:
        return f"{rec.task_id}({rec.summary[:30]})"
    return rec.task_id
