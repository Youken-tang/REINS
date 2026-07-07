"""Run-scoped causal task runtime."""

from __future__ import annotations

import concurrent.futures
import heapq
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from threading import Condition, Lock, RLock
from typing import Any

from reins._nogil import ensure_nogil
from reins.runtime.components import ComponentStore
from reins.runtime.context_store import ContextStore
from reins.runtime.ledger import LedgerDigest, RuntimeLedger
from reins.runtime.resource_access import (
    ResourceAccess,
    access_conflicts,
    normalize_component,
    resources_overlap,
)
from reins.runtime.trace import TraceWriter
from reins.runtime.types import (
    AgentTaskSpec,
    ComponentWrite,
    DeliveryBatch,
    DeliveryEvent,
    DependencyPredicate,
    TaskContext,
    TaskHandler,
    TaskResult,
)


@dataclass
class _SuspendedTask:
    """Internal record for a task that returned TaskResult.suspended.


    All access goes through ``self._cond`` — no new locks introduced.
    """

    resume_handler: Any
    awaiting: list[DependencyPredicate]
    suspend_token: str
    snapshot: dict[str, Any] | None
    suspended_result: TaskResult
    cycle: int = 0  # increments on each resume; used by trace and tests


class CausalRuntime:
    """One-run causal scheduler and delivery fact source."""

    def __init__(
        self,
        *,
        max_workers: int = 8,
        workspace_root: str | None = None,
        delivery_debounce: float = 0.05,
        trace_path: str | Path | None = None,
        executors: dict[str, TaskHandler] | None = None,
        strict_nogil: bool = True,
        on_refill_needed: Any = None,
        on_critical_path_progress: Any = None,
        critical_path_fanout: int = 2,
        critical_path_signal_budget: int = 16,
        critical_path_signal_window: float = 30.0,
    ) -> None:
        ensure_nogil(strict=strict_nogil)
        self.max_workers = max(1, int(max_workers))
        self.workspace_root = str(Path(workspace_root or ".").resolve())
        self.delivery_debounce = max(0.0, float(delivery_debounce))
        self.components = ComponentStore()
        self.store = ContextStore()
        self.ledger = RuntimeLedger(store=self.store)
        self.trace = TraceWriter(trace_path)
        self.executors: dict[str, TaskHandler] = dict(executors or {})
        self.on_refill_needed = on_refill_needed
        self.on_critical_path_progress = on_critical_path_progress
        self.critical_path_fanout = max(2, int(critical_path_fanout))
        self.critical_path_signal_budget = max(0, int(critical_path_signal_budget))
        self.critical_path_signal_window = max(0.0, float(critical_path_signal_window))
        self._critical_path_signal_times: list[float] = []

        self._lock = RLock()
        self._cond = Condition(self._lock)
        self._delivery_lock = Lock()
        self._delivery_cond = Condition(self._delivery_lock)
        self._completion_queue: list[tuple[str, TaskResult]] = []
        self._completion_queue_lock = Lock()
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._started = False
        self._shutdown = False

        self._tasks: dict[str, AgentTaskSpec] = {}
        self._waiting: set[str] = set()
        self._ready: list[tuple[float, int, str]] = []  # heapq: (-priority_score, seq, task_id)
        self._ready_set: set[str] = set()
        self._ready_seq = 0
        self._running: dict[str, concurrent.futures.Future[TaskResult]] = {}
        self._running_access: dict[str, ResourceAccess] = {}
        self._results: dict[str, TaskResult] = {}
        # task ids that have reached a terminal state (completed, failed
        # or cancelled). DependencyPredicate.task_completed checks membership
        # here, so dependents wake on terminal regardless of result.status —
        # the dependent handler decides what to do with a [failed] result, the
        # scheduler does not block waking on success. See
        self._task_terminals: set[str] = set()
        # Tasks marked dead by cancel_stale_tasks. The worker future may still
        # complete; Phase 2 of _drain_completions checks this set and skips the
        # state overwrite + dependent wake + delivery so a "stale-cancelled"
        # task cannot retroactively flip back to completed.
        self._cancelled_task_ids: set[str] = set()
        self._dep_reverse: dict[str, set[str]] = {}
        self._component_waiters: dict[str, set[str]] = {}

        # Suspend/resume protocol (). All four structures live
        # in self._cond's lock domain — no new lock added total
        # order preserved).
        # _suspended[task_id] holds the _SuspendedTask record while the task
        # awaits its predicates. _suspended_started_at[task_id] is the
        # monotonic wall clock when the most recent suspend cycle began;
        # _suspended_total_seconds[task_id] accumulates suspend wall-time so
        # cancel_stale_tasks can deduct it from "running" duration.
        # _awaiting_index is a reverse index from each pending awaiting
        # predicate's key() back to the suspended task ids that depend on it,
        # so wake paths can find affected suspended tasks in O(1).
        # _future_registry tracks futures registered via register_future():
        # IO loop / outside threads call _wake_future_done_locked(token) when
        # they observe done. _future_meta carries the (started_at, task_id,
        # model) metadata trace.py needs to compute rtt_ms for
        # ``model.future.resumed`` and stamp ``model.future.suspended`` with
        # the originating task / provider.
        self._suspended: dict[str, _SuspendedTask] = {}
        self._suspended_started_at: dict[str, float] = {}
        self._suspended_total_seconds: dict[str, float] = {}
        self._awaiting_index: dict[tuple, set[str]] = {}
        self._future_registry: dict[str, concurrent.futures.Future[Any]] = {}
        self._future_meta: dict[str, dict[str, Any]] = {}

        self._delivery_seq = 0
        self._batch_seq = 0
        self._ready_delivery: list[DeliveryEvent] = []
        self._buffered_delivery: list[DeliveryEvent] = []
        # Mirror flags written under _delivery_cond. Producers set
        # _delivery_pending=True after appending to _ready_delivery and
        # _delivery_shutdown_signaled=True from shutdown(); the waiter checks
        # them while holding _delivery_cond just before calling wait(), which
        # closes the lost-wakeup window when the producer's notify lands
        # between the waiter's predicate check and its wait() call.
        self._delivery_pending = False
        self._delivery_shutdown_signaled = False

    def start(self) -> None:
        with self._cond:
            if self._started:
                return
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers,
                thread_name_prefix="high-agent",
            )
            self._started = True
            self.trace.emit("runtime.started", max_workers=self.max_workers, workspace_root=self.workspace_root)
            self._schedule_locked()

    def submit(self, tasks: list[AgentTaskSpec]) -> list[str]:
        with self._cond:
            self._ensure_started_locked()
            task_ids = []
            for task in tasks:
                self._normalize_task_locked(task)
                self._tasks[task.task_id] = task
                task_ids.append(task.task_id)
            self._add_implicit_runtime_dependencies_locked(task_ids)
            self._linearize_dependency_cycles_locked()
            for task_id in task_ids:
                task = self._tasks[task_id]
                resource_summary = _resource_access_summary(task.resource_access)
                deps_summary = _dependency_summary(task.dependencies)
                parent_id = task.metadata.get("discovered_by") or task.metadata.get(
                    "parent_task_id"
                )
                if self._dependencies_satisfied_locked(task):
                    self.ledger.add_task(task, "ready")
                    self._enqueue_ready_locked(task.task_id)
                    self.trace.emit_typed(
                        "task.submitted",
                        task_id=task.task_id,
                        kind=task.kind,
                        state="ready",
                        deps=deps_summary,
                        resource_access=resource_summary,
                        parent_id=parent_id,
                    )
                else:
                    self._waiting.add(task.task_id)
                    self._register_dep_reverse_locked(task)
                    waiting_on = self._waiting_reasons_locked(task)
                    self.ledger.add_task(task, "waiting")
                    self.ledger.set_state(task.task_id, "waiting", waiting_on=waiting_on)
                    self.trace.emit_typed(
                        "task.submitted",
                        task_id=task.task_id,
                        kind=task.kind,
                        state="waiting",
                        deps=deps_summary,
                        resource_access=resource_summary,
                        parent_id=parent_id,
                        waiting_on=waiting_on,
                    )
            self._schedule_locked()
            self._cond.notify_all()
            return task_ids

    def add_discovered(self, parent_task_id: str, tasks: list[AgentTaskSpec]) -> list[str]:
        self.trace.emit("task.discovered", parent_task_id=parent_task_id, count=len(tasks))
        for task in tasks:
            task.metadata.setdefault("discovered_by", parent_task_id)
        return self.submit(tasks)

    def wait_next_delivery(self, timeout: float | None = None) -> DeliveryBatch | None:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            # Predicate check + wait must share _delivery_cond, otherwise a
            # producer that notifies between an outer-lock check and the inner
            # wait() call leaves the waiter sleeping forever.
            with self._delivery_cond:
                if self._delivery_pending or self._delivery_shutdown_signaled:
                    break
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                self._delivery_cond.wait(remaining)
                # Loop: re-check the mirror flags under the same lock.

        with self._lock:
            if not self._ready_delivery:
                with self._delivery_cond:
                    self._delivery_pending = False
                return None
            running_count = len(self._running)
            initial_count = len(self._ready_delivery)

        # Conditional debounce: full coalesce only on burst (>=3 ready events)
        if self.delivery_debounce and running_count > 0 and initial_count >= 3:
            scale = min(running_count / 2.0, 3.0) if running_count > 1 else 0.2
            adaptive_wait = self.delivery_debounce * scale
            with self._delivery_cond:
                self._delivery_cond.wait(adaptive_wait)
        elif self.delivery_debounce and running_count > 0:
            with self._delivery_cond:
                self._delivery_cond.wait(min(self.delivery_debounce * 0.2, 0.01))

        with self._lock:
            if not self._ready_delivery:
                with self._delivery_cond:
                    self._delivery_pending = False
                return None
            events = list(self._ready_delivery)
            self._ready_delivery.clear()
            self._batch_seq += 1
            self.ledger.note_delivery("delivered", len(events))
            digest = self.ledger.digest().text
            task_ids = [event.task_id for event in events]
            self.trace.emit(
                "delivery.delivered",
                batch_seq=self._batch_seq,
                task_ids=task_ids,
                count=len(events),
            )
            # REINS ``delivery.batch`` is the schema-named view used by
            # parse_trace.py; ``delivery.delivered`` stays for back-compat with
            # existing trace consumers (tests/test_runtime.py asserts on it).
            self.trace.emit_typed(
                "delivery.batch",
                count=len(events),
                debounce_ms=int(self.delivery_debounce * 1000),
                batch_seq=self._batch_seq,
                task_ids=task_ids,
            )
            with self._delivery_cond:
                self._delivery_pending = False
            return DeliveryBatch(events=events, digest=digest, batch_seq=self._batch_seq)

    def status_digest(self, since_seq: int | None = None) -> LedgerDigest:
        return self.ledger.digest(since_seq=since_seq)

    def collect(self, ids: list[str] | set[str]) -> dict[str, Any]:
        with self._lock:
            out: dict[str, Any] = {}
            for item_id in ids:
                if item_id in self._results:
                    out[item_id] = self._results[item_id]
                else:
                    comp = self.components.get(item_id)
                    if comp is not None:
                        out[item_id] = comp
            return out

    def cancel(self, scope: str | None = None) -> None:
        with self._cond:
            targets = [scope] if scope else list(self._tasks)
            for task_id in targets:
                if task_id in self._waiting:
                    self._waiting.remove(task_id)
                if task_id in self._ready_set:
                    self._dequeue_ready_locked(task_id)
                future = self._running.get(task_id)
                if future is not None:
                    future.cancel()
                if task_id in self._suspended:
                    self._discard_suspended_locked(task_id)
                if task_id in self._tasks:
                    self.ledger.set_state(task_id, "cancelled")
                    self.trace.emit("task.cancelled", task_id=task_id)
            self._cond.notify_all()

    def _discard_suspended_locked(self, task_id: str) -> _SuspendedTask | None:
        """Forget a suspended task's awaiting bookkeeping.

        Used by cancel paths and shutdown. Returns the popped record so the
        caller can decide whether to fire delivery / trace / state flips.
        """
        record = self._suspended.pop(task_id, None)
        self._suspended_started_at.pop(task_id, None)
        self._suspended_total_seconds.pop(task_id, None)
        if record is not None:
            for pred in record.awaiting:
                bucket = self._awaiting_index.get(pred.key())
                if bucket is not None:
                    bucket.discard(task_id)
                    if not bucket:
                        self._awaiting_index.pop(pred.key(), None)
        return record

    def shutdown(self) -> None:
        executor = None
        with self._cond:
            if self._shutdown:
                return
            self._shutdown = True
            # Suspended tasks are neither running nor terminal; without an
            # explicit drain they would leak awaiting_index entries and look
            # "pending" forever to wait_all() / pending_count(). Mark them
            # cancelled in the ledger (so digests are honest) and free their
            # bookkeeping. The futures themselves are owned by callers; we do
            # not call .cancel() here because IO loop futures may be shared
            # across runs.
            for task_id in list(self._suspended.keys()):
                self._discard_suspended_locked(task_id)
                self._cancelled_task_ids.add(task_id)
                self.ledger.set_state(task_id, "cancelled", summary="shutdown")
                self.trace.emit("task.suspended_cancelled_at_shutdown", task_id=task_id)
            self._future_registry.clear()
            self._future_meta.clear()
            executor = self._executor
            self._executor = None
            self._cond.notify_all()
        with self._delivery_cond:
            self._delivery_shutdown_signaled = True
            self._delivery_cond.notify_all()
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        self.trace.emit("runtime.shutdown")
        self.trace.close()

    def wait_all(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while True:
                if (
                    not self._waiting
                    and not self._ready_set
                    and not self._running
                    and not self._suspended
                ):
                    return True
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._cond.wait(remaining)

    def pending_count(self) -> int:
        with self._lock:
            return (
                len(self._waiting)
                + len(self._ready_set)
                + len(self._running)
                + len(self._suspended)
            )

    def wait_tasks(self, task_ids: list[str] | set[str], timeout: float | None = None) -> dict[str, TaskResult]:
        """Wait for specific tasks to complete and return their results."""
        deadline = None if timeout is None else time.monotonic() + timeout
        pending = set(task_ids)
        results: dict[str, TaskResult] = {}
        while pending:
            collected = self.collect(pending)
            for tid, result in collected.items():
                if isinstance(result, TaskResult):
                    results[tid] = result
                    pending.discard(tid)
            if not pending:
                break
            with self._cond:
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._cond.wait(timeout=remaining)
                else:
                    self._cond.wait()
        return results

    def cancel_stale_tasks(self, max_seconds: float = 120.0) -> list[str]:
        """Mark running tasks that exceed max_seconds as failed-by-timeout.

        ``concurrent.futures.Future.cancel`` only succeeds for futures still in
        PENDING. Once a worker has picked up the future its run cannot be
        interrupted from outside, so we cannot truly stop the work; we can only
        record that we have given up on it. To keep the rest of the runtime
        consistent:

        - Ledger state flips to ``failed`` immediately so digests, gate
          decisions and external observers see the timeout.
        - ``_results`` and ``_task_terminals`` are NOT written here; the late
          worker arrival in :meth:`_drain_completions` honours
          ``_cancelled_task_ids`` and skips state overwrite, dependent wake and
          delivery, so a timed-out task cannot resurrect itself as completed.
        - Delivery waiters are still notified so callers blocked on
          ``wait_next_delivery`` can reassess (the failed ledger digest is now
          authoritative even though no DeliveryEvent is enqueued for the
          cancelled task).
        - Suspended tasks count their wall time as ``(now - started_at)`` minus
          accumulated ``_suspended_total_seconds`` and the still-open suspend
          segment, so a task that has been mostly idling does not get harvested
          by a short timeout.

        Returns the task ids that were marked.
        """
        now = time.monotonic()
        cancelled: list[str] = []
        with self._lock:
            candidates: list[tuple[str, concurrent.futures.Future[TaskResult] | None]] = []
            for task_id, future in list(self._running.items()):
                started_at = self.ledger.task_started_at(task_id)
                if started_at is None:
                    continue
                total_suspend = self._suspended_total_seconds.get(task_id, 0.0)
                effective = (now - started_at) - total_suspend
                if effective > max_seconds:
                    candidates.append((task_id, future))
            for task_id in list(self._suspended.keys()):
                started_at = self.ledger.task_started_at(task_id)
                if started_at is None:
                    continue
                total_suspend = self._suspended_total_seconds.get(task_id, 0.0)
                started_segment = self._suspended_started_at.get(task_id, now)
                # The currently-open suspend segment is not yet booked into
                # _suspended_total_seconds; subtract it explicitly so the
                # "running" duration excludes time the task is genuinely idle.
                in_flight_suspend = max(0.0, now - started_segment)
                effective = (now - started_at) - total_suspend - in_flight_suspend
                if effective > max_seconds:
                    candidates.append((task_id, None))
            # Pre-mark BEFORE calling future.cancel(): a successful cancel on
            # a still-PENDING future fires the done_callback synchronously in
            # this thread, which triggers _drain_completions; that path checks
            # _cancelled_task_ids in Phase 2 to skip state overwrite + delivery.
            # If we marked after the cancel call, the synchronous drain would
            # see an empty set and emit a normal cancelled delivery.
            for task_id, _ in candidates:
                self._cancelled_task_ids.add(task_id)
                self._running.pop(task_id, None)
                self._running_access.pop(task_id, None)
                if task_id in self._suspended:
                    self._discard_suspended_locked(task_id)
                self.ledger.set_state(task_id, "failed", summary=f"timeout after {max_seconds:.0f}s")
                self.trace.emit(
                    "task.cancelled_stale",
                    task_id=task_id,
                    timeout_seconds=max_seconds,
                )
            for task_id, future in candidates:
                if future is not None:
                    future.cancel()
                cancelled.append(task_id)
            if cancelled:
                self._schedule_locked()
                self._cond.notify_all()
        if cancelled:
            with self._delivery_cond:
                self._delivery_cond.notify_all()
        return cancelled

    def _ensure_started_locked(self) -> None:
        if not self._started:
            self.start()

    def _normalize_task_locked(self, task: AgentTaskSpec) -> None:
        access = task.resource_access or ResourceAccess(
            reads=frozenset(task.reads),
            writes=frozenset(task.writes),
        )
        task.resource_access = access.normalized(self.workspace_root)
        task.reads = set(task.resource_access.reads)
        task.writes = set(task.resource_access.writes)
        self._add_implicit_directory_dependencies_locked(task)

    def _add_implicit_runtime_dependencies_locked(self, task_ids: list[str]) -> None:
        for task_id in task_ids:
            task = self._tasks[task_id]
            self._add_implicit_test_dependencies_locked(task)

    def _add_implicit_test_dependencies_locked(self, task: AgentTaskSpec) -> None:
        if not _is_run_tests_task(task):
            return
        deps = list(task.dependencies)
        existing = {_task_dep_id(pred) for pred in deps if pred.kind == "task_completed"}
        for other in self._tasks.values():
            if other.task_id == task.task_id or other.task_id in existing:
                continue
            if _is_run_tests_task(other):
                continue
            state = self.ledger.task_state(other.task_id)
            if state in {"completed", "failed", "blocked", "cancelled"}:
                continue
            access = other.resource_access or ResourceAccess.empty()
            if not any(_is_source_write(write) for write in access.writes | access.appends):
                continue
            deps.append(DependencyPredicate.task_completed(other.task_id))
            existing.add(other.task_id)
        task.dependencies = deps

    def _add_implicit_directory_dependencies_locked(self, task: AgentTaskSpec) -> None:
        if not task.resource_access:
            return
        deps = list(task.dependencies)
        for write in task.resource_access.writes:
            kind, path = _path_component(write)
            if kind not in {"file", "dir"}:
                continue
            parent = Path(path).parent if kind == "file" else Path(path)
            if parent.exists():
                continue
            parent_component = f"dir:{parent}"
            if self.components.exists(parent_component):
                continue
            for other in self._tasks.values():
                if _is_run_tests_task(other):
                    continue
                other_access = other.resource_access
                if not other_access:
                    continue
                if other.task_id == task.task_id or self.ledger.task_state(other.task_id) == "completed":
                    continue
                for other_write in other_access.writes:
                    if other_write.startswith("dir:") and resources_overlap(other_write, parent_component):
                        deps.append(DependencyPredicate.task_completed(other.task_id))
        task.dependencies = deps

    def _dependencies_satisfied_locked(self, task: AgentTaskSpec) -> bool:
        return all(self._predicate_satisfied_locked(pred) for pred in task.dependencies)

    def _predicate_satisfied_locked(self, pred: DependencyPredicate) -> bool:
        if pred.kind == "exists" and pred.component:
            return self.components.exists(normalize_component(pred.component, self.workspace_root))
        if pred.kind == "version_at_least" and pred.component:
            comp = normalize_component(pred.component, self.workspace_root)
            return self.components.version(comp) >= int(pred.min_version or 0)
        if pred.kind == "task_completed" and pred.task_id:
            return pred.task_id in self._task_terminals
        if pred.kind == "task_terminal" and pred.task_id:
            # Explicit alias: scheduler-visibility view ("task left running") —
            # currently identical to task_completed (which after path2-C1 is
            # also terminal-on-any-status). Spelled separately so handlers can
            # signal intent independently of the legacy name.
            return pred.task_id in self._task_terminals
        if pred.kind == "future_done" and pred.token:
            future = self._future_registry.get(pred.token)
            return future is not None and future.done()
        if pred.kind == "all":
            return all(self._predicate_satisfied_locked(child) for child in pred.children)
        if pred.kind == "any":
            return any(self._predicate_satisfied_locked(child) for child in pred.children)
        return False

    def _waiting_reasons_locked(self, task: AgentTaskSpec) -> list[str]:
        reasons: list[str] = []
        for pred in task.dependencies:
            if not self._predicate_satisfied_locked(pred):
                reasons.append(_predicate_label(pred))
        return reasons

    def _wake_waiting_locked(self) -> None:
        for task_id in list(self._waiting):
            task = self._tasks[task_id]
            if self._dependencies_satisfied_locked(task):
                self._waiting.remove(task_id)
                self._enqueue_ready_locked(task_id)
                self._unregister_dep_reverse_locked(task)
                self.ledger.set_state(task_id, "ready", waiting_on=[])
                self.trace.emit("task.woken", task_id=task_id)
            else:
                self.ledger.set_state(task_id, "waiting", waiting_on=self._waiting_reasons_locked(task))

    def _wake_dependents_of_task_locked(self, completed_task_id: str) -> list[str]:
        """Event-driven wake: only check tasks that depend on the completed task.

        Returns the list of task_ids that transitioned waiting → ready as a
        direct consequence of this completion. The caller uses the size of this
        list to decide whether to fire the critical-path planner-refill hook.
        """
        candidates = self._dep_reverse.pop(completed_task_id, set())
        woken: list[str] = []
        for task_id in candidates:
            if task_id not in self._waiting:
                continue
            task = self._tasks[task_id]
            if self._dependencies_satisfied_locked(task):
                self._waiting.remove(task_id)
                self._enqueue_ready_locked(task_id)
                self._unregister_dep_reverse_locked(task)
                self.ledger.record_trigger(completed_task_id, task_id)
                self.ledger.set_state(task_id, "ready", waiting_on=[])
                self.trace.emit("task.woken", task_id=task_id, triggered_by=completed_task_id)
                woken.append(task_id)
        # Suspended tasks awaiting either kind of "task is done" predicate on
        # this completed task may now have all predicates satisfied. The
        # _awaiting_index lookup is keyed by predicate identity, so we probe
        # both task_completed and task_terminal flavours and resume any task
        # whose awaiting list is fully satisfied.
        for pred_kind in ("task_completed", "task_terminal"):
            key = (pred_kind, None, None, completed_task_id, None, None)
            self._maybe_resume_suspended_for_key_locked(key, completed_task_id)
        return woken

    def _maybe_resume_suspended_for_key_locked(self, key: tuple, trigger_id: str) -> None:
        """Resume any suspended task whose awaiting predicates are all satisfied.

        ``key`` is the predicate key that just became satisfied; we use it as
        the reverse-index entry, but we still re-check the full awaiting list
        for each candidate so suspended tasks awaiting on multiple predicates
        only resume when the last one fires.
        """
        candidates = self._awaiting_index.get(key)
        if not candidates:
            return
        ready_to_resume: list[str] = []
        for task_id in list(candidates):
            record = self._suspended.get(task_id)
            if record is None:
                continue
            if all(self._predicate_satisfied_locked(p) for p in record.awaiting):
                ready_to_resume.append(task_id)
        for task_id in ready_to_resume:
            self._resume_suspended_locked(task_id, trigger_id=trigger_id)

    def _resume_suspended_locked(self, task_id: str, *, trigger_id: str) -> None:
        """Move a suspended task back to ready and rebind its handler.

        The accumulated suspend wall-time is added to ``_suspended_total_seconds``
        so ``cancel_stale_tasks`` can deduct it from the running duration. The
        original ``handler`` is replaced with a thunk that calls the
        ``resume_handler`` with ``(prev_result, ctx)`` per the protocol in

        """
        record = self._suspended.pop(task_id, None)
        if record is None:
            return
        started = self._suspended_started_at.pop(task_id, None)
        if started is not None:
            self._suspended_total_seconds[task_id] = (
                self._suspended_total_seconds.get(task_id, 0.0)
                + max(0.0, time.monotonic() - started)
            )
        for pred in record.awaiting:
            bucket = self._awaiting_index.get(pred.key())
            if bucket is not None:
                bucket.discard(task_id)
                if not bucket:
                    self._awaiting_index.pop(pred.key(), None)
        record.cycle += 1
        task = self._tasks.get(task_id)
        if task is None:
            return
        prev_result = record.suspended_result
        resume_handler = record.resume_handler

        def _resumed_handler(ctx: TaskContext, _resume=resume_handler, _prev=prev_result) -> TaskResult:
            return _resume(_prev, ctx)

        task.handler = _resumed_handler
        self._enqueue_ready_locked(task_id)
        self.ledger.record_trigger(trigger_id, task_id)
        self.ledger.set_state(task_id, "ready", waiting_on=[])
        suspended_seconds = round(
            self._suspended_total_seconds.get(task_id, 0.0), 6
        )
        self.trace.emit_typed(
            "task.resumed",
            task_id=task_id,
            triggered_by=trigger_id,
            suspended_seconds=suspended_seconds,
            trigger=trigger_id,
        )

    def _wake_future_done_locked(self, token: str) -> None:
        """Resume suspended tasks whose awaiting future_done(token) is satisfied.

        This entry point is intended for the IO loop thread (C6) to call when a
        registered future completes. The caller MUST hold ``self._cond``.
        """
        meta = self._future_meta.pop(token, None)
        if meta is not None:
            started_at = float(meta.get("started_at") or 0.0)
            rtt_ms = int(max(0.0, (time.monotonic() - started_at) * 1000)) if started_at else 0
            self.trace.emit_typed(
                "model.future.resumed",
                token=token,
                rtt_ms=rtt_ms,
                task_id=meta.get("task_id"),
            )
        key = ("future_done", None, None, None, None, token)
        self._maybe_resume_suspended_for_key_locked(key, trigger_id=f"future:{token}")
        self._schedule_locked()
        self._cond.notify_all()

    def register_future(
        self,
        future: concurrent.futures.Future[Any],
        *,
        task_id: str | None = None,
        model: str | None = None,
        est_rtt: float | None = None,
    ) -> str:
        """Register an external future and return a token for ``future_done``.

        When the future completes, the IO loop / outside thread should call
        ``_wake_future_done_locked(token)`` under ``self._cond`` to resume any
        task awaiting on it. The runtime also installs a ``done_callback`` to
        cover the common case where no external code orchestrates the wake.

        Optional ``task_id`` / ``model`` / ``est_rtt`` are recorded in
        ``_future_meta`` so trace.py
        events can stamp the originating task and provider, and so the resume
        path can compute ``rtt_ms`` without the caller needing a wall clock.
        """
        token = uuid.uuid4().hex
        started_at = time.monotonic()
        with self._cond:
            self._future_registry[token] = future
            self._future_meta[token] = {
                "started_at": started_at,
                "task_id": task_id,
                "model": model,
                "est_rtt": est_rtt,
            }
        self.trace.emit_typed(
            "model.future.suspended",
            token=token,
            model=model,
            est_rtt=est_rtt,
            task_id=task_id,
        )

        def _on_done(_fut: concurrent.futures.Future[Any], _token: str = token) -> None:
            with self._cond:
                self._wake_future_done_locked(_token)

        future.add_done_callback(_on_done)
        return token

    def _wake_dependents_of_component_locked(self, component_id: str) -> None:
        """Event-driven wake for component exists/version predicates."""
        candidates = self._component_waiters.pop(component_id, set())
        for task_id in candidates:
            if task_id not in self._waiting:
                continue
            task = self._tasks[task_id]
            if self._dependencies_satisfied_locked(task):
                self._waiting.remove(task_id)
                self._enqueue_ready_locked(task_id)
                self._unregister_dep_reverse_locked(task)
                self.ledger.record_trigger(component_id, task_id)
                self.ledger.set_state(task_id, "ready", waiting_on=[])
                self.trace.emit("task.woken", task_id=task_id, triggered_by=component_id)

    def _register_dep_reverse_locked(self, task: AgentTaskSpec) -> None:
        for pred in task.dependencies:
            self._register_pred_reverse_locked(task.task_id, pred)

    def _register_pred_reverse_locked(self, waiter_id: str, pred: DependencyPredicate) -> None:
        if pred.kind == "task_completed" and pred.task_id:
            self._dep_reverse.setdefault(pred.task_id, set()).add(waiter_id)
        elif pred.kind in ("exists", "version_at_least") and pred.component:
            comp = normalize_component(pred.component, self.workspace_root)
            self._component_waiters.setdefault(comp, set()).add(waiter_id)
        elif pred.kind in ("all", "any"):
            for child in pred.children:
                self._register_pred_reverse_locked(waiter_id, child)

    def _unregister_dep_reverse_locked(self, task: AgentTaskSpec) -> None:
        for pred in task.dependencies:
            self._unregister_pred_reverse_locked(task.task_id, pred)

    def _unregister_pred_reverse_locked(self, waiter_id: str, pred: DependencyPredicate) -> None:
        if pred.kind == "task_completed" and pred.task_id:
            waiters = self._dep_reverse.get(pred.task_id)
            if waiters:
                waiters.discard(waiter_id)
        elif pred.kind in ("exists", "version_at_least") and pred.component:
            comp = normalize_component(pred.component, self.workspace_root)
            waiters = self._component_waiters.get(comp)
            if waiters:
                waiters.discard(waiter_id)
        elif pred.kind in ("all", "any"):
            for child in pred.children:
                self._unregister_pred_reverse_locked(waiter_id, child)

    def _linearize_dependency_cycles_locked(self) -> None:
        graph = {task_id: _task_dependency_ids(task) & set(self._tasks) for task_id, task in self._tasks.items()}
        components = _strongly_connected_components(graph)
        for component in components:
            if len(component) <= 1:
                continue
            ordered = sorted(component)
            component_set = set(ordered)
            for index, task_id in enumerate(ordered):
                task = self._tasks[task_id]
                task.dependencies = [
                    pred for pred in task.dependencies
                    if not (pred.kind == "task_completed" and pred.task_id in component_set)
                ]
                if index > 0:
                    task.dependencies.append(DependencyPredicate.task_completed(ordered[index - 1]))
                task.metadata["causal_boundary"] = "cycle_linearized"
            self.trace.emit("dependency.cycle_linearized", task_ids=ordered)

    def _enqueue_ready_locked(self, task_id: str) -> None:
        if task_id in self._ready_set:
            return
        task = self._tasks[task_id]
        dependents = len(self._dep_reverse.get(task_id, set()))
        score = -(task.priority * 1000 + dependents)
        self._ready_seq += 1
        heapq.heappush(self._ready, (score, self._ready_seq, task_id))
        self._ready_set.add(task_id)
        # task.ready captures the wait time between
        # submission and ready (post-dependency / post-resume). For tasks
        # that flip from waiting/suspended → ready this is the time spent
        # blocked; for tasks that go submitted → ready directly it is ~0.
        created_at = self.ledger.task_created_at(task_id)
        wait_seconds = (
            round(time.monotonic() - created_at, 6) if created_at is not None else 0.0
        )
        self.trace.emit_typed(
            "task.ready",
            task_id=task_id,
            wait_seconds=wait_seconds,
        )

    def _dequeue_ready_locked(self, task_id: str) -> None:
        self._ready_set.discard(task_id)

    def _schedule_locked(self) -> None:
        if not self._executor or self._shutdown:
            return
        skipped: list[tuple[float, int, str]] = []
        while len(self._running) < self.max_workers and self._ready:
            score, seq, task_id = heapq.heappop(self._ready)
            if task_id not in self._ready_set:
                continue
            task = self._tasks[task_id]
            access = task.resource_access or ResourceAccess.empty()
            conflict = self._first_conflict_locked(access)
            if conflict:
                self.ledger.note_conflict(task_id, conflict)
                conflicting = self._running_access.get(conflict)
                self.trace.emit_typed(
                    "resource.conflict",
                    task_id=task_id,
                    conflict=conflict,
                    attempted_access=_resource_access_summary(access),
                    running_access=_resource_access_summary(conflicting),
                )
                skipped.append((score, seq, task_id))
                continue
            self._ready_set.discard(task_id)
            self._running[task_id] = self._executor.submit(self._run_task, task)
            self._running_access[task_id] = access
            self.ledger.set_state(task_id, "running")
            self.trace.emit_typed("task.started", task_id=task_id, kind=task.kind)
            self._running[task_id].add_done_callback(lambda fut, tid=task_id: self._on_future_done(tid, fut))
        for item in skipped:
            heapq.heappush(self._ready, item)

    def _first_conflict_locked(self, access: ResourceAccess) -> str | None:
        for running_id, running_access in self._running_access.items():
            if access_conflicts(access, running_access):
                return running_id
        return None

    def _run_task(self, task: AgentTaskSpec) -> TaskResult:
        handler = task.handler or self.executors.get(task.kind)
        if handler is None:
            return TaskResult.completed(task.goal)
        context = TaskContext(task=task, runtime=self, ledger_digest=self.ledger.digest().text)
        try:
            return handler(context)
        except Exception as exc:
            return TaskResult.failed(str(exc), error_type=type(exc).__name__)

    def _on_future_done(self, task_id: str, future: concurrent.futures.Future[TaskResult]) -> None:
        try:
            result = future.result()
        except concurrent.futures.CancelledError:
            result = TaskResult(status="cancelled", summary="cancelled")
        except Exception as exc:
            result = TaskResult.failed(str(exc), error_type=type(exc).__name__)

        with self._completion_queue_lock:
            self._completion_queue.append((task_id, result))

        self._drain_completions()

    def _drain_completions(self) -> None:
        with self._completion_queue_lock:
            batch = list(self._completion_queue)
            self._completion_queue.clear()
        if not batch:
            return

        # ── Phase 1 (lock-free): component writes only ──
        # Component writes go through ComponentStore's own RLock and are
        # consumed in Phase 2 only via component_ids → dependent wake. They
        # are safe to publish before Phase 2 because dependents do not
        # observe the component's "live" value until the wake fires.
        # Ledger.set_state and the matching task.completed trace are NOT
        # done here anymore — see Phase 2.
        prepared: list[tuple[AgentTaskSpec, TaskResult, list[str]]] = []
        for task_id, result in batch:
            task = self._tasks[task_id]
            # cancel_stale_tasks already wrote the ledger as failed and removed
            # the task from _running. The worker's late result must not flip
            # the ledger or fire a delivery — it would contradict the timeout
            # that callers already observed. Drop both Phase 1 and Phase 2
            # work for cancelled task ids; we only emit a trace marker so the
            # late arrival is still visible.
            with self._cond:
                already_cancelled = task_id in self._cancelled_task_ids
            if already_cancelled:
                self.trace.emit_typed(
                    "task.late_after_cancel",
                    task_id=task_id,
                    status=result.status,
                )
                continue
            component_ids = self._apply_component_writes(task, result)
            prepared.append((task, result, component_ids))

        # ── Phase 2 (self._cond): atomic ledger publish + state machine + delivery + scheduling ──
        # ledger.set_state used to live in Phase 1, but that left
        # a race window where ledger said "completed" while scheduler still
        # had the task in _running. Observers calling ledger.counts() then
        # pending_count() (or the converse) saw torn views, and digest()
        # delivered to refill callbacks reflected an inconsistent state.
        # Flip the ledger inside self._cond, AFTER scheduler bookkeeping,
        # so the atomic publish order observers see is "scheduler dropped
        # the task → ledger committed the new state" — never the reverse.
        signals: list[tuple[bool, tuple[str, int] | None]] = []
        last_task_id: str | None = None
        delivery_enqueued = False
        ready_delivery_size_before: int
        trace_records: list[tuple[str, dict[str, Any]]] = []
        with self._cond:
            ready_delivery_size_before = len(self._ready_delivery)
            now = time.monotonic()
            if self.critical_path_signal_window > 0:
                self._critical_path_signal_times = [
                    t for t in self._critical_path_signal_times
                    if now - t <= self.critical_path_signal_window
                ]
            for task, result, component_ids in prepared:
                task_id = task.task_id
                last_task_id = task_id
                # Suspend handling MUST run before "1) Scheduler bookkeeping"
                # below: a suspended task does NOT get added to _task_terminals,
                # does NOT enqueue delivery, and does NOT release dependents.
                # Drop from _running but stash in _suspended + _awaiting_index
                # so wake paths can resume it when its predicates satisfy.
                if result.status == "suspended":
                    self._running.pop(task_id, None)
                    self._running_access.pop(task_id, None)
                    self._suspended[task_id] = _SuspendedTask(
                        resume_handler=result.resume_handler,
                        awaiting=list(result.awaiting),
                        suspend_token=result.suspend_token,
                        snapshot=result.snapshot,
                        suspended_result=result,
                    )
                    self._suspended_started_at[task_id] = time.monotonic()
                    self._suspended_total_seconds.setdefault(task_id, 0.0)
                    for pred in result.awaiting:
                        self._awaiting_index.setdefault(pred.key(), set()).add(task_id)
                    self.ledger.set_state(
                        task_id,
                        "suspended",
                        summary=result.summary or f"awaiting {len(result.awaiting)} predicate(s)",
                    )
                    duration_seconds = self.ledger.task_duration(task_id) or 0.0
                    snapshot_size = 0
                    if isinstance(result.snapshot, dict):
                        try:
                            snapshot_size = sum(
                                len(str(k)) + len(str(v)) for k, v in result.snapshot.items()
                            )
                        except Exception:
                            snapshot_size = 0
                    trace_records.append((
                        "task.suspended",
                        {
                            "task_id": task_id,
                            "suspend_token": result.suspend_token,
                            "awaiting": [p.kind for p in result.awaiting],
                            "snapshot_size": snapshot_size,
                        },
                    ))
                    # Race: if every awaiting predicate is *already* satisfied
                    # at suspend time (e.g. the child task completed before
                    # this drain reached us), the wake event has already passed
                    # and there is no future wake that will retry. Resume
                    # synchronously so we do not deadlock waiting for an event
                    # that already happened. _resume_suspended_locked is safe
                    # to call here because it only mutates state inside
                    # ``self._cond`` and this branch is already inside it.
                    if all(self._predicate_satisfied_locked(p) for p in result.awaiting):
                        self._resume_suspended_locked(task_id, trigger_id="suspend.synchronous")
                    continue
                # 1) Scheduler bookkeeping — remove from running, publish result.
                self._running.pop(task_id, None)
                self._running_access.pop(task_id, None)
                self._results[task_id] = result
                # task_completed predicate satisfies on any terminal state
                # (completed / failed / cancelled / blocked). The dependent's
                # handler reads result.status from runtime.collect() and decides
                # how to proceed; the scheduler must not block waking on
                # success — that produces permanent waiting when children fail
                # (the path-2 stop-gap originally added for the retired
                # sub_agent_step kind; the same semantics now serve
                # agent_loop_step).
                self._task_terminals.add(task_id)
                # 2) Ledger flip — same lock domain so ledger digest and
                #    pending_count() now agree at every observation point.
                if result.status == "completed":
                    self.ledger.set_state(task_id, "completed", summary=result.summary)
                elif result.status in {"blocked", "partial"}:
                    self.ledger.set_state(task_id, "blocked", summary=result.summary)
                elif result.status == "cancelled":
                    self.ledger.set_state(task_id, "cancelled", summary=result.summary)
                else:
                    self.ledger.set_state(task_id, "failed", summary=result.summary, failure=result.failure_event)
                duration_seconds = self.ledger.task_duration(task_id) or 0.0
                # Buffer trace records and emit them outside the lock in
                # Phase 3 — trace I/O should not be on the hot scheduler path.
                trace_records.append((
                    "task.completed",
                    {
                        "task_id": task_id,
                        "status": result.status,
                        "summary": result.summary,
                        "duration_seconds": round(duration_seconds, 6),
                    },
                ))
                for cid in component_ids:
                    self._wake_dependents_of_component_locked(cid)
                if result.discovered_tasks:
                    for discovered in result.discovered_tasks:
                        discovered.metadata.setdefault("discovered_by", task_id)
                    # submit() 内部重入 self._cond (RLock)，安全。
                    discovered_ids = self.submit(result.discovered_tasks)
                    for discovered_id in discovered_ids:
                        self.ledger.record_discovery(task_id, discovered_id)
                woken_ids = self._wake_dependents_of_task_locked(task_id)
                self._enqueue_delivery_locked(task, result)

                should_refill = (
                    result.status == "completed"
                    and not self._ready_set
                    and not self._waiting
                    and not self._running
                    and self.on_refill_needed is not None
                )
                critical_signal: tuple[str, int] | None = None
                if (
                    result.status == "completed"
                    and self.on_critical_path_progress is not None
                    and len(self._critical_path_signal_times) < self.critical_path_signal_budget
                    and len(woken_ids) >= self.critical_path_fanout
                ):
                    self._critical_path_signal_times.append(time.monotonic())
                    critical_signal = (task_id, len(woken_ids))
                signals.append((should_refill, critical_signal))

            self._schedule_locked()
            self._cond.notify_all()
            delivery_enqueued = len(self._ready_delivery) > ready_delivery_size_before

        # ── Phase 3 (lock-free): notify delivery waiters + emit trace + fire callbacks ──
        for event_name, payload in trace_records:
            self.trace.emit_typed(event_name, **payload)
        with self._delivery_cond:
            # Mirror flag the waiter checks under _delivery_cond. Phase 2
            # determined whether new events landed in _ready_delivery; we
            # cannot reach back into _lock here without inverting the lock
            # order observed in wait_next_delivery (_lock → _delivery_cond),
            # which would deadlock.
            if delivery_enqueued:
                self._delivery_pending = True
            self._delivery_cond.notify_all()
        # shutdown() may have flipped _shutdown=True between Phase 2
        # releasing the lock and us reaching the callback fan-out. Firing
        # callbacks now lets controllers re-enter submit(), which would land
        # in _schedule_locked's `if self._shutdown: return` and silently
        # accept a task that will never run. Snapshot _shutdown once and
        # skip the callbacks if the runtime is going down.
        with self._cond:
            shutting_down = self._shutdown
        if shutting_down:
            return
        for should_refill, critical_signal in signals:
            if critical_signal is not None:
                try:
                    self.on_critical_path_progress(critical_signal[0], critical_signal[1], self.ledger.digest())
                except Exception:
                    pass
            if should_refill:
                try:
                    self.on_refill_needed(last_task_id, self.ledger.digest())
                except Exception:
                    pass

    def _apply_component_writes(self, task: AgentTaskSpec, result: TaskResult) -> list[str]:
        """Apply component writes outside scheduler lock.

        Returns the list of component_ids that were written, so the caller can
        wake component-dependents under self._cond. self.components.apply has
        its own RLock, so calling this lock-free is safe.
        """
        explicit = {write.component_id for write in result.component_writes}
        writes = list(result.component_writes)
        if result.status == "completed":
            for component_id in task.writes:
                if component_id not in explicit:
                    writes.append(ComponentWrite(component_id=component_id, value=result.summary or True))
        elif result.status == "failed":
            failure_id = f"failure:{task.task_id}"
            if failure_id not in explicit:
                writes.append(ComponentWrite(component_id=failure_id, value=result.failure_event or result.summary))

        component_ids: list[str] = []
        for write in writes:
            normalized = ComponentWrite(
                component_id=normalize_component(write.component_id, self.workspace_root),
                value=write.value,
                mode=write.mode,
            )
            comp = self.components.apply(task.task_id, normalized)
            self.ledger.note_component(comp.component_id, comp.version)
            self.trace.emit(
                "component.updated",
                task_id=task.task_id,
                component=comp.component_id,
                version=comp.version,
            )
            component_ids.append(comp.component_id)
        return component_ids

    def _enqueue_delivery_locked(self, task: AgentTaskSpec, result: TaskResult) -> None:
        if not task.deliverable or not result.deliverable:
            return
        self._delivery_seq += 1
        event = DeliveryEvent(
            seq=self._delivery_seq,
            task_id=task.task_id,
            kind=task.kind,
            summary=result.summary,
            result=result,
            barrier=task.barrier,
            metadata={
                **dict(task.metadata),
                "duration_seconds": round(self.ledger.task_duration(task.task_id) or 0.0, 6),
            },
        )
        if task.barrier != "none":
            self._ready_delivery.append(event)
            self.ledger.note_delivery("ready")
            self.trace.emit("delivery.ready", task_id=task.task_id, barrier=task.barrier)
            self._flush_buffered_if_unblocked_locked()
            return
        if self._has_active_barrier_locked():
            self._buffered_delivery.append(event)
            self.ledger.note_delivery("buffered")
            self.trace.emit("delivery.buffered", task_id=task.task_id)
        else:
            self._ready_delivery.append(event)
            self.ledger.note_delivery("ready")
            self.trace.emit("delivery.ready", task_id=task.task_id)

    def _has_active_barrier_locked(self) -> bool:
        active = self._waiting | self._ready_set | set(self._running) | set(self._suspended)
        return any(self._tasks[task_id].barrier != "none" for task_id in active)

    def _flush_buffered_if_unblocked_locked(self) -> None:
        if self._has_active_barrier_locked() or not self._buffered_delivery:
            return
        self._ready_delivery.extend(self._buffered_delivery)
        self.trace.emit("delivery.buffer_flushed", count=len(self._buffered_delivery))
        self._buffered_delivery.clear()


def _predicate_label(pred: DependencyPredicate) -> str:
    if pred.component:
        return normalize_component(pred.component)
    if pred.task_id:
        return f"task:{pred.task_id}"
    if pred.children:
        return f"{pred.kind}({len(pred.children)})"
    return pred.kind


def _resource_access_summary(access: ResourceAccess | None) -> dict[str, Any] | None:
    """Compact ResourceAccess view for trace.py

    Lists are sorted so trace JSONL stays deterministic across runs (parse_trace
    joins on resource_access for F2 ``conflict_density`` characterization).
    """
    if access is None:
        return None
    return {
        "reads": sorted(access.reads),
        "writes": sorted(access.writes),
        "appends": sorted(getattr(access, "appends", set()) or set()),
        "unknown": bool(getattr(access, "unknown", False)),
        "side_effect_level": str(getattr(access, "side_effect_level", "none") or "none"),
    }


def _dependency_summary(deps: list[DependencyPredicate]) -> list[dict[str, Any]]:
    """Compact dependency view for trace.py"""
    out: list[dict[str, Any]] = []
    for pred in deps:
        item: dict[str, Any] = {"kind": pred.kind}
        if pred.task_id:
            item["task_id"] = pred.task_id
        if pred.component:
            item["component"] = pred.component
        if getattr(pred, "token", None):
            item["token"] = pred.token
        if pred.children:
            item["children"] = len(pred.children)
        out.append(item)
    return out


def _path_component(component: str) -> tuple[str | None, str]:
    if component.startswith("file:"):
        return "file", component[5:]
    if component.startswith("dir:"):
        return "dir", component[4:]
    return None, component


def _task_dependency_ids(task: AgentTaskSpec) -> set[str]:
    out: set[str] = set()
    for pred in task.dependencies:
        out.update(_predicate_task_ids(pred))
    return out


def _task_dep_id(pred: DependencyPredicate) -> str:
    return pred.task_id or ""


def _is_run_tests_task(task: AgentTaskSpec) -> bool:
    return (task.input or {}).get("name") == "run_tests" or task.metadata.get("tool_name") == "run_tests"


def _is_source_write(resource: str) -> bool:
    if not resource.startswith("file:"):
        return False
    path = resource[5:]
    cache_tokens = (
        "/.pytest_cache",
        "/htmlcov",
        "/build",
        "/dist",
        "/.tox",
        "/.nox",
        "/__pycache__",
    )
    if any(token in path for token in cache_tokens):
        return False
    if path.endswith((".pyc", ".pyo", ".coverage")):
        return False
    return True


def _predicate_task_ids(pred: DependencyPredicate) -> set[str]:
    if pred.kind == "task_completed" and pred.task_id:
        return {pred.task_id}
    out: set[str] = set()
    for child in pred.children:
        out.update(_predicate_task_ids(child))
    return out


def _strongly_connected_components(graph: dict[str, set[str]]) -> list[list[str]]:
    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    components: list[list[str]] = []

    def strongconnect(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for child in graph.get(node, set()):
            if child not in indices:
                strongconnect(child)
                lowlinks[node] = min(lowlinks[node], lowlinks[child])
            elif child in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[child])
        if lowlinks[node] == indices[node]:
            component: list[str] = []
            while True:
                child = stack.pop()
                on_stack.remove(child)
                component.append(child)
                if child == node:
                    break
            components.append(component)

    for node in graph:
        if node not in indices:
            strongconnect(node)
    return components
