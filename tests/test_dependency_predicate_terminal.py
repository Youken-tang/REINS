"""Regression tests for ``DependencyPredicate.task_completed`` semantics.

The predicate must satisfy on **any** terminal state (completed / failed /
cancelled / blocked) — not only on success. The dependent's handler is
responsible for inspecting ``result.status`` via ``runtime.collect()`` and
deciding what to do with a non-success outcome (e.g. controller-style fact
replay: feed ``[failed] <summary>`` back to the model so the upper-layer
planner can recover). Blocking dependents on success leads to the path-2
real-world freeze: the now-retired sub_agent_step (v11-C10 replaced it
with agent_loop_step) waited forever on a child file-read that 404'd.


"""
from __future__ import annotations

import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from high_agent.runtime import (
    AgentTaskSpec,
    DependencyPredicate,
    TaskResult,
)
from high_agent.runtime.scheduler import CausalRuntime


class DependencyPredicateTerminalTests(unittest.TestCase):
    def _make_runtime(self, tmp: str, **kwargs) -> CausalRuntime:
        runtime = CausalRuntime(
            workspace_root=tmp,
            strict_nogil=True,
            max_workers=kwargs.pop("max_workers", 4),
            delivery_debounce=kwargs.pop("delivery_debounce", 0.0),
            **kwargs,
        )
        runtime.start()
        self.addCleanup(runtime.shutdown)
        return runtime

    def test_completed_parent_wakes_dependent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self._make_runtime(tmp)
            observed: dict[str, str] = {}

            def parent(ctx):
                return TaskResult.completed("parent-ok")

            def child(ctx):
                results = ctx.runtime.collect(["parent"])
                parent_result = results.get("parent")
                if isinstance(parent_result, TaskResult):
                    observed["parent_status"] = parent_result.status
                    observed["parent_summary"] = parent_result.summary
                return TaskResult.completed("child-ok")

            runtime.submit([
                AgentTaskSpec(task_id="parent", kind="tool", goal="parent", handler=parent),
                AgentTaskSpec(
                    task_id="child",
                    kind="tool",
                    goal="child",
                    dependencies=[DependencyPredicate.task_completed("parent")],
                    handler=child,
                ),
            ])
            self.assertTrue(runtime.wait_all(timeout=2.0))
            self.assertEqual(runtime.ledger.task_state("parent"), "completed")
            self.assertEqual(runtime.ledger.task_state("child"), "completed")
            self.assertEqual(observed.get("parent_status"), "completed")
            self.assertEqual(observed.get("parent_summary"), "parent-ok")

    def test_failed_parent_wakes_dependent_with_failed_status_visible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self._make_runtime(tmp)
            observed: dict[str, str] = {}

            def parent(ctx):
                return TaskResult.failed(
                    "boom: file not found",
                    error_type="filesystem_error",
                )

            def child(ctx):
                results = ctx.runtime.collect(["parent"])
                parent_result = results.get("parent")
                if isinstance(parent_result, TaskResult):
                    observed["parent_status"] = parent_result.status
                    observed["parent_summary"] = parent_result.summary
                    if parent_result.failure_event is not None:
                        observed["parent_error_type"] = parent_result.failure_event.error_type
                # Handler still runs and decides what to do — typically replay
                # `[failed] ...` as a fact for the upper planner. For the test
                # we just complete to prove we WERE woken.
                return TaskResult.completed("child-handled-failure")

            runtime.submit([
                AgentTaskSpec(task_id="parent", kind="tool", goal="parent", handler=parent),
                AgentTaskSpec(
                    task_id="child",
                    kind="tool",
                    goal="child",
                    dependencies=[DependencyPredicate.task_completed("parent")],
                    handler=child,
                ),
            ])
            self.assertTrue(runtime.wait_all(timeout=2.0))
            self.assertEqual(runtime.ledger.task_state("parent"), "failed")
            # Child must have been woken AND completed — the predicate must
            # not gate on success.
            self.assertEqual(runtime.ledger.task_state("child"), "completed")
            self.assertEqual(observed.get("parent_status"), "failed")
            self.assertEqual(observed.get("parent_summary"), "boom: file not found")
            self.assertEqual(observed.get("parent_error_type"), "filesystem_error")

    def test_cancelled_stale_parent_does_not_wake_dependent(self) -> None:
        """``cancel_stale_tasks`` is the explicit exception (cascade-poisoning).

        A task killed by cancel_stale_tasks does
        NOT enter ``_task_terminals`` — the late worker arrival is suppressed
        in ``_drain_completions`` Phase 2 via ``_cancelled_task_ids``. So
        dependents of a cancelled-stale parent must remain waiting (they will
        be torn down on shutdown). This protects against a cancelled parent
        appearing to "succeed late" and waking children based on a poisoned
        result.
        """
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self._make_runtime(tmp)
            release = threading.Event()
            child_started = threading.Event()

            def parent(ctx):
                release.wait(timeout=2.0)
                return TaskResult.completed("never-delivered")

            def child(ctx):
                child_started.set()
                return TaskResult.completed("child")

            runtime.submit([
                AgentTaskSpec(
                    task_id="parent",
                    kind="tool",
                    goal="parent",
                    handler=parent,
                ),
                AgentTaskSpec(
                    task_id="child",
                    kind="tool",
                    goal="child",
                    dependencies=[DependencyPredicate.task_completed("parent")],
                    handler=child,
                ),
            ])

            # Wait for parent to actually start.
            for _ in range(100):
                if runtime.ledger.task_started_at("parent") is not None:
                    break
                time.sleep(0.01)
            self.assertIsNotNone(runtime.ledger.task_started_at("parent"))

            cancelled = runtime.cancel_stale_tasks(max_seconds=0.0)
            self.assertEqual(cancelled, ["parent"])
            self.assertEqual(runtime.ledger.task_state("parent"), "failed")

            # Let the doomed worker actually finish (its result is suppressed).
            release.set()
            time.sleep(0.2)

            # Child must NOT have run: cancel-stale poisons the cascade
            # deliberately.
            self.assertFalse(child_started.is_set())
            self.assertNotIn(runtime.ledger.task_state("child"), {"completed", "running"})

    def test_blocked_parent_wakes_dependent(self) -> None:
        """``blocked`` is a terminal scheduling state — predicate must satisfy.

        A handler may return ``TaskResult.blocked(...)`` when waiting on
        approval or external input. Like ``failed``, this is "the task left
        running"; dependents should be woken so their handler can decide
        what to do with the blocked result (typically: replay as a fact).
        """
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self._make_runtime(tmp)
            observed: dict[str, str] = {}

            def parent(ctx):
                return TaskResult.blocked("awaiting approval")

            def child(ctx):
                results = ctx.runtime.collect(["parent"])
                parent_result = results.get("parent")
                if isinstance(parent_result, TaskResult):
                    observed["parent_status"] = parent_result.status
                return TaskResult.completed("child-saw-blocked")

            runtime.submit([
                AgentTaskSpec(task_id="parent", kind="tool", goal="parent", handler=parent),
                AgentTaskSpec(
                    task_id="child",
                    kind="tool",
                    goal="child",
                    dependencies=[DependencyPredicate.task_completed("parent")],
                    handler=child,
                ),
            ])
            self.assertTrue(runtime.wait_all(timeout=2.0))
            self.assertEqual(runtime.ledger.task_state("parent"), "blocked")
            self.assertEqual(runtime.ledger.task_state("child"), "completed")
            self.assertEqual(observed.get("parent_status"), "blocked")


if __name__ == "__main__":
    unittest.main()
