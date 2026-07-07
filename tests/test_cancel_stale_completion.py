"""Regression tests for cancel_stale_tasks late-arrival semantics (audit C1).

When ``cancel_stale_tasks`` "kills" a long-running future, the worker thread
keeps running because ``Future.cancel`` is a no-op once the task is RUNNING.
The runtime must not let that worker's eventual completion flip the ledger
back to completed, wake dependents, or emit a delivery — those would all
contradict the timeout that observers already saw.
"""
from __future__ import annotations

import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from high_agent.runtime import AgentTaskSpec, DependencyPredicate, TaskResult
from high_agent.runtime.scheduler import CausalRuntime


def _trace_events(runtime: CausalRuntime) -> list[dict]:
    return list(runtime.trace.events) if hasattr(runtime.trace, "events") else []


class CancelStaleLateCompletionTests(unittest.TestCase):
    def _make_runtime(self, tmp: str) -> CausalRuntime:
        runtime = CausalRuntime(
            workspace_root=tmp,
            strict_nogil=True,
            max_workers=4,
            delivery_debounce=0.01,
        )
        runtime.start()
        self.addCleanup(runtime.shutdown)
        return runtime

    def test_late_completion_does_not_overwrite_failed_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self._make_runtime(tmp)
            release = threading.Event()

            def slow(ctx):
                # Will be flagged stale and "cancelled" before it returns.
                # The future cannot actually be interrupted, so it keeps
                # running until the test releases it.
                release.wait(timeout=2.0)
                return TaskResult.completed("late-success")

            (task_id,) = runtime.submit([
                AgentTaskSpec(kind="tool", goal="slow", handler=slow, deliverable=True),
            ])
            # Wait for the worker to actually pick up the future.
            for _ in range(50):
                if runtime.ledger.task_started_at(task_id) is not None:
                    break
                time.sleep(0.01)
            self.assertIsNotNone(runtime.ledger.task_started_at(task_id))

            # Pretend it has been running too long.
            cancelled = runtime.cancel_stale_tasks(max_seconds=0.0)
            self.assertEqual(cancelled, [task_id])
            self.assertEqual(runtime.ledger.task_state(task_id), "failed")

            # Now release the worker and let its real completion arrive.
            release.set()
            self.assertTrue(runtime.wait_all(timeout=2.0))

            # Ledger must remain failed; no delivery for the cancelled task.
            self.assertEqual(runtime.ledger.task_state(task_id), "failed")
            late_batch = runtime.wait_next_delivery(timeout=0.1)
            self.assertIsNone(late_batch)

    def test_late_completion_does_not_wake_dependents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self._make_runtime(tmp)
            release = threading.Event()
            child_started = threading.Event()

            def parent(ctx):
                release.wait(timeout=2.0)
                return TaskResult.completed("parent-late")

            def child(ctx):
                child_started.set()
                return TaskResult.completed("child")

            parent_id = "parent-task"
            child_id = "child-task"
            runtime.submit([
                AgentTaskSpec(
                    task_id=parent_id,
                    kind="tool",
                    goal="parent",
                    handler=parent,
                    deliverable=True,
                ),
                AgentTaskSpec(
                    task_id=child_id,
                    kind="tool",
                    goal="child",
                    handler=child,
                    dependencies=[DependencyPredicate.task_completed(parent_id)],
                    deliverable=True,
                ),
            ])
            for _ in range(50):
                if runtime.ledger.task_started_at(parent_id) is not None:
                    break
                time.sleep(0.01)

            runtime.cancel_stale_tasks(max_seconds=0.0)
            release.set()
            time.sleep(0.2)

            # Child must NOT have started: parent is failed-by-timeout, so
            # task_completed dependency stays unsatisfied.
            self.assertFalse(child_started.is_set())
            # Child stays waiting (or cancelled if a future change cascades);
            # the key invariant is that it never becomes completed via the
            # poisoned parent.
            self.assertNotEqual(runtime.ledger.task_state(child_id), "completed")


if __name__ == "__main__":
    unittest.main()
