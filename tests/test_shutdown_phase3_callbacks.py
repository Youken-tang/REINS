"""Regression test for Phase 3 callback fan-out after shutdown (audit L1).

When _drain_completions runs Phase 3 (callback fan-out) it must not call
on_refill_needed / on_critical_path_progress if shutdown() has flipped
_shutdown=True in the meantime. Those callbacks usually re-enter
runtime.submit(), which in _schedule_locked sees _shutdown and silently
returns; the task is tracked in the ledger as submitted but never runs.
"""
from __future__ import annotations

import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from high_agent.runtime import AgentTaskSpec, TaskResult
from high_agent.runtime.scheduler import CausalRuntime


class ShutdownBetweenPhase2AndCallbacksTests(unittest.TestCase):
    def test_callbacks_not_fired_after_shutdown(self) -> None:
        runtime = CausalRuntime(
            workspace_root="/tmp",
            strict_nogil=True,
            max_workers=2,
            delivery_debounce=0.0,
        )
        runtime.start()

        callback_fired: list[str] = []

        def _on_refill(task_id, digest):
            callback_fired.append(task_id)

        def _on_critical_path(task_id, count, digest):
            callback_fired.append(f"critical:{task_id}")

        runtime.on_refill_needed = _on_refill
        runtime.on_critical_path_progress = _on_critical_path

        # Submit a long-running task, shutdown will cancel it.
        gate = threading.Event()

        def slow(ctx) -> TaskResult:
            gate.wait(timeout=2.0)
            return TaskResult.completed("done")

        runtime.submit([
            AgentTaskSpec(kind="tool", goal="slow", handler=slow, deliverable=True),
        ])

        # Trigger shutdown while task is in-flight, then release the worker.
        # The worker's completion arrives via _on_future_done, which schedules
        # _drain_completions; that call must observe _shutdown=True before
        # firing callbacks.
        runtime.shutdown()
        gate.set()
        # Allow any straggling Phase 3 to run.
        time.sleep(0.1)

        # No callback should have fired after shutdown.
        self.assertEqual(callback_fired, [])


if __name__ == "__main__":
    unittest.main()
