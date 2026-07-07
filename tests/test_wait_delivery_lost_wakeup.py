"""Regression test for wait_next_delivery lost-wakeup with timeout=None (audit C2).

Before the fix, the predicate check happened under self._lock and the wait
happened on self._delivery_cond. A producer that notified between releasing
_lock and the consumer entering wait() would lose the notification, leaving
a wait(None) call sleeping forever.

The fix moved the predicate (mirror flags _delivery_pending and
_delivery_shutdown_signaled) under _delivery_cond so check + wait share
a single lock.
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


class WaitNextDeliveryNoLostWakeupTests(unittest.TestCase):
    def test_unbounded_wait_wakes_on_completion(self) -> None:
        runtime = CausalRuntime(workspace_root="/tmp", strict_nogil=True, max_workers=2)
        runtime.start()
        self.addCleanup(runtime.shutdown)

        gate = threading.Event()

        def slow(ctx) -> TaskResult:
            gate.wait(timeout=2.0)
            return TaskResult.completed("ok")

        runtime.submit([
            AgentTaskSpec(kind="tool", goal="slow", handler=slow, deliverable=True),
        ])

        result_holder: list = []

        def consumer() -> None:
            batch = runtime.wait_next_delivery(timeout=None)
            result_holder.append(batch)

        thread = threading.Thread(target=consumer, daemon=True)
        thread.start()

        # Give the consumer time to enter wait().
        time.sleep(0.05)
        gate.set()
        thread.join(timeout=2.0)

        self.assertFalse(thread.is_alive(), "consumer never woke from wait_next_delivery(None)")
        self.assertEqual(len(result_holder), 1)
        self.assertIsNotNone(result_holder[0])

    def test_unbounded_wait_wakes_on_shutdown(self) -> None:
        runtime = CausalRuntime(workspace_root="/tmp", strict_nogil=True, max_workers=1)
        runtime.start()

        result_holder: list = []

        def consumer() -> None:
            batch = runtime.wait_next_delivery(timeout=None)
            result_holder.append(batch)

        thread = threading.Thread(target=consumer, daemon=True)
        thread.start()

        time.sleep(0.05)
        runtime.shutdown()
        thread.join(timeout=2.0)

        self.assertFalse(thread.is_alive(), "consumer never woke after shutdown")
        # Shutdown wakes the consumer with a None batch.
        self.assertEqual(result_holder, [None])


if __name__ == "__main__":
    unittest.main()
