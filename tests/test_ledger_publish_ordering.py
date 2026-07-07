"""Regression test for atomic ledger publish ordering (audit L3).

Before the fix, _drain_completions's Phase 1 wrote ledger.set_state(...,
"completed") while Phase 2 (under self._cond) was still responsible for
self._running.pop. External observers calling ledger.counts() and then
runtime.pending_count() saw torn views — ledger said completed,
scheduler still had the task in _running.

The fix moves ledger.set_state into Phase 2 AFTER the _running.pop, so
"observer sees ledger=completed" implies "observer sees scheduler
running set without that task". This test instruments ledger.set_state
to assert the invariant at the moment the ledger flip happens.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from high_agent.runtime import AgentTaskSpec, TaskResult
from high_agent.runtime.scheduler import CausalRuntime


class LedgerPublishOrderingTests(unittest.TestCase):
    def test_ledger_publish_happens_after_running_pop(self) -> None:
        runtime = CausalRuntime(
            workspace_root="/tmp",
            strict_nogil=True,
            max_workers=2,
            delivery_debounce=0.0,
        )
        runtime.start()
        self.addCleanup(runtime.shutdown)

        # Pair each set_state(..., "completed") observation with the
        # task_id being committed and the running-set snapshot at that
        # exact moment. The two completions can drain in either order
        # (different worker threads), so we cannot assume submit order.
        observations: list[tuple[str, set[str]]] = []
        real_set_state = runtime.ledger.set_state

        def observing_set_state(task_id: str, state: str, **kwargs):  # type: ignore[no-untyped-def]
            if state == "completed":
                # The scheduler must have already removed `task_id` from
                # `_running` before publishing the completed state.
                observations.append((task_id, set(runtime._running.keys())))
            return real_set_state(task_id, state, **kwargs)

        runtime.ledger.set_state = observing_set_state  # type: ignore[assignment]

        ids = runtime.submit([
            AgentTaskSpec(
                kind="tool",
                goal="t1",
                handler=lambda ctx: TaskResult.completed("done"),
                deliverable=True,
                task_id="task-1",
            ),
            AgentTaskSpec(
                kind="tool",
                goal="t2",
                handler=lambda ctx: TaskResult.completed("done"),
                deliverable=True,
                task_id="task-2",
            ),
        ])
        self.assertEqual(set(ids), {"task-1", "task-2"})
        self.assertTrue(runtime.wait_all(timeout=2))

        self.assertEqual(len(observations), 2)
        for task_id, snapshot in observations:
            self.assertNotIn(
                task_id,
                snapshot,
                f"ledger flipped {task_id}=completed while scheduler still had it in _running",
            )


if __name__ == "__main__":
    unittest.main()
