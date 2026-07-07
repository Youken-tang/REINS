"""Tests for AgentRunController cancel_event support.

cancel_event lets an external supervisor (e.g. the benchmark adapter
enforcing a wall-clock budget) ask the controller to bail out without
killing the runtime threads abruptly. A pre-set event must short-circuit
run() on the very first iteration; a None event must keep the historical
behavior intact.
"""

from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from high_agent.agent.controller import AgentRunController


class _FakeTrace:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def emit(self, event: str, **payload: Any) -> None:
        self.events.append((event, payload))

    def emit_typed(self, event: str, **payload: Any) -> None:
        self.emit(event, **payload)


class _FakeRuntime:
    def __init__(self) -> None:
        self.trace = _FakeTrace()
        self.on_refill_needed: Any = None
        self.on_critical_path_progress: Any = None

    def pending_count(self) -> int:
        return 1


class _FakeAgent:
    def __init__(self) -> None:
        self.runtime = _FakeRuntime()
        self._active_action_index: dict[tuple[str, str], int] = {}
        self._active_action_index_lock = threading.Lock()

    def wait_delivery(self, timeout: float | None = None) -> Any:
        # The cancel-event short-circuit path drains ready deliveries before
        # returning. With no real runtime there is nothing to drain — return
        # None immediately so _drain_ready_deliveries() exits its loop.
        return None


class CancelEventShortCircuitTests(unittest.TestCase):
    def test_preset_cancel_event_returns_immediately(self) -> None:
        cancel = threading.Event()
        cancel.set()
        controller = AgentRunController(
            agent=_FakeAgent(),
            objective="cancel-test",
            messages=[],
            max_iterations=10,
            cancel_event=cancel,
        )
        # No planner threads should ever spin up; run() returns the empty
        # candidate after emitting controller.cancelled.
        result = controller.run()
        self.assertEqual(result, "")
        events = [name for name, _ in controller.agent.runtime.trace.events]
        self.assertIn("controller.cancelled", events)

    def test_no_cancel_event_keeps_legacy_behavior(self) -> None:
        # Sanity check: when cancel_event is None, the field default carries
        # through and the controller loop does not crash on the new check.
        # We don't exercise a full run() loop here (would need a real
        # runtime), but constructing without cancel_event must work.
        controller = AgentRunController(
            agent=_FakeAgent(),
            objective="legacy",
            messages=[],
        )
        self.assertIsNone(controller.cancel_event)


if __name__ == "__main__":
    unittest.main()
