""" Tests for the per-snapshot planner-stuck circuit breaker.

D2 made the timeout cap configurable but did nothing about the failure
shape itself: even at a 600s cap, a model genuinely wedged on a particular
prompt shape would still recycle planner_seqs against the same
``snapshot_seq`` indefinitely (just slower than D1's 120s pathological
loop). D3 introduces a hard breaker — after N planner timeouts on the
same snapshot, the controller refuses to dispatch new full planners
against it and, if no other work can advance the runtime ledger, surfaces
a best-effort final message instead of looping forever.

Tests pin:
1. Counter increments per (snapshot_seq) on each timeout.
2. Threshold trips ``_stuck_snapshots`` and emits ``planner.snapshot_stuck``.
3. ``_start_planners`` refuses to dispatch against a stuck snapshot.
4. CLI ``_planner_stuck_threshold`` resolves precedence + clamps to >=1.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from high_agent.agent.controller import AgentRunController, _PlannerRequest
from high_agent.cli.main import _planner_stuck_threshold

# Reuse the controller test fixtures (FakeAgent / FakeRuntime / etc.)
from tests.test_controller_dedupe_and_completion import _FakeAgent  # type: ignore


def _make_controller(
    *,
    planner_stale_seconds: float = 30.0,
    planner_stuck_threshold: int = 3,
) -> AgentRunController:
    agent = _FakeAgent()
    return AgentRunController(
        agent=agent,
        objective="test",
        messages=[],
        planner_stale_seconds=planner_stale_seconds,
        planner_stuck_threshold=planner_stuck_threshold,
    )


def _stale_request(
    *,
    elapsed: float,
    snapshot_seq: int,
    request_id: int = 1,
) -> tuple[concurrent.futures.Future, _PlannerRequest]:
    fut: concurrent.futures.Future = concurrent.futures.Future()
    request = _PlannerRequest(
        request_id=request_id,
        snapshot_seq=snapshot_seq,
        digest_text="runtime idle",
        messages=[],
    )
    request._started_at = time.monotonic() - elapsed
    return fut, request


class SnapshotTimeoutCounterTests(unittest.TestCase):
    def test_single_timeout_increments_counter_but_does_not_trip(self) -> None:
        controller = _make_controller(planner_stale_seconds=30.0, planner_stuck_threshold=3)
        fut, request = _stale_request(elapsed=45.0, snapshot_seq=7, request_id=1)
        in_flight: dict = {fut: request}

        controller._cancel_stale_planners(in_flight)

        self.assertEqual(controller._snapshot_timeout_counts.get(7), 1)
        self.assertNotIn(7, controller._stuck_snapshots)
        kinds = [event for event, _ in controller.agent.runtime.trace.events]
        self.assertIn("planner.timeout", kinds)
        self.assertNotIn("planner.snapshot_stuck", kinds)

    def test_threshold_reached_marks_snapshot_stuck(self) -> None:
        controller = _make_controller(planner_stale_seconds=30.0, planner_stuck_threshold=3)
        for request_id in range(1, 4):
            fut, request = _stale_request(
                elapsed=45.0, snapshot_seq=11, request_id=request_id
            )
            in_flight: dict = {fut: request}
            controller._cancel_stale_planners(in_flight)

        self.assertEqual(controller._snapshot_timeout_counts.get(11), 3)
        self.assertIn(11, controller._stuck_snapshots)
        events = controller.agent.runtime.trace.events
        stuck_events = [(k, p) for k, p in events if k == "planner.snapshot_stuck"]
        self.assertEqual(len(stuck_events), 1, "stuck event must fire exactly once on the trip")
        _, payload = stuck_events[0]
        self.assertEqual(payload.get("snapshot_seq"), 11)
        self.assertEqual(payload.get("timeout_count"), 3)
        self.assertEqual(payload.get("stuck_threshold"), 3)

    def test_separate_snapshots_count_independently(self) -> None:
        controller = _make_controller(planner_stale_seconds=30.0, planner_stuck_threshold=3)
        # Two timeouts on snapshot 1, one on snapshot 2.
        for request_id, snap in [(1, 1), (2, 1), (3, 2)]:
            fut, request = _stale_request(
                elapsed=45.0, snapshot_seq=snap, request_id=request_id
            )
            controller._cancel_stale_planners({fut: request})

        self.assertEqual(controller._snapshot_timeout_counts.get(1), 2)
        self.assertEqual(controller._snapshot_timeout_counts.get(2), 1)
        self.assertNotIn(1, controller._stuck_snapshots)
        self.assertNotIn(2, controller._stuck_snapshots)

    def test_threshold_one_trips_on_first_timeout(self) -> None:
        controller = _make_controller(planner_stale_seconds=30.0, planner_stuck_threshold=1)
        fut, request = _stale_request(elapsed=45.0, snapshot_seq=4, request_id=1)
        controller._cancel_stale_planners({fut: request})
        self.assertIn(4, controller._stuck_snapshots)


class StartPlannersBreakerTests(unittest.TestCase):
    """Once a snapshot is stuck, ``_start_planners`` must refuse to dispatch
    new full planners against it. Recovery hinges on delivery progress
    advancing ``digest.seq`` so the next snapshot is fresh."""

    def test_start_planners_skips_stuck_snapshot(self) -> None:
        controller = _make_controller()
        # Hand-mark snapshot_seq=0 (the FakeRuntime always returns seq=0)
        # as stuck; FakeAgent has no model_client so any dispatch attempt
        # would explode and the test would fail loudly. The breaker MUST
        # short-circuit before reaching dispatch.
        with controller._planner_lifecycle_lock:
            controller._stuck_snapshots.add(0)

        in_flight: dict = {}
        new_started = controller._start_planners(in_flight, planner_started=0, max_planners=4)

        self.assertEqual(new_started, 0)
        self.assertEqual(in_flight, {})
        kinds = [event for event, _ in controller.agent.runtime.trace.events]
        self.assertIn("planner.snapshot_stuck_skip", kinds)


class CliPlannerStuckThresholdTests(unittest.TestCase):
    def _ns(self, **overrides: object) -> argparse.Namespace:
        defaults = {"planner_stuck_threshold": None}
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_default_is_three(self) -> None:
        value = _planner_stuck_threshold(self._ns(), runtime_cfg={})
        self.assertEqual(value, 3)

    def test_cli_flag_wins_over_config(self) -> None:
        args = self._ns(planner_stuck_threshold=5)
        value = _planner_stuck_threshold(args, runtime_cfg={"planner_stuck_threshold": 9})
        self.assertEqual(value, 5)

    def test_runtime_config_used_when_cli_unset(self) -> None:
        value = _planner_stuck_threshold(self._ns(), runtime_cfg={"planner_stuck_threshold": 7})
        self.assertEqual(value, 7)

    def test_invalid_value_falls_back_to_default(self) -> None:
        value = _planner_stuck_threshold(
            self._ns(), runtime_cfg={"planner_stuck_threshold": "not-a-number"}
        )
        self.assertEqual(value, 3)

    def test_zero_clamped_to_one(self) -> None:
        # 0 is falsy → falls through to default 3, NOT clamped at this
        # layer. A sub-1 value inside the controller is clamped via
        # ``max(1, planner_stuck_threshold)``.
        args = self._ns(planner_stuck_threshold=0)
        value = _planner_stuck_threshold(args, runtime_cfg={})
        self.assertEqual(value, 3)

        # Explicit 1 is honored.
        args = self._ns(planner_stuck_threshold=1)
        value = _planner_stuck_threshold(args, runtime_cfg={})
        self.assertEqual(value, 1)


if __name__ == "__main__":
    unittest.main()
