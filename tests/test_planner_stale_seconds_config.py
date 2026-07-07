""" Tests for configurable AgentRunController.planner_stale_seconds.

Pre-D2 the cap was a hardcoded 120.0 inside ``_cancel_stale_planners``. For
large project-build prompts the model genuinely needs 5-7 minutes on the
first turn, but the controller would kill the planner future at exactly
120s and re-dispatch a fresh one against the unchanged snapshot — observed
in trace as planner_seqs cycling endlessly with the same snapshot_seq and
elapsed_seconds=120.0.

D2 makes the cap configurable on AgentRunController; cli/main.py defaults
it to ``--model-timeout`` so the controller cap matches the underlying
httpx wait budget.
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
from high_agent.cli.main import _planner_stale_seconds

# Reuse the controller test fixtures (FakeAgent / FakeRuntime / etc.)
from tests.test_controller_dedupe_and_completion import _FakeAgent  # type: ignore


def _make_controller(*, planner_stale_seconds: float = 600.0) -> AgentRunController:
    agent = _FakeAgent()
    return AgentRunController(
        agent=agent,
        objective="test",
        messages=[],
        planner_stale_seconds=planner_stale_seconds,
    )


def _stale_request(elapsed: float) -> tuple[concurrent.futures.Future, _PlannerRequest]:
    fut: concurrent.futures.Future = concurrent.futures.Future()
    request = _PlannerRequest(
        request_id=1,
        snapshot_seq=0,
        digest_text="runtime idle",
        messages=[],
    )
    request._started_at = time.monotonic() - elapsed
    return fut, request


class CancelStalePlannerCapTests(unittest.TestCase):
    def test_default_cap_does_not_cancel_under_120s(self) -> None:
        controller = _make_controller(planner_stale_seconds=600.0)
        fut, request = _stale_request(elapsed=130.0)
        in_flight: dict = {fut: request}

        controller._cancel_stale_planners(in_flight)

        self.assertIn(fut, in_flight, "120s elapsed should NOT trigger 600s cap")
        self.assertFalse(fut.cancelled())

    def test_explicit_cap_cancels_long_running_planner(self) -> None:
        controller = _make_controller(planner_stale_seconds=30.0)
        fut, request = _stale_request(elapsed=45.0)
        in_flight: dict = {fut: request}

        controller._cancel_stale_planners(in_flight)

        self.assertNotIn(fut, in_flight)
        # _cancel_stale_planners records the cap on the trace event.
        events = controller.agent.runtime.trace.events
        self.assertTrue(events, "expected planner.timeout trace emission")
        kind, payload = events[-1]
        self.assertEqual(kind, "planner.timeout")
        self.assertEqual(payload.get("stale_cap_seconds"), 30.0)

    def test_cap_under_one_second_clamped(self) -> None:
        # The handler clamps to >= 1.0 to avoid pathological 0s caps that
        # would kill every in-flight planner instantly.
        controller = _make_controller(planner_stale_seconds=0.0)
        fut, request = _stale_request(elapsed=0.5)
        in_flight: dict = {fut: request}

        controller._cancel_stale_planners(in_flight)

        # 0.5s elapsed < 1.0s clamp floor → not cancelled.
        self.assertIn(fut, in_flight)


class CliPlannerStaleSecondsResolutionTests(unittest.TestCase):
    """v11-D2: ``cli/main.py`` resolves planner_stale_seconds defaulting to model_timeout."""

    def _ns(self, **overrides: object) -> argparse.Namespace:
        defaults = {"planner_stale_seconds": None}
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_defaults_to_model_timeout_when_unset(self) -> None:
        args = self._ns()
        value = _planner_stale_seconds(args, config={}, runtime_cfg={}, model_timeout=600.0)
        self.assertEqual(value, 600.0)

    def test_cli_flag_wins_over_config(self) -> None:
        args = self._ns(planner_stale_seconds=45.0)
        value = _planner_stale_seconds(
            args,
            config={},
            runtime_cfg={"planner_stale_seconds": 200.0},
            model_timeout=600.0,
        )
        self.assertEqual(value, 45.0)

    def test_runtime_config_wins_over_default(self) -> None:
        args = self._ns()
        value = _planner_stale_seconds(
            args,
            config={"runtime": {"planner_stale_seconds": 240.0}},
            runtime_cfg={"planner_stale_seconds": 240.0},
            model_timeout=600.0,
        )
        self.assertEqual(value, 240.0)

    def test_invalid_string_falls_back_to_model_timeout(self) -> None:
        args = self._ns()
        value = _planner_stale_seconds(
            args,
            config={},
            runtime_cfg={"planner_stale_seconds": "not-a-number"},
            model_timeout=300.0,
        )
        self.assertEqual(value, 300.0)

    def test_minimum_clamped_to_one_second(self) -> None:
        args = self._ns(planner_stale_seconds=0.0)
        value = _planner_stale_seconds(args, config={}, runtime_cfg={}, model_timeout=600.0)
        # 0.0 is falsy → falls through to model_timeout (not clamped here).
        self.assertEqual(value, 600.0)

        args = self._ns(planner_stale_seconds=0.1)
        value = _planner_stale_seconds(args, config={}, runtime_cfg={}, model_timeout=600.0)
        self.assertEqual(value, 1.0)


if __name__ == "__main__":
    unittest.main()
