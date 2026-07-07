""" Tests for async planner dispatch on AgentRunController.

C9 routes ``_run_planner_request`` through ``ModelClient.complete_streaming_async`` /
``complete_async`` so the LLM HTTP RTT runs on the IO loop's shared async
executor instead of dedicating a planner ThreadPoolExecutor thread for the
duration of the round-trip. The dedicated ``planner_executor`` is removed.

These tests verify the dispatch contract:

- Streaming async path: ``complete_streaming_async`` is preferred; the
  ``on_tool_call`` callback wired by the controller fires from the IO
  worker thread and feeds dedupe + early-dispatch as before.
- Non-streaming async fallback: when the model client only exposes
  ``complete_async`` (no streaming async), the controller still produces
  a ``_PlannerResult`` from the resolved future.
- Sync test fallback: hand-rolled stubs that lack any ``*_async`` surface
  fall through to the legacy synchronous path; this preserves the test
  doubles in test_controller_dedupe_and_completion.
- Failure propagation: an exception raised by the model client surfaces
  as a failed ``_PlannerResult`` future, not a hung in_flight slot.
"""
from __future__ import annotations

import concurrent.futures
import json
import sys
import threading
import unittest
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from high_agent.agent.controller import AgentRunController, _PlannerRequest
from high_agent.llm.types import NormalizedResponse, ToolCall

# Reuse the controller test fixtures (FakeAgent / FakeRuntime / etc.) so we
# do not duplicate the scaffolding.
from tests.test_controller_dedupe_and_completion import _FakeAgent  # type: ignore


def _make_controller() -> AgentRunController:
    agent = _FakeAgent()
    return AgentRunController(agent=agent, objective="test", messages=[])


class _StreamingAsyncModel:
    """Test double mimicking ModelClient.complete_streaming_async.

    Resolves the future synchronously after firing a single ``on_tool_call``,
    so we can assert both the early-dispatch and the result plumbing without
    actually spinning up the IO loop.
    """

    def __init__(self, response: NormalizedResponse, *, fire_tool: ToolCall | None = None) -> None:
        self._response = response
        self._fire_tool = fire_tool
        self.calls: list[dict[str, Any]] = []

    def complete_streaming_async(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        on_tool_call: Any = None,
        **params: Any,
    ) -> concurrent.futures.Future[NormalizedResponse]:
        self.calls.append({"messages": messages, "tools": tools})
        if on_tool_call is not None and self._fire_tool is not None:
            on_tool_call(self._fire_tool)
        fut: concurrent.futures.Future[NormalizedResponse] = concurrent.futures.Future()
        fut.set_result(self._response)
        return fut


class _NonStreamingAsyncModel:
    def __init__(self, response: NormalizedResponse) -> None:
        self._response = response

    def complete_async(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **params: Any,
    ) -> concurrent.futures.Future[NormalizedResponse]:
        fut: concurrent.futures.Future[NormalizedResponse] = concurrent.futures.Future()
        fut.set_result(self._response)
        return fut


class _AsyncRaisingModel:
    """complete_streaming_async hands back a future that rejects with an error."""

    def complete_streaming_async(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        on_tool_call: Any = None,
        **params: Any,
    ) -> concurrent.futures.Future[NormalizedResponse]:
        fut: concurrent.futures.Future[NormalizedResponse] = concurrent.futures.Future()
        fut.set_exception(RuntimeError("upstream failure"))
        return fut


class _LegacySyncOnlyModel:
    """Test double with neither *_async surface; controller falls back to sync."""

    def __init__(self, response: NormalizedResponse) -> None:
        self._response = response

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **params: Any,
    ) -> NormalizedResponse:
        return self._response


def _make_request() -> _PlannerRequest:
    return _PlannerRequest(
        request_id=1,
        snapshot_seq=0,
        digest_text="runtime idle",
        messages=[{"role": "user", "content": "hi"}],
    )


class StreamingAsyncDispatchTests(unittest.TestCase):
    def test_streaming_async_path_resolves_planner_result(self) -> None:
        controller = _make_controller()
        controller.agent.tools = type("Tools", (), {"definitions": lambda self: []})()
        response = NormalizedResponse(content="ok", tool_calls=None, finish_reason="stop")
        controller.agent.model_client = _StreamingAsyncModel(response)
        request = _make_request()
        controller._register_planner_request(request)

        future = controller._dispatch_planner_request_async(request)

        result = future.result(timeout=2.0)
        self.assertIs(result.response, response)
        self.assertEqual(result.early_dispatched_ids, frozenset())

    def test_streaming_async_fires_on_tool_call_for_early_dispatch(self) -> None:
        controller = _make_controller()
        controller.agent.tools = type("Tools", (), {"definitions": lambda self: []})()
        response = NormalizedResponse(content="", tool_calls=None, finish_reason="tool_calls")
        early_call = ToolCall(
            id="early-1",
            name="write_file",
            arguments=json.dumps({"path": "out.txt", "content": "hi"}),
        )
        model = _StreamingAsyncModel(response, fire_tool=early_call)
        controller.agent.model_client = model
        request = _make_request()
        controller._register_planner_request(request)

        future = controller._dispatch_planner_request_async(request)
        result = future.result(timeout=2.0)

        self.assertEqual(result.early_dispatched_ids, frozenset({"early-1"}))
        self.assertEqual(len(controller.agent.submitted), 1)
        self.assertEqual(controller.agent.submitted[0][0].call.id, "early-1")


class NonStreamingAsyncDispatchTests(unittest.TestCase):
    def test_complete_async_only_still_resolves(self) -> None:
        controller = _make_controller()
        controller.agent.tools = type("Tools", (), {"definitions": lambda self: []})()
        response = NormalizedResponse(content="done", tool_calls=None, finish_reason="stop")
        controller.agent.model_client = _NonStreamingAsyncModel(response)
        request = _make_request()
        controller._register_planner_request(request)

        future = controller._dispatch_planner_request_async(request)
        result = future.result(timeout=2.0)

        self.assertIs(result.response, response)
        # No streaming → no early dispatch.
        self.assertEqual(result.early_dispatched_ids, frozenset())


class AsyncFailureDispatchTests(unittest.TestCase):
    def test_future_rejection_propagates_to_planner_result_future(self) -> None:
        controller = _make_controller()
        controller.agent.tools = type("Tools", (), {"definitions": lambda self: []})()
        controller.agent.model_client = _AsyncRaisingModel()
        request = _make_request()
        controller._register_planner_request(request)

        future = controller._dispatch_planner_request_async(request)
        with self.assertRaises(RuntimeError):
            future.result(timeout=2.0)


class LegacySyncFallbackDispatchTests(unittest.TestCase):
    def test_clients_without_async_surface_fall_back_to_sync(self) -> None:
        controller = _make_controller()
        controller.agent.tools = type("Tools", (), {"definitions": lambda self: []})()
        response = NormalizedResponse(content="legacy", tool_calls=None, finish_reason="stop")
        controller.agent.model_client = _LegacySyncOnlyModel(response)
        request = _make_request()
        controller._register_planner_request(request)

        future = controller._dispatch_planner_request_async(request)
        result = future.result(timeout=2.0)

        self.assertIs(result.response, response)


if __name__ == "__main__":
    unittest.main()
