""" Tests for agent_loop / agent_loop_step kinds with suspend-on-LLM.

The agent_loop pair replaces sub_agent's residual worker-thread occupation
during LLM HTTP RTT. Each step:

1. Aggregates prev results (Phase A, in worker thread).
2. Calls model_client.complete_async() and immediately returns
   TaskResult.suspended(awaiting=[future_done(token)]) — the worker thread
   is freed.
3. The scheduler resumes the task on the next ready slot when the future
   completes; resume_handler reads the response, lowers tool_calls into
   children, and emits the next step.

These tests assert:

- Entry handler returns immediately with the first step as a discovered
  task (mirrors sub_agent entry semantics).
- A single step suspends on the LLM future and resumes to a final answer
  when the model returns no tool_calls.
- Multi-step chains: suspend-on-model → resume → emit children → next
  step suspends again on the next LLM call.
- timeout / max_iterations short-circuit before complete_async
  is invoked and surface as TaskResult.failed with the right error_type.
- Children failures are folded back into the message history as
  ``[failed] <summary>`` (path2-C1 task_completed terminal semantics).
- Context isolation: parent and child agent_loops do not share message
  state through any shared instance.
"""
from __future__ import annotations

import concurrent.futures
import json
import sys
import time
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from high_agent.agent.loop import (
    AgentLoopState,
    create_agent_loop_handler,
    create_agent_loop_step_handler,
)
from high_agent.runtime.resource_access import ResourceAccess
from high_agent.runtime.scheduler import CausalRuntime
from high_agent.runtime.types import (
    AgentTaskSpec,
    TaskContext,
    TaskResult,
)
from high_agent.tools.registry import ToolRegistry


@dataclass
class _MockToolCall:
    id: str = ""
    name: str = ""
    arguments: str = ""
    provider_data: Any = None

    def args_dict(self) -> dict[str, Any]:
        return json.loads(self.arguments) if self.arguments else {}


@dataclass
class _MockResponse:
    content: str = ""
    tool_calls: list[Any] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: Any = None


def _make_runtime(**kwargs) -> CausalRuntime:
    return CausalRuntime(strict_nogil=False, **kwargs)


def _make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        name="echo",
        schema={
            "description": "Echo input",
            "parameters": {"type": "object", "properties": {"text": {"type": "string"}}},
        },
        handler=lambda args: f"echo: {args.get('text', '')}",
        resource_access=lambda args, root: ResourceAccess.empty(),
    )
    return registry


def _make_immediate_async(response: _MockResponse):
    """Return a fake complete_async that resolves synchronously."""

    def _complete_async(messages, tools=None, **params):
        fut: concurrent.futures.Future = concurrent.futures.Future()
        fut.set_result(response)
        return fut

    return _complete_async


def _register_executors(
    runtime: CausalRuntime,
    model_client: Any,
    tools: ToolRegistry,
    *,
    max_iterations: int = 4,
    timeout: float = 10.0,
    max_depth: int = 2,
) -> None:
    runtime.executors["agent_loop"] = create_agent_loop_handler(
        model_client, tools,
        max_iterations=max_iterations,
        timeout=timeout,
        max_depth=max_depth,
    )
    runtime.executors["agent_loop_step"] = create_agent_loop_step_handler(
        model_client, tools,
    )


class TestEntryReleasesWorkerThread(unittest.TestCase):
    def test_entry_handler_returns_immediately(self) -> None:
        runtime = _make_runtime(max_workers=2)
        tools = _make_registry()
        mock_client = MagicMock()
        mock_client.complete_async = _make_immediate_async(_MockResponse(content="all done"))
        _register_executors(runtime, mock_client, tools)

        runtime.start()
        entry = AgentTaskSpec(kind="agent_loop", goal="entry test", task_id="loop-entry-1")
        runtime.submit([entry])
        runtime.wait_all(timeout=5.0)
        runtime.shutdown()

        result = runtime.collect(["loop-entry-1"]).get("loop-entry-1")
        self.assertEqual(result.status, "completed")
        self.assertIn("delegated", result.summary)
        self.assertFalse(result.deliverable)


class TestSingleStepSuspendResume(unittest.TestCase):
    def test_step_suspends_on_future_then_resumes_to_completion(self) -> None:
        runtime = _make_runtime(max_workers=2)
        tools = _make_registry()
        mock_client = MagicMock()
        mock_client.complete_async = _make_immediate_async(_MockResponse(content="final"))
        _register_executors(runtime, mock_client, tools)

        runtime.start()
        entry = AgentTaskSpec(kind="agent_loop", goal="single step", task_id="loop-single-1")
        runtime.submit([entry])
        runtime.wait_all(timeout=5.0)
        runtime.shutdown()

        # Step task should have completed with a deliverable final summary.
        records = runtime.ledger.records_snapshot()
        loop_step_records = [r for r in records.values() if r.kind == "agent_loop_step"]
        self.assertEqual(len(loop_step_records), 1)
        step_id = loop_step_records[0].task_id
        step_result = runtime.collect([step_id]).get(step_id)
        self.assertEqual(step_result.status, "completed")
        self.assertEqual(step_result.summary, "final")
        self.assertTrue(step_result.deliverable)


class TestMultiStepWithChildren(unittest.TestCase):
    def test_step_lowers_tool_calls_then_chains_next_step(self) -> None:
        runtime = _make_runtime(max_workers=4)
        tools = _make_registry()

        responses = [
            _MockResponse(
                content="",
                tool_calls=[_MockToolCall(id="c1", name="echo", arguments='{"text": "round-1"}')],
            ),
            _MockResponse(content="multi-step done"),
        ]
        call_count = [0]

        def fake_async(messages, tools=None, **params):
            i = call_count[0]
            call_count[0] += 1
            fut: concurrent.futures.Future = concurrent.futures.Future()
            fut.set_result(responses[min(i, len(responses) - 1)])
            return fut

        mock_client = MagicMock()
        mock_client.complete_async = fake_async
        _register_executors(runtime, mock_client, tools, max_iterations=4)

        runtime.start()
        entry = AgentTaskSpec(kind="agent_loop", goal="multi", task_id="loop-multi-1")
        runtime.submit([entry])
        runtime.wait_all(timeout=10.0)
        runtime.shutdown()

        # Two model calls: first emits a child, second terminates.
        self.assertEqual(call_count[0], 2)
        # Some agent_loop_step task must be deliverable=True with the final answer.
        records = runtime.ledger.records_snapshot()
        step_results = [
            (r.task_id, runtime.collect([r.task_id]).get(r.task_id))
            for r in records.values() if r.kind == "agent_loop_step"
        ]
        deliverable = [tr for _id, tr in step_results
                       if isinstance(tr, TaskResult) and tr.deliverable and tr.status == "completed"]
        self.assertEqual(len(deliverable), 1)
        self.assertEqual(deliverable[0].summary, "multi-step done")

        # And the lowered echo child ran.
        echo_records = [r for r in records.values() if r.kind == "tool" and "echo" in r.goal]
        self.assertEqual(len(echo_records), 1)
        self.assertEqual(echo_records[0].state, "completed")


class TestAuditH6FailureSemantics(unittest.TestCase):
    def test_step_handler_reports_timeout_as_failed_without_calling_model(self) -> None:
        runtime = _make_runtime(max_workers=2)
        tools = _make_registry()
        mock_client = MagicMock()
        mock_client.complete_async = MagicMock()
        step_handler = create_agent_loop_step_handler(mock_client, tools)

        runtime.start()
        try:
            state = AgentLoopState(
                objective="timed-out",
                parent_task_id="loop-h6-1",
                messages=[{"role": "system", "content": "x"}],
                deadline_monotonic=time.monotonic() - 1.0,
                timeout_seconds=0.5,
                max_iterations=4,
                iteration=1,
            )
            task = AgentTaskSpec(
                kind="agent_loop_step",
                goal="step 1: timed-out",
                task_id="loop-step-h6-1",
                input={
                    "loop_state": state.to_input(),
                    "exclude_tools": [],
                    "max_depth": 2,
                    "last_round_task_ids": [],
                },
            )
            ctx = TaskContext(task=task, runtime=runtime, ledger_digest="idle")
            result = step_handler(ctx)
        finally:
            runtime.shutdown()

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.failure_event.error_type, "agent_loop_timeout")
        mock_client.complete_async.assert_not_called()

    def test_step_handler_reports_max_iterations_as_failed_without_calling_model(self) -> None:
        runtime = _make_runtime(max_workers=2)
        tools = _make_registry()
        mock_client = MagicMock()
        mock_client.complete_async = MagicMock()
        step_handler = create_agent_loop_step_handler(mock_client, tools)

        runtime.start()
        try:
            state = AgentLoopState(
                objective="loopy",
                parent_task_id="loop-h6-2",
                messages=[{"role": "system", "content": "x"}],
                deadline_monotonic=time.monotonic() + 60.0,
                timeout_seconds=60.0,
                max_iterations=2,
                iteration=2,
            )
            task = AgentTaskSpec(
                kind="agent_loop_step",
                goal="step 2: loopy",
                task_id="loop-step-h6-2",
                input={
                    "loop_state": state.to_input(),
                    "exclude_tools": [],
                    "max_depth": 2,
                    "last_round_task_ids": [],
                },
            )
            ctx = TaskContext(task=task, runtime=runtime, ledger_digest="idle")
            result = step_handler(ctx)
        finally:
            runtime.shutdown()

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.failure_event.error_type, "agent_loop_max_iterations")
        mock_client.complete_async.assert_not_called()


class TestFailedChildSurfacesInResume(unittest.TestCase):
    def test_failed_child_summary_appended_to_messages_for_next_step(self) -> None:
        # The parent step submits an "echo" child whose handler we override
        # to fail. The next step's resume_handler must see the failure summary
        # rolled into messages as `[failed] ...`. We surface this by capturing
        # the api_messages passed into the *second* complete_async call.
        runtime = _make_runtime(max_workers=4)
        tools = _make_registry()

        seen_messages: list[list[dict[str, Any]]] = []
        responses = [
            _MockResponse(
                content="",
                tool_calls=[_MockToolCall(id="c1", name="echo", arguments='{"text": "first"}')],
            ),
            _MockResponse(content="recovered"),
        ]

        # Register a failing override for echo so the lowered child fails.
        registry = _make_registry()

        def _fail_echo(args: dict[str, Any]) -> Any:
            raise RuntimeError("echo failed on purpose")

        registry.register(
            name="echo",
            schema={
                "description": "Echo input",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                },
            },
            handler=_fail_echo,
            resource_access=lambda args, root: ResourceAccess.empty(),
            override=True,
        )

        def fake_async(messages, tools=None, **params):
            seen_messages.append(list(messages))
            i = len(seen_messages) - 1
            fut: concurrent.futures.Future = concurrent.futures.Future()
            fut.set_result(responses[min(i, len(responses) - 1)])
            return fut

        mock_client = MagicMock()
        mock_client.complete_async = fake_async
        _register_executors(runtime, mock_client, registry, max_iterations=4)

        runtime.start()
        entry = AgentTaskSpec(kind="agent_loop", goal="failure test", task_id="loop-fail-1")
        runtime.submit([entry])
        runtime.wait_all(timeout=10.0)
        runtime.shutdown()

        self.assertGreaterEqual(len(seen_messages), 2)
        # The second LLM call must include a tool message describing the
        # failed child summary with the `[failed]` prefix from
        # _append_results_to_messages.
        second_round = seen_messages[1]
        tool_messages = [m for m in second_round if m.get("role") == "tool"]
        self.assertTrue(tool_messages, "second round must include the failed child's tool message")
        # The combined content of all tool messages should mention failed.
        combined = " ".join(m.get("content", "") for m in tool_messages)
        self.assertIn("[failed]", combined)


class TestContextIsolation(unittest.TestCase):
    def test_two_concurrent_loops_have_independent_messages(self) -> None:
        # Two agent_loops with different objectives must each get their own
        # initial user prompt; complete_async should not see the other loop's
        # objective bleeding into its messages.
        runtime = _make_runtime(max_workers=4)
        tools = _make_registry()
        seen_user_prompts: list[str] = []
        seen_lock = __import__("threading").Lock()

        def fake_async(messages, tools=None, **params):
            with seen_lock:
                user = next((m for m in messages if m.get("role") == "user"), None)
                if user is not None:
                    seen_user_prompts.append(user.get("content", ""))
            fut: concurrent.futures.Future = concurrent.futures.Future()
            fut.set_result(_MockResponse(content="done"))
            return fut

        mock_client = MagicMock()
        mock_client.complete_async = fake_async
        _register_executors(runtime, mock_client, tools)

        runtime.start()
        entries = [
            AgentTaskSpec(kind="agent_loop", goal="goal-A", task_id="loop-iso-A"),
            AgentTaskSpec(kind="agent_loop", goal="goal-B", task_id="loop-iso-B"),
        ]
        runtime.submit(entries)
        runtime.wait_all(timeout=10.0)
        runtime.shutdown()

        # Each loop got its own goal in its initial user prompt.
        self.assertEqual(sorted(seen_user_prompts), sorted(["Goal: goal-A", "Goal: goal-B"]))


if __name__ == "__main__":
    unittest.main()
