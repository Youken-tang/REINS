"""Tests for SubAgent multi-turn executor."""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from high_agent.agent.sub_agent import SubAgent, create_sub_agent_handler
from high_agent.runtime.scheduler import CausalRuntime
from high_agent.runtime.types import AgentTaskSpec, TaskContext, TaskResult, new_id
from high_agent.runtime.resource_access import ResourceAccess
from high_agent.tools.registry import ToolRegistry
from high_agent.tools.delegate import delegate_task_handler


def _make_runtime(**kwargs) -> CausalRuntime:
    return CausalRuntime(strict_nogil=False, **kwargs)


def _make_mock_client() -> MagicMock:
    """Create a mock model client that explicitly lacks streaming support."""
    client = MagicMock()
    client.complete_streaming = None
    return client


def _make_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        name="echo",
        schema={"description": "Echo input", "parameters": {"type": "object", "properties": {"text": {"type": "string"}}}},
        handler=lambda args: f"echo: {args.get('text', '')}",
        resource_access=lambda args, root: ResourceAccess.empty(),
    )
    registry.register(
        name="write_file",
        schema={"description": "Write file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}}},
        handler=lambda args: f"wrote {args.get('path', '')}",
        resource_access=lambda args, root: ResourceAccess(writes=frozenset({f"file:{args.get('path', '')}"})),
    )
    return registry


@dataclass
class MockToolCall:
    id: str = ""
    name: str = ""
    arguments: str = ""
    provider_data: Any = None

    def args_dict(self) -> dict[str, Any]:
        import json
        return json.loads(self.arguments) if self.arguments else {}


@dataclass
class MockResponse:
    content: str = ""
    tool_calls: list[Any] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: Any = None


class TestSubAgentBasic(unittest.TestCase):
    def test_sub_agent_no_tool_calls_returns_immediately(self):
        runtime = _make_runtime()
        runtime.start()
        tools = _make_tool_registry()
        mock_client = _make_mock_client()
        mock_client.complete.return_value = MockResponse(content="Done with the task")

        sub = SubAgent(
            objective="test goal",
            model_client=mock_client,
            tools=tools,
            runtime=runtime,
            parent_task_id="parent-1",
        )
        result = sub.run()
        runtime.shutdown()

        self.assertEqual(result.status, "completed")
        self.assertIn("Done with the task", result.summary)
        mock_client.complete.assert_called_once()

    def test_sub_agent_executes_tool_calls_and_loops(self):
        runtime = _make_runtime()
        runtime.start()
        tools = _make_tool_registry()
        mock_client = _make_mock_client()

        call1 = MockToolCall(id="c1", name="echo", arguments='{"text": "hello"}')
        call2 = MockToolCall(id="c2", name="echo", arguments='{"text": "world"}')

        mock_client.complete.side_effect = [
            MockResponse(content="", tool_calls=[call1, call2]),
            MockResponse(content="All done"),
        ]

        sub = SubAgent(
            objective="echo test",
            model_client=mock_client,
            tools=tools,
            runtime=runtime,
            parent_task_id="parent-2",
        )
        result = sub.run()
        runtime.shutdown()

        self.assertEqual(result.status, "completed")
        self.assertIn("All done", result.summary)
        self.assertEqual(mock_client.complete.call_count, 2)

    def test_sub_agent_respects_max_iterations(self):
        runtime = _make_runtime()
        runtime.start()
        tools = _make_tool_registry()
        mock_client = _make_mock_client()

        call = MockToolCall(id="c1", name="echo", arguments='{"text": "loop"}')
        mock_client.complete.return_value = MockResponse(content="", tool_calls=[call])

        sub = SubAgent(
            objective="infinite loop",
            model_client=mock_client,
            tools=tools,
            runtime=runtime,
            parent_task_id="parent-3",
            max_iterations=3,
        )
        result = sub.run()
        runtime.shutdown()

        self.assertEqual(result.status, "completed")
        self.assertIn("max iterations", result.summary)
        self.assertEqual(mock_client.complete.call_count, 3)

    def test_sub_agent_timeout(self):
        runtime = _make_runtime()
        runtime.start()
        tools = _make_tool_registry()
        mock_client = _make_mock_client()

        def slow_complete(*args, **kwargs):
            time.sleep(0.3)
            return MockResponse(content="", tool_calls=[
                MockToolCall(id="c1", name="echo", arguments='{"text": "x"}')
            ])

        mock_client.complete.side_effect = slow_complete

        sub = SubAgent(
            objective="timeout test",
            model_client=mock_client,
            tools=tools,
            runtime=runtime,
            parent_task_id="parent-4",
            max_iterations=20,
            timeout_seconds=0.5,
        )
        result = sub.run()
        runtime.shutdown()

        self.assertEqual(result.status, "completed")
        self.assertIn("timeout", result.summary.lower())

    def test_sub_agent_tasks_use_parent_runtime_parallelism(self):
        runtime = _make_runtime(max_workers=4)
        runtime.start()
        tools = _make_tool_registry()
        mock_client = _make_mock_client()

        calls = [
            MockToolCall(id=f"c{i}", name="echo", arguments=f'{{"text": "item{i}"}}')
            for i in range(4)
        ]
        mock_client.complete.side_effect = [
            MockResponse(content="", tool_calls=calls),
            MockResponse(content="parallel done"),
        ]

        sub = SubAgent(
            objective="parallel test",
            model_client=mock_client,
            tools=tools,
            runtime=runtime,
            parent_task_id="parent-5",
        )
        result = sub.run()
        runtime.shutdown()

        self.assertEqual(result.status, "completed")
        self.assertEqual(len(sub._submitted_ids), 4)


class TestSubAgentHandler(unittest.TestCase):
    def test_create_sub_agent_handler_excludes_delegate(self):
        tools = _make_tool_registry()
        tools.register(
            name="delegate_task",
            schema={"description": "delegate", "parameters": {"type": "object", "properties": {}}},
            handler=lambda args: "delegated",
            resource_access=lambda args, root: ResourceAccess.empty(),
        )
        mock_client = _make_mock_client()
        mock_client.complete.return_value = MockResponse(content="handler done")

        handler = create_sub_agent_handler(mock_client, tools)

        runtime = _make_runtime()
        runtime.start()
        task = AgentTaskSpec(
            kind="sub_agent",
            goal="handler test",
            task_id="sub-test-1",
            metadata={"depth": 2},
        )
        ctx = TaskContext(task=task, runtime=runtime, ledger_digest="idle")
        result = handler(ctx)
        runtime.shutdown()

        self.assertEqual(result.status, "completed")

    def test_recursive_delegation_allowed_below_max_depth(self):
        tools = _make_tool_registry()
        tools.register(
            name="delegate_task",
            schema={"description": "delegate", "parameters": {"type": "object", "properties": {}}},
            handler=lambda args: "delegated",
            resource_access=lambda args, root: ResourceAccess.empty(),
        )
        mock_client = _make_mock_client()
        mock_client.complete.return_value = MockResponse(content="depth 0 done")

        handler = create_sub_agent_handler(mock_client, tools, max_depth=2)

        runtime = _make_runtime()
        runtime.start()
        task = AgentTaskSpec(
            kind="sub_agent",
            goal="depth 0 test",
            task_id="sub-depth-0",
            metadata={"depth": 0},
        )
        ctx = TaskContext(task=task, runtime=runtime, ledger_digest="idle")
        result = handler(ctx)
        runtime.shutdown()

        self.assertEqual(result.status, "completed")

    def test_delegate_task_propagates_depth(self):
        result = delegate_task_handler({
            "goal": "nested",
            "mode": "sub_agent",
            "tasks": [{"goal": "child"}],
            "_depth": 1,
        })
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.discovered_tasks[0].metadata.get("depth"), 1)


class TestDelegateTaskMode(unittest.TestCase):
    def test_delegate_mode_worker_creates_worker_kind(self):
        result = delegate_task_handler({
            "goal": "test batch",
            "mode": "worker",
            "tasks": [{"goal": "task A"}],
        })
        self.assertEqual(result.status, "completed")
        self.assertEqual(len(result.discovered_tasks), 1)
        self.assertEqual(result.discovered_tasks[0].kind, "worker")

    def test_delegate_mode_sub_agent_creates_sub_agent_kind(self):
        result = delegate_task_handler({
            "goal": "complex batch",
            "mode": "sub_agent",
            "tasks": [{"goal": "complex task A"}, {"goal": "complex task B"}],
        })
        self.assertEqual(result.status, "completed")
        self.assertEqual(len(result.discovered_tasks), 2)
        for task in result.discovered_tasks:
            self.assertEqual(task.kind, "sub_agent")
            self.assertTrue(task.task_id.startswith("subagent-"))

    def test_delegate_default_mode_is_worker(self):
        result = delegate_task_handler({
            "goal": "default",
            "tasks": [{"goal": "t1"}],
        })
        self.assertEqual(result.discovered_tasks[0].kind, "worker")


class TestWaitTasks(unittest.TestCase):
    def test_wait_tasks_returns_completed_results(self):
        runtime = _make_runtime()
        runtime.start()

        def fast_handler(ctx: TaskContext) -> TaskResult:
            return TaskResult.completed("fast done")

        tasks = [
            AgentTaskSpec(kind="tool", goal="t1", task_id="wt-1", handler=fast_handler),
            AgentTaskSpec(kind="tool", goal="t2", task_id="wt-2", handler=fast_handler),
        ]
        runtime.submit(tasks)
        results = runtime.wait_tasks(["wt-1", "wt-2"], timeout=5.0)
        runtime.shutdown()

        self.assertEqual(len(results), 2)
        self.assertEqual(results["wt-1"].status, "completed")
        self.assertEqual(results["wt-2"].status, "completed")

    def test_wait_tasks_timeout_returns_partial(self):
        runtime = _make_runtime()
        runtime.start()

        def slow_handler(ctx: TaskContext) -> TaskResult:
            time.sleep(5.0)
            return TaskResult.completed("slow")

        tasks = [
            AgentTaskSpec(kind="tool", goal="slow", task_id="wt-slow", handler=slow_handler),
        ]
        runtime.submit(tasks)
        results = runtime.wait_tasks(["wt-slow"], timeout=0.3)
        runtime.shutdown()

        self.assertNotIn("wt-slow", results)


class TestSubAgentStreaming(unittest.TestCase):
    def test_streaming_early_dispatch(self):
        """When complete_streaming is available, tool calls dispatch before response completes."""
        runtime = _make_runtime()
        runtime.start()
        tools = _make_tool_registry()
        mock_client = MagicMock()

        call1 = MockToolCall(id="s1", name="echo", arguments='{"text": "early"}')
        call2 = MockToolCall(id="s2", name="echo", arguments='{"text": "early2"}')

        def fake_streaming(messages, tools=None, on_tool_call=None, **kwargs):
            if on_tool_call:
                on_tool_call(call1)
                on_tool_call(call2)
            return MockResponse(content="", tool_calls=[call1, call2])

        mock_client.complete_streaming = fake_streaming

        sub = SubAgent(
            objective="streaming test",
            model_client=mock_client,
            tools=tools,
            runtime=runtime,
            parent_task_id="parent-stream",
            max_iterations=2,
        )

        mock_client.complete = MagicMock(return_value=MockResponse(content="streaming done"))
        # First iteration uses streaming, second returns final answer via sync fallback
        # But since we set complete_streaming, it will always use streaming
        def second_call(messages, tools=None, on_tool_call=None, **kwargs):
            return MockResponse(content="streaming done")

        mock_client.complete_streaming = MagicMock(side_effect=[
            fake_streaming(None, on_tool_call=lambda c: None),  # dummy
        ])
        # Simpler: just use the fake_streaming for first call, then final answer
        call_count = [0]

        def streaming_side_effect(messages, tools=None, on_tool_call=None, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                if on_tool_call:
                    on_tool_call(call1)
                    on_tool_call(call2)
                return MockResponse(content="", tool_calls=[call1, call2])
            return MockResponse(content="streaming done")

        mock_client.complete_streaming = streaming_side_effect

        sub = SubAgent(
            objective="streaming test",
            model_client=mock_client,
            tools=tools,
            runtime=runtime,
            parent_task_id="parent-stream",
            max_iterations=5,
        )
        result = sub.run()
        runtime.shutdown()

        self.assertEqual(result.status, "completed")
        self.assertIn("streaming done", result.summary)
        self.assertTrue(len(sub._submitted_ids) >= 2)

    def test_fallback_to_sync_when_no_streaming(self):
        """When complete_streaming is None, falls back to sync complete."""
        runtime = _make_runtime()
        runtime.start()
        tools = _make_tool_registry()
        mock_client = _make_mock_client()
        mock_client.complete.return_value = MockResponse(content="sync result")

        sub = SubAgent(
            objective="sync test",
            model_client=mock_client,
            tools=tools,
            runtime=runtime,
            parent_task_id="parent-sync",
        )
        result = sub.run()
        runtime.shutdown()

        self.assertEqual(result.status, "completed")
        self.assertIn("sync result", result.summary)
        mock_client.complete.assert_called_once()

    def test_streaming_dispatch_exception_emits_trace(self):
        """v8-P6: when _on_tool_call fails, the failure is reported via trace."""
        import tempfile
        import json as _json

        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.jsonl"
            runtime = CausalRuntime(strict_nogil=False, trace_path=trace_path)
            runtime.start()
            tools = _make_tool_registry()

            bad_call = MockToolCall(id="bad", name="echo", arguments='{"text": "x"}')

            def streaming_side_effect(messages, tools=None, on_tool_call=None, **kwargs):
                if on_tool_call:
                    on_tool_call(bad_call)
                return MockResponse(content="done")

            mock_client = MagicMock()
            mock_client.complete_streaming = streaming_side_effect

            sub = SubAgent(
                objective="trace test",
                model_client=mock_client,
                tools=tools,
                runtime=runtime,
                parent_task_id="parent-trace",
                max_iterations=2,
            )

            # Force _lower_tool_calls to raise inside the streaming callback.
            with patch.object(SubAgent, "_lower_tool_calls", side_effect=RuntimeError("synthetic-lower-fail")):
                result = sub.run()

            runtime.shutdown()
            self.assertEqual(result.status, "completed")

            events = [
                _json.loads(line)
                for line in trace_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            dispatch_failures = [e for e in events if e.get("event") == "subagent.dispatch_failed"]
            self.assertEqual(len(dispatch_failures), 1)
            self.assertEqual(dispatch_failures[0]["parent_task_id"], "parent-trace")
            self.assertEqual(dispatch_failures[0]["error_type"], "RuntimeError")
            self.assertIn("synthetic-lower-fail", dispatch_failures[0]["error"])
            self.assertEqual(dispatch_failures[0]["tool"], "echo")


if __name__ == "__main__":
    unittest.main()
