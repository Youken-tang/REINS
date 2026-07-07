""" Worker handler dispatches model tool_calls as discovered_tasks.

Before v11-D1 the worker handler called ``WorkerAgent.run_one_turn(...).content``
and threw away ``response.tool_calls``. The model could narrate that it had
"created services/auth.py", but no AgentTaskSpec was ever scheduled — empty
``services/`` / ``schemas/`` / ``templates/`` directories under a delegated
scaffold were the symptom in the trace. These tests pin the new contract:

1. When the worker model returns tool_calls, the handler lowers them into
   ``TaskResult.discovered_tasks`` and the runtime executes them.
2. ``delegate_task`` is filtered out of the worker's tool view, so workers
   cannot recursively fan out further worker batches (use mode='sub_agent'
   when iteration is needed).
3. With no ``tool_registry`` wired, tool_calls are surfaced in the summary
   text rather than dropped silently.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from high_agent.agent import create_worker_handler
from high_agent.llm.types import NormalizedResponse, ToolCall
from high_agent.runtime import AgentTaskSpec
from high_agent.runtime.scheduler import CausalRuntime
from high_agent.tools.core import create_core_registry


class _ToolCallingFakeModel:
    """Fake ModelClient that returns a fixed list of tool_calls once."""

    settings = type("S", (), {"model": "test-model"})()

    def __init__(self, tool_calls: list[ToolCall], content: str = "") -> None:
        self._tool_calls = tool_calls
        self._content = content
        self.invocations: list[dict[str, object]] = []

    def complete(self, messages, tools=None, **params):
        self.invocations.append({"messages": list(messages), "tools": tools})
        return NormalizedResponse(self._content, list(self._tool_calls), "tool_calls")


class WorkerDispatchesToolCallsTests(unittest.TestCase):
    def test_worker_lowers_write_file_tool_call_into_discovered_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = _ToolCallingFakeModel(
                tool_calls=[
                    ToolCall(
                        id="call-write-1",
                        name="write_file",
                        arguments='{"path": "services/auth.py", "content": "x = 1\\n"}',
                    ),
                ],
                content="creating services/auth.py",
            )
            registry = create_core_registry()
            runtime = CausalRuntime(
                workspace_root=tmp,
                strict_nogil=True,
                delivery_debounce=0.01,
                executors={"worker": create_worker_handler(model, registry)},  # type: ignore[arg-type]
            )
            runtime.start()
            self.addCleanup(runtime.shutdown)

            task = AgentTaskSpec(
                kind="worker",
                goal="scaffold auth service",
                input={"input": "use FastAPI"},
                task_id="worker-dispatch-1",
            )
            runtime.submit([task])
            self.assertTrue(runtime.wait_all(timeout=5))

            self.assertTrue((Path(tmp) / "services" / "auth.py").exists())
            self.assertEqual(
                (Path(tmp) / "services" / "auth.py").read_text(encoding="utf-8"),
                "x = 1\n",
            )
            results = runtime.collect({"worker-dispatch-1"})
            self.assertEqual(results["worker-dispatch-1"].status, "completed")
            self.assertIn("write_file", results["worker-dispatch-1"].summary)
            self.assertIn("dispatched", results["worker-dispatch-1"].summary)
            self.assertEqual(len(model.invocations), 1)
            self.assertIsNotNone(model.invocations[0]["tools"])

    def test_worker_filters_delegate_task_out_of_its_tool_view(self) -> None:
        captured_tools: list[list[dict[str, object]] | None] = []

        class _CapturingModel:
            settings = type("S", (), {"model": "test-model"})()

            def complete(self, messages, tools=None, **params):
                captured_tools.append(tools)
                return NormalizedResponse("ok", None, "stop")

        with tempfile.TemporaryDirectory() as tmp:
            registry = create_core_registry()
            self.assertIn("delegate_task", registry.names())
            runtime = CausalRuntime(
                workspace_root=tmp,
                strict_nogil=True,
                delivery_debounce=0.01,
                executors={
                    "worker": create_worker_handler(_CapturingModel(), registry)  # type: ignore[arg-type]
                },
            )
            runtime.start()
            self.addCleanup(runtime.shutdown)

            runtime.submit([
                AgentTaskSpec(
                    kind="worker",
                    goal="check filtered tools",
                    task_id="worker-filter-1",
                ),
            ])
            self.assertTrue(runtime.wait_all(timeout=3))

            self.assertEqual(len(captured_tools), 1)
            tool_names = {
                (t.get("function", {}) or {}).get("name")
                for t in (captured_tools[0] or [])
            }
            self.assertNotIn("delegate_task", tool_names)
            self.assertIn("write_file", tool_names)

    def test_worker_without_registry_surfaces_tool_calls_as_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = _ToolCallingFakeModel(
                tool_calls=[ToolCall(id="c1", name="write_file", arguments="{}")],
                content="narrative only",
            )
            runtime = CausalRuntime(
                workspace_root=tmp,
                strict_nogil=True,
                delivery_debounce=0.01,
                executors={"worker": create_worker_handler(model)},  # type: ignore[arg-type]
            )
            runtime.start()
            self.addCleanup(runtime.shutdown)

            runtime.submit([
                AgentTaskSpec(
                    kind="worker",
                    goal="no registry",
                    task_id="worker-no-reg-1",
                ),
            ])
            self.assertTrue(runtime.wait_all(timeout=3))
            result = runtime.collect({"worker-no-reg-1"})["worker-no-reg-1"]
            self.assertEqual(result.status, "completed")
            self.assertIn("write_file", result.summary)
            self.assertIn("no tool_registry", result.summary)


if __name__ == "__main__":
    unittest.main()
