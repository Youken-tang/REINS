from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from high_agent.agent import (
    Agent,
    AgentContext,
    AgentTurnResult,
    MainAgent,
    WorkerAgent,
    PlannerAgent,
    create_worker_handler,
)
from high_agent.runtime import AgentTaskSpec, TaskResult
from high_agent.runtime.scheduler import CausalRuntime
from high_agent.tools.delegate import delegate_task_handler, register_delegate_task


class AgentProtocolTests(unittest.TestCase):
    """Verify structural protocol compliance."""

    def test_worker_agent_satisfies_protocol(self) -> None:
        worker = WorkerAgent(objective="test task", input_data={})
        self.assertIsInstance(worker, Agent)
        self.assertEqual(worker.objective, "test task")

    def test_worker_agent_has_no_runtime_attribute(self) -> None:
        worker = WorkerAgent(objective="x", input_data={})
        self.assertFalse(hasattr(worker, "runtime"))

    def test_worker_without_model_returns_summary(self) -> None:
        worker = WorkerAgent(objective="compute sum", input_data={"input": "1+2"})
        result = worker.run_one_turn("ledger: idle")
        self.assertTrue(result.is_final)
        self.assertIn("compute sum", result.content)

    def test_worker_context_is_scoped(self) -> None:
        worker = WorkerAgent(
            objective="write file",
            input_data={
                "component_summaries": ["dir /src created"],
                "ledger_digest": "completed: task-a",
            },
        )
        self.assertEqual(worker.context.objective, "write file")
        self.assertIn("dir /src created", worker.context.completed_summaries)
        self.assertIn("completed: task-a", worker.context.runtime_digests)


class WorkerHandlerTests(unittest.TestCase):
    """Worker handler integration with CausalRuntime."""

    def test_worker_handler_executes_kind_worker_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = CausalRuntime(
                workspace_root=tmp,
                strict_nogil=True,
                delivery_debounce=0.01,
                executors={"worker": create_worker_handler(None)},
            )
            runtime.start()
            self.addCleanup(runtime.shutdown)

            task = AgentTaskSpec(
                kind="worker",
                goal="say hello",
                input={"input": "world"},
                task_id="worker-test-1",
            )
            runtime.submit([task])
            self.assertTrue(runtime.wait_all(timeout=2))

            results = runtime.collect({"worker-test-1"})
            self.assertEqual(results["worker-test-1"].status, "completed")
            self.assertIn("say hello", results["worker-test-1"].summary)

    def test_worker_handler_passes_model_client_through(self) -> None:
        # previously create_worker_handler discarded the model_client
        # parameter, hardwiring WorkerAgent(model_client=None). Worker tasks
        # silently ran the no-model fallback that just echoed the goal.
        from high_agent.llm.types import NormalizedResponse

        captured: dict[str, object] = {}

        class _FakeModel:
            settings = type("S", (), {"model": "test-model"})()

            def complete(self, messages, tools=None, **params):
                captured["messages"] = messages
                return NormalizedResponse("worker model output", None, "stop")

        with tempfile.TemporaryDirectory() as tmp:
            runtime = CausalRuntime(
                workspace_root=tmp,
                strict_nogil=True,
                delivery_debounce=0.01,
                executors={"worker": create_worker_handler(_FakeModel())},  # type: ignore[arg-type]
            )
            runtime.start()
            self.addCleanup(runtime.shutdown)

            task = AgentTaskSpec(
                kind="worker",
                goal="produce something with the model",
                input={"input": "hello"},
                task_id="worker-test-2",
            )
            runtime.submit([task])
            self.assertTrue(runtime.wait_all(timeout=2))

            results = runtime.collect({"worker-test-2"})
            self.assertEqual(results["worker-test-2"].status, "completed")
            self.assertEqual(results["worker-test-2"].summary, "worker model output")
            # And the model was actually invoked.
            self.assertIn("messages", captured)

    def test_multiple_workers_run_in_parallel(self) -> None:
        import time

        with tempfile.TemporaryDirectory() as tmp:
            runtime = CausalRuntime(
                workspace_root=tmp,
                strict_nogil=True,
                delivery_debounce=0.01,
                max_workers=4,
                executors={"worker": create_worker_handler(None)},
            )
            runtime.start()
            self.addCleanup(runtime.shutdown)

            tasks = [
                AgentTaskSpec(kind="worker", goal=f"task-{i}", input={}, task_id=f"w-{i}")
                for i in range(4)
            ]
            start = time.monotonic()
            runtime.submit(tasks)
            self.assertTrue(runtime.wait_all(timeout=2))
            elapsed = time.monotonic() - start
            self.assertLess(elapsed, 0.5)

            results = runtime.collect({f"w-{i}" for i in range(4)})
            self.assertTrue(all(r.status == "completed" for r in results.values()))


class DelegateTaskTests(unittest.TestCase):
    """delegate_task tool lowering."""

    def test_delegate_creates_worker_discovered_tasks(self) -> None:
        result = delegate_task_handler({
            "goal": "build feature",
            "tasks": [
                {"goal": "write module A", "input": "specs..."},
                {"goal": "write module B", "input": "specs..."},
            ],
        })
        self.assertEqual(result.status, "completed")
        self.assertEqual(len(result.discovered_tasks), 2)
        self.assertTrue(all(t.kind == "worker" for t in result.discovered_tasks))
        self.assertEqual(result.discovered_tasks[0].goal, "write module A")
        self.assertEqual(result.discovered_tasks[1].goal, "write module B")

    def test_delegate_empty_tasks_fails(self) -> None:
        result = delegate_task_handler({"goal": "nothing", "tasks": []})
        self.assertEqual(result.status, "failed")

    def test_delegate_with_resource_declarations(self) -> None:
        result = delegate_task_handler({
            "goal": "parallel writes",
            "_workspace_root": "/workspace",
            "tasks": [
                {"goal": "write a", "writes": ["src/a.py"]},
                {"goal": "write b", "reads": ["src/c.py"], "writes": ["src/b.py"]},
            ],
        })
        self.assertEqual(result.status, "completed")
        task_a = result.discovered_tasks[0]
        task_b = result.discovered_tasks[1]
        self.assertTrue(task_a.resource_access.writes)
        self.assertTrue(task_b.resource_access.reads)
        self.assertTrue(task_b.resource_access.writes)

    def test_delegate_integration_with_runtime(self) -> None:
        """End-to-end: delegate_task → runtime → worker handler → delivery."""
        with tempfile.TemporaryDirectory() as tmp:
            runtime = CausalRuntime(
                workspace_root=tmp,
                strict_nogil=True,
                delivery_debounce=0.01,
                executors={"worker": create_worker_handler(None)},
            )
            runtime.start()
            self.addCleanup(runtime.shutdown)

            from high_agent.tools.core import create_core_registry
            registry = create_core_registry()
            task = registry.task_from_call(
                "delegate_task",
                {"goal": "test batch", "tasks": [{"goal": "sub1"}, {"goal": "sub2"}]},
                workspace_root=tmp,
                task_id="delegate-1",
            )
            runtime.submit([task])
            self.assertTrue(runtime.wait_all(timeout=3))

            results = runtime.collect(set(runtime._tasks))
            worker_results = {tid: r for tid, r in results.items() if "worker" in tid}
            self.assertEqual(len(worker_results), 2)
            self.assertTrue(all(r.status == "completed" for r in worker_results.values()))


class AgentTurnResultTests(unittest.TestCase):
    """AgentTurnResult factory methods."""

    def test_final_answer(self) -> None:
        r = AgentTurnResult.final_answer("done")
        self.assertTrue(r.is_final)
        self.assertEqual(r.content, "done")

    def test_with_tool_calls(self) -> None:
        r = AgentTurnResult.with_tool_calls([{"name": "read_file"}], content="thinking")
        self.assertFalse(r.is_final)
        self.assertEqual(len(r.tool_calls), 1)

    def test_from_task_result(self) -> None:
        tr = TaskResult.completed("all good")
        r = AgentTurnResult.from_task_result(tr)
        self.assertTrue(r.is_final)
        self.assertEqual(r.content, "all good")


class AgentContextTests(unittest.TestCase):
    """AgentContext construction and rendering."""

    def test_for_worker(self) -> None:
        ctx = AgentContext.for_worker("write file", input_summaries=["dir created"], ledger_digest="idle")
        self.assertEqual(ctx.objective, "write file")
        self.assertIn("dir created", ctx.completed_summaries)
        rendered = ctx.render()
        self.assertIn("write file", rendered)

    def test_from_total(self) -> None:
        from high_agent.agent.context import TotalContext
        total = TotalContext("build project")
        total.add_delivery("step 1 done", "ledger: 1 completed")
        ctx = AgentContext.from_total(total)
        self.assertEqual(ctx.objective, "build project")
        self.assertIn("step 1 done", ctx.completed_summaries)


if __name__ == "__main__":
    unittest.main()
