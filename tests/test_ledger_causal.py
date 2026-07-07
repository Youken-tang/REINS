from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from high_agent.runtime import AgentTaskSpec, DependencyPredicate, TaskResult
from high_agent.runtime.scheduler import CausalRuntime


class LedgerCausalChainTests(unittest.TestCase):
    def make_runtime(self, tmp: str, **kwargs) -> CausalRuntime:
        runtime = CausalRuntime(
            workspace_root=tmp,
            strict_nogil=True,
            delivery_debounce=kwargs.pop("delivery_debounce", 0.01),
            **kwargs,
        )
        runtime.start()
        self.addCleanup(runtime.shutdown)
        return runtime

    def _trivial_handler(self, label: str):
        def _h(ctx):
            return TaskResult.completed(label)
        return _h

    def test_completed_task_records_triggered_dependents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self.make_runtime(tmp, max_workers=4)
            task_a = AgentTaskSpec(kind="tool", goal="A", handler=self._trivial_handler("a"), task_id="task-a")
            task_b = AgentTaskSpec(
                kind="tool", goal="B",
                handler=self._trivial_handler("b"),
                task_id="task-b",
                dependencies=[DependencyPredicate.task_completed("task-a")],
            )
            task_c = AgentTaskSpec(
                kind="tool", goal="C",
                handler=self._trivial_handler("c"),
                task_id="task-c",
                dependencies=[DependencyPredicate.task_completed("task-a")],
            )
            runtime.submit([task_a, task_b, task_c])
            self.assertTrue(runtime.wait_all(timeout=2))

            digest = runtime.ledger.digest()
            self.assertIn("task-a", digest.causal_chains)
            self.assertEqual(set(digest.causal_chains["task-a"]), {"task-b", "task-c"})

            records = runtime.ledger.records_snapshot()
            self.assertEqual(records["task-b"].triggered_by, "task-a")
            self.assertEqual(records["task-c"].triggered_by, "task-a")
            self.assertEqual(set(records["task-a"].triggered_tasks), {"task-b", "task-c"})

            # text digest 也应当包含因果链摘要
            self.assertIn("chains:", digest.text)

    def test_recent_completions_are_ordered_by_finish_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self.make_runtime(tmp, max_workers=4)

            def slow_handler(label: str, delay: float):
                def _h(ctx):
                    time.sleep(delay)
                    return TaskResult.completed(label)
                return _h

            runtime.submit([
                AgentTaskSpec(kind="tool", goal="A", handler=slow_handler("a", 0.05), task_id="task-a"),
                AgentTaskSpec(
                    kind="tool", goal="B",
                    handler=slow_handler("b", 0.02),
                    task_id="task-b",
                    dependencies=[DependencyPredicate.task_completed("task-a")],
                ),
            ])
            self.assertTrue(runtime.wait_all(timeout=2))

            digest = runtime.ledger.digest()
            self.assertEqual(len(digest.recent_completions), 2)
            most_recent_task_id = digest.recent_completions[0][0]
            self.assertEqual(most_recent_task_id, "task-b")

    def test_discovered_tasks_appear_in_discovery_chains(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self.make_runtime(tmp, max_workers=4)

            child = AgentTaskSpec(
                kind="tool",
                goal="dynamic-child",
                handler=self._trivial_handler("child"),
                task_id="task-child",
            )

            def parent_handler(ctx):
                return TaskResult(status="completed", summary="parent", discovered_tasks=[child])

            parent = AgentTaskSpec(kind="tool", goal="parent", handler=parent_handler, task_id="task-parent")
            runtime.submit([parent])
            self.assertTrue(runtime.wait_all(timeout=2))

            digest = runtime.ledger.digest()
            self.assertIn("task-parent", digest.discovery_chains)
            self.assertIn("task-child", digest.discovery_chains["task-parent"])

            records = runtime.ledger.records_snapshot()
            self.assertEqual(records["task-child"].discovered_by, "task-parent")
            self.assertIn("task-child", records["task-parent"].discovered_tasks)


if __name__ == "__main__":
    unittest.main()
