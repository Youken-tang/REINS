from __future__ import annotations

import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from high_agent.runtime import AgentTaskSpec, DependencyPredicate, TaskResult
from high_agent.runtime.scheduler import CausalRuntime


class CriticalPathRefillTests(unittest.TestCase):
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

    def test_high_fanout_completion_fires_critical_path_callback(self) -> None:
        signals: list[tuple[str, int]] = []
        signal_event = threading.Event()

        def on_signal(task_id: str, fanout: int, digest):
            signals.append((task_id, fanout))
            signal_event.set()

        with tempfile.TemporaryDirectory() as tmp:
            runtime = self.make_runtime(
                tmp,
                max_workers=4,
                on_critical_path_progress=on_signal,
                critical_path_fanout=2,
            )
            parent = AgentTaskSpec(
                kind="tool",
                goal="parent",
                handler=self._trivial_handler("parent"),
                task_id="task-parent",
            )
            children = [
                AgentTaskSpec(
                    kind="tool",
                    goal=f"child-{idx}",
                    handler=self._trivial_handler(f"child-{idx}"),
                    task_id=f"task-child-{idx}",
                    dependencies=[DependencyPredicate.task_completed("task-parent")],
                )
                for idx in range(3)
            ]
            runtime.submit([parent, *children])
            self.assertTrue(runtime.wait_all(timeout=2))
            self.assertTrue(signal_event.wait(timeout=1.0))

            critical_signals_for_parent = [s for s in signals if s[0] == "task-parent"]
            self.assertEqual(len(critical_signals_for_parent), 1)
            self.assertEqual(critical_signals_for_parent[0][1], 3)

            for child_id in ("task-child-0", "task-child-1", "task-child-2"):
                self.assertFalse(any(s[0] == child_id for s in signals))

    def test_low_fanout_does_not_fire_critical_path_callback(self) -> None:
        signals: list[str] = []

        def on_signal(task_id: str, fanout: int, digest):
            signals.append(task_id)

        with tempfile.TemporaryDirectory() as tmp:
            runtime = self.make_runtime(
                tmp,
                max_workers=4,
                on_critical_path_progress=on_signal,
                critical_path_fanout=2,
            )
            chain = [
                AgentTaskSpec(kind="tool", goal="a", handler=self._trivial_handler("a"), task_id="task-a"),
                AgentTaskSpec(
                    kind="tool",
                    goal="b",
                    handler=self._trivial_handler("b"),
                    task_id="task-b",
                    dependencies=[DependencyPredicate.task_completed("task-a")],
                ),
            ]
            runtime.submit(chain)
            self.assertTrue(runtime.wait_all(timeout=2))
            time.sleep(0.05)
            self.assertEqual(signals, [])

    def test_critical_path_signal_budget_caps_callbacks(self) -> None:
        signals: list[str] = []

        def on_signal(task_id: str, fanout: int, digest):
            signals.append(task_id)

        with tempfile.TemporaryDirectory() as tmp:
            runtime = self.make_runtime(
                tmp,
                max_workers=4,
                on_critical_path_progress=on_signal,
                critical_path_fanout=2,
                critical_path_signal_budget=1,
            )

            # 两层 fan-out，预算 1，只应该响一次
            tasks = [
                AgentTaskSpec(kind="tool", goal="a", handler=self._trivial_handler("a"), task_id="task-a"),
                AgentTaskSpec(
                    kind="tool",
                    goal="b1",
                    handler=self._trivial_handler("b1"),
                    task_id="task-b1",
                    dependencies=[DependencyPredicate.task_completed("task-a")],
                ),
                AgentTaskSpec(
                    kind="tool",
                    goal="b2",
                    handler=self._trivial_handler("b2"),
                    task_id="task-b2",
                    dependencies=[DependencyPredicate.task_completed("task-a")],
                ),
                AgentTaskSpec(
                    kind="tool",
                    goal="c1",
                    handler=self._trivial_handler("c1"),
                    task_id="task-c1",
                    dependencies=[DependencyPredicate.task_completed("task-b1")],
                ),
                AgentTaskSpec(
                    kind="tool",
                    goal="c2",
                    handler=self._trivial_handler("c2"),
                    task_id="task-c2",
                    dependencies=[DependencyPredicate.task_completed("task-b1")],
                ),
            ]
            runtime.submit(tasks)
            self.assertTrue(runtime.wait_all(timeout=2))
            time.sleep(0.05)
            self.assertEqual(len(signals), 1)

    def test_critical_path_signal_window_expires_and_reallows(self) -> None:
        """Sliding window: after window elapses, budget resets and signals fire again."""
        signals: list[str] = []

        def on_signal(task_id: str, fanout: int, digest):
            signals.append(task_id)

        with tempfile.TemporaryDirectory() as tmp:
            runtime = self.make_runtime(
                tmp,
                max_workers=4,
                on_critical_path_progress=on_signal,
                critical_path_fanout=2,
                critical_path_signal_budget=1,
                critical_path_signal_window=0.1,
            )

            phase1 = [
                AgentTaskSpec(kind="tool", goal="a", handler=self._trivial_handler("a"), task_id="task-a"),
                AgentTaskSpec(
                    kind="tool", goal="b1", handler=self._trivial_handler("b1"),
                    task_id="task-b1", dependencies=[DependencyPredicate.task_completed("task-a")],
                ),
                AgentTaskSpec(
                    kind="tool", goal="b2", handler=self._trivial_handler("b2"),
                    task_id="task-b2", dependencies=[DependencyPredicate.task_completed("task-a")],
                ),
            ]
            runtime.submit(phase1)
            self.assertTrue(runtime.wait_all(timeout=2))
            time.sleep(0.05)
            self.assertEqual(len(signals), 1)

            # 等窗口耗尽
            time.sleep(0.15)

            phase2 = [
                AgentTaskSpec(kind="tool", goal="x", handler=self._trivial_handler("x"), task_id="task-x"),
                AgentTaskSpec(
                    kind="tool", goal="y1", handler=self._trivial_handler("y1"),
                    task_id="task-y1", dependencies=[DependencyPredicate.task_completed("task-x")],
                ),
                AgentTaskSpec(
                    kind="tool", goal="y2", handler=self._trivial_handler("y2"),
                    task_id="task-y2", dependencies=[DependencyPredicate.task_completed("task-x")],
                ),
            ]
            runtime.submit(phase2)
            self.assertTrue(runtime.wait_all(timeout=2))
            time.sleep(0.05)
            self.assertEqual(len(signals), 2)
            self.assertEqual(signals, ["task-a", "task-x"])


if __name__ == "__main__":
    unittest.main()
