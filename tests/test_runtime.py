from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from high_agent.agent import MainAgent
from high_agent.runtime import AgentTaskSpec, ComponentWrite, DependencyPredicate, TaskResult
from high_agent.runtime.resource_access import ResourceAccess, access_conflicts
from high_agent.runtime.scheduler import CausalRuntime


class RuntimeTests(unittest.TestCase):
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

    def test_independent_tasks_run_in_parallel_and_deliver_incrementally(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self.make_runtime(tmp, max_workers=3)

            def sleeper(name: str, delay: float):
                def _handle(ctx):
                    time.sleep(delay)
                    return TaskResult.completed(name)
                return _handle

            start = time.monotonic()
            runtime.submit([
                AgentTaskSpec(kind="tool", goal="fast", handler=sleeper("fast", 0.05)),
                AgentTaskSpec(kind="tool", goal="slow-a", handler=sleeper("slow-a", 0.25)),
                AgentTaskSpec(kind="tool", goal="slow-b", handler=sleeper("slow-b", 0.25)),
            ])
            first = runtime.wait_next_delivery(timeout=0.15)
            self.assertIsNotNone(first)
            self.assertIn("fast", first.summaries())
            self.assertGreater(runtime.pending_count(), 0)
            self.assertTrue(runtime.wait_all(timeout=1))
            elapsed = time.monotonic() - start
            self.assertLess(elapsed, 0.45)

    def test_parent_directory_creation_unblocks_child_file_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root_dir = Path(tmp) / "tree"
            child_file = root_dir / "child.txt"
            runtime = self.make_runtime(tmp, max_workers=2)
            order: list[str] = []

            def make_root(ctx):
                time.sleep(0.05)
                root_dir.mkdir()
                order.append("root")
                return TaskResult.completed("root ready", writes=[ComponentWrite(f"dir:{root_dir}", True)])

            def make_child(ctx):
                self.assertTrue(root_dir.exists())
                child_file.write_text("ok", encoding="utf-8")
                order.append("child")
                return TaskResult.completed("child ready", writes=[ComponentWrite(f"file:{child_file}", "ok")])

            runtime.submit([
                AgentTaskSpec(kind="tool", goal="mkdir", writes={f"dir:{root_dir}"}, handler=make_root),
                AgentTaskSpec(kind="tool", goal="write child", writes={f"file:{child_file}"}, handler=make_child),
            ])
            self.assertTrue(runtime.wait_all(timeout=1))
            self.assertEqual(order, ["root", "child"])
            self.assertTrue(child_file.exists())

    def test_conflicting_writes_serialize_unrelated_writes_parallelize(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self.make_runtime(tmp, max_workers=3)
            same_file = Path(tmp) / "same.txt"
            other_file = Path(tmp) / "other.txt"
            active_same = 0
            max_same = 0
            active_any = 0
            max_any = 0
            lock = threading.Lock()

            def tracked(label: str, path: Path):
                def _handle(ctx):
                    nonlocal active_same, max_same, active_any, max_any
                    with lock:
                        active_any += 1
                        max_any = max(max_any, active_any)
                        if path == same_file:
                            active_same += 1
                            max_same = max(max_same, active_same)
                    time.sleep(0.08)
                    path.write_text(label, encoding="utf-8")
                    with lock:
                        active_any -= 1
                        if path == same_file:
                            active_same -= 1
                    return TaskResult.completed(label, writes=[ComponentWrite(f"file:{path}", label)])
                return _handle

            runtime.submit([
                AgentTaskSpec(kind="tool", goal="same 1", resource_access=ResourceAccess.write(f"file:{same_file}"), handler=tracked("same1", same_file)),
                AgentTaskSpec(kind="tool", goal="same 2", resource_access=ResourceAccess.write(f"file:{same_file}"), handler=tracked("same2", same_file)),
                AgentTaskSpec(kind="tool", goal="other", resource_access=ResourceAccess.write(f"file:{other_file}"), handler=tracked("other", other_file)),
            ])
            self.assertTrue(runtime.wait_all(timeout=1))
            self.assertEqual(max_same, 1)
            self.assertGreaterEqual(max_any, 2)
            self.assertIn("conflicts:", runtime.status_digest().text)

    def test_external_reads_and_declared_readonly_processes_do_not_globally_serialize(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self.make_runtime(tmp, max_workers=4, delivery_debounce=0.0)
            active = 0
            max_active = 0
            lock = threading.Lock()

            def tracked(label: str):
                def _handle(ctx):
                    nonlocal active, max_active
                    with lock:
                        active += 1
                        max_active = max(max_active, active)
                    time.sleep(0.06)
                    with lock:
                        active -= 1
                    return TaskResult.completed(label)
                return _handle

            runtime.submit([
                AgentTaskSpec(kind="tool", goal="fetch a", resource_access=ResourceAccess.external_read("external:https://a.test"), handler=tracked("a")),
                AgentTaskSpec(kind="tool", goal="fetch b", resource_access=ResourceAccess.external_read("external:https://b.test"), handler=tracked("b")),
                AgentTaskSpec(kind="tool", goal="declared readonly process", resource_access=ResourceAccess.empty(), handler=tracked("p")),
                AgentTaskSpec(kind="tool", goal="local write", resource_access=ResourceAccess.write(f"file:{Path(tmp) / 'x.txt'}"), handler=tracked("w")),
            ])
            self.assertTrue(runtime.wait_all(timeout=1))
            self.assertGreaterEqual(max_active, 3)
            self.assertFalse(access_conflicts(ResourceAccess.external_read("external:a"), ResourceAccess.external_read("external:b")))
            self.assertTrue(access_conflicts(ResourceAccess.external_write("external:a"), ResourceAccess.external_read("external:b")))

    def test_dynamic_task_insertion_runs_while_unrelated_task_is_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self.make_runtime(tmp, max_workers=2)
            done: list[str] = []

            def discover(ctx):
                child = AgentTaskSpec(kind="tool", goal="child", handler=lambda c: _complete("child", done))
                return TaskResult.completed("parent", discovered_tasks=[child])

            def slow(ctx):
                time.sleep(0.25)
                return _complete("slow", done)

            runtime.submit([
                AgentTaskSpec(kind="planner", goal="discover", handler=discover),
                AgentTaskSpec(kind="tool", goal="slow", handler=slow),
            ])
            deadline = time.monotonic() + 0.2
            saw_child = False
            while time.monotonic() < deadline:
                batch = runtime.wait_next_delivery(timeout=0.05)
                if batch and "child" in batch.summaries():
                    saw_child = True
                    break
            self.assertTrue(saw_child)
            self.assertNotIn("slow", done)
            self.assertTrue(runtime.wait_all(timeout=1))

    def test_dependency_cycle_is_linearized_as_non_parallel_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self.make_runtime(tmp, max_workers=2, delivery_debounce=0.0)
            order: list[str] = []

            def mark(label: str):
                def _handle(ctx):
                    order.append(label)
                    return TaskResult.completed(label)
                return _handle

            runtime.submit([
                AgentTaskSpec(kind="tool", goal="a", task_id="a", dependencies=[DependencyPredicate.task_completed("b")], handler=mark("a")),
                AgentTaskSpec(kind="tool", goal="b", task_id="b", dependencies=[DependencyPredicate.task_completed("a")], handler=mark("b")),
            ])
            self.assertTrue(runtime.wait_all(timeout=1))
            self.assertEqual(order, ["a", "b"])
            self.assertEqual(runtime._tasks["a"].metadata.get("causal_boundary"), "cycle_linearized")

    def test_main_context_records_delivery_summary_not_worker_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self.make_runtime(tmp, max_workers=2)
            agent = MainAgent("objective", runtime)
            agent.submit_tasks([
                AgentTaskSpec(kind="worker", goal="worker", input={"transcript": "SECRET INTERNAL"}, handler=lambda c: TaskResult.completed("public summary")),
                AgentTaskSpec(kind="tool", goal="slow", handler=lambda c: (time.sleep(0.2), TaskResult.completed("slow"))[1]),
            ])
            batch = agent.wait_delivery(timeout=0.2)
            self.assertIsNotNone(batch)
            rendered = agent.context.render()
            self.assertIn("public summary", rendered)
            self.assertNotIn("SECRET INTERNAL", rendered)
            self.assertGreater(runtime.pending_count(), 0)

    def test_barrier_buffers_non_barrier_delivery_until_barrier_resolves(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self.make_runtime(tmp, max_workers=2, delivery_debounce=0.0)

            runtime.submit([
                AgentTaskSpec(kind="tool", goal="barrier", barrier="interactive", handler=lambda c: (time.sleep(0.18), TaskResult.completed("barrier done"))[1]),
                AgentTaskSpec(kind="tool", goal="fast", handler=lambda c: (time.sleep(0.03), TaskResult.completed("fast done"))[1]),
            ])
            self.assertIsNone(runtime.wait_next_delivery(timeout=0.08))
            batch = runtime.wait_next_delivery(timeout=0.4)
            self.assertIsNotNone(batch)
            self.assertEqual(batch.summaries(), ["barrier done", "fast done"])

    def test_worker_model_semaphore_limits_concurrent_model_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self.make_runtime(tmp, max_workers=5)
            gate = threading.Semaphore(2)
            active = 0
            max_active = 0
            lock = threading.Lock()

            def model_bound(ctx):
                nonlocal active, max_active
                with gate:
                    with lock:
                        active += 1
                        max_active = max(max_active, active)
                    time.sleep(0.07)
                    with lock:
                        active -= 1
                return TaskResult.completed(ctx.task.goal)

            runtime.submit([
                AgentTaskSpec(kind="worker", goal=f"w{i}", handler=model_bound)
                for i in range(5)
            ])
            self.assertTrue(runtime.wait_all(timeout=1))
            self.assertEqual(max_active, 2)

    def test_ledger_records_task_timing_and_delivery_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self.make_runtime(tmp, max_workers=1, delivery_debounce=0.0)

            def sleeper(ctx):
                time.sleep(0.04)
                return TaskResult.completed("slept")

            runtime.submit([AgentTaskSpec(kind="tool", goal="sleep", task_id="sleep", handler=sleeper)])
            self.assertTrue(runtime.wait_all(timeout=1))
            timing = runtime.ledger.timing()
            self.assertEqual(timing.completed_tasks, 1)
            self.assertGreaterEqual(timing.task_seconds, 0.03)
            self.assertGreaterEqual(runtime.ledger.task_duration("sleep") or 0.0, 0.03)
            self.assertIn("time:", runtime.status_digest().text)
            batch = runtime.wait_next_delivery(timeout=0.1)
            self.assertIsNotNone(batch)
            assert batch is not None
            self.assertGreaterEqual(float(batch.events[0].metadata["duration_seconds"]), 0.03)

    def test_trace_jsonl_records_runtime_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.jsonl"
            runtime = self.make_runtime(tmp, max_workers=1, trace_path=trace)
            runtime.submit([
                AgentTaskSpec(kind="tool", goal="x", writes={"artifact:x"}, handler=lambda c: TaskResult.completed("x"))
            ])
            self.assertTrue(runtime.wait_all(timeout=1))
            runtime.wait_next_delivery(timeout=0.1)
            runtime.shutdown()
            rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
            events = [row["event"] for row in rows]
            self.assertIn("runtime.started", events)
            self.assertIn("task.submitted", events)
            self.assertIn("task.started", events)
            self.assertIn("task.completed", events)
            completed = next(row for row in rows if row["event"] == "task.completed")
            self.assertIn("duration_seconds", completed)
            self.assertIn("component.updated", events)
            self.assertIn("delivery.delivered", events)
            self.assertIn("runtime.shutdown", events)


def _complete(label: str, done: list[str]) -> TaskResult:
    done.append(label)
    return TaskResult.completed(label)


if __name__ == "__main__":
    unittest.main()
