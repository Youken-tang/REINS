"""Type-level and scheduler-level tests for the suspend/resume protocol
( introduced the type extensions; wires real suspend
bookkeeping into the scheduler — these tests cover both surfaces).
"""

from __future__ import annotations

import time
import unittest

from high_agent.runtime import (
    AgentTaskSpec,
    DependencyPredicate,
    TaskResult,
)
from high_agent.runtime.ledger import RuntimeLedger
from high_agent.runtime.scheduler import CausalRuntime


class TaskResultSuspendedFactoryTests(unittest.TestCase):
    def test_factory_populates_fields(self) -> None:
        token = "tok-1"
        awaiting = [DependencyPredicate.task_completed("task-x")]
        snapshot = {"messages": []}

        def resume(prev, ctx):  # pragma: no cover - never called in this test
            return TaskResult.completed("noop")

        result = TaskResult.suspended(
            resume_handler=resume,
            awaiting=awaiting,
            suspend_token=token,
            snapshot=snapshot,
            summary="awaiting child",
        )

        self.assertEqual(result.status, "suspended")
        self.assertEqual(result.summary, "awaiting child")
        self.assertEqual(result.suspend_token, token)
        self.assertEqual(result.awaiting, awaiting)
        self.assertIs(result.snapshot, snapshot)
        self.assertIs(result.resume_handler, resume)
        self.assertFalse(result.deliverable)

    def test_empty_awaiting_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TaskResult.suspended(
                resume_handler=lambda prev, ctx: TaskResult.completed(""),
                awaiting=[],
                suspend_token="tok",
            )

    def test_empty_token_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TaskResult.suspended(
                resume_handler=lambda prev, ctx: TaskResult.completed(""),
                awaiting=[DependencyPredicate.task_completed("a")],
                suspend_token="",
            )


class DependencyPredicateExtensionTests(unittest.TestCase):
    def test_task_terminal_predicate(self) -> None:
        pred = DependencyPredicate.task_terminal("task-y")
        self.assertEqual(pred.kind, "task_terminal")
        self.assertEqual(pred.task_id, "task-y")

    def test_future_done_predicate(self) -> None:
        pred = DependencyPredicate.future_done("future-tok-7")
        self.assertEqual(pred.kind, "future_done")
        self.assertEqual(pred.token, "future-tok-7")

    def test_predicate_key_distinguishes_kinds(self) -> None:
        a = DependencyPredicate.task_completed("t1").key()
        b = DependencyPredicate.task_terminal("t1").key()
        c = DependencyPredicate.future_done("t1").key()
        self.assertNotEqual(a, b)
        self.assertNotEqual(b, c)
        self.assertNotEqual(a, c)
        self.assertEqual(a, DependencyPredicate.task_completed("t1").key())


class LedgerSuspendedStateTests(unittest.TestCase):
    def test_set_state_suspended_is_non_terminal(self) -> None:
        ledger = RuntimeLedger()
        spec = AgentTaskSpec(kind="tool", goal="g", task_id="task-1")
        ledger.add_task(spec, "running")
        ledger.set_state("task-1", "suspended", summary="awaiting future")
        rec = ledger.records_snapshot()["task-1"]
        self.assertEqual(rec.state, "suspended")
        self.assertIsNone(rec.finished_at)
        self.assertEqual(rec.summary, "awaiting future")

    def test_digest_text_renders_suspended_section(self) -> None:
        ledger = RuntimeLedger()
        spec = AgentTaskSpec(kind="tool", goal="goal", task_id="task-2")
        ledger.add_task(spec, "running")
        ledger.set_state("task-2", "suspended", summary="awaiting llm")
        digest = ledger.digest()
        self.assertIn("suspended:", digest.text)


class SchedulerSuspendedSemanticsTests(unittest.TestCase):
    """C5: returning TaskResult.suspended must register suspended bookkeeping.

    The scheduler should keep the ledger flipped to "suspended" indefinitely
    while the awaiting predicate is unmet, count it in pending_count(), and
    cleanly transition it to cancelled at shutdown.
    """

    def test_suspended_task_stays_suspended_until_predicate(self) -> None:
        runtime = CausalRuntime(strict_nogil=False, max_workers=1)

        def handler(ctx):
            return TaskResult.suspended(
                resume_handler=lambda prev, c: TaskResult.completed("resumed"),
                awaiting=[DependencyPredicate.task_completed("never")],
                suspend_token="tok-sched-1",
                snapshot={"k": 1},
                summary="awaiting never",
            )

        spec = AgentTaskSpec(kind="tool", goal="suspend-test", handler=handler)
        runtime.start()
        try:
            ids = runtime.submit([spec])
            # Wait for the handler to run and the scheduler to flip to suspended.
            for _ in range(200):
                rec = runtime.ledger.records_snapshot().get(ids[0])
                if rec and rec.state == "suspended":
                    break
                time.sleep(0.01)
            rec = runtime.ledger.records_snapshot()[ids[0]]
            self.assertEqual(rec.state, "suspended")
            self.assertIsNone(rec.finished_at)
            self.assertEqual(runtime.pending_count(), 1)
            self.assertIn(ids[0], runtime._suspended)
        finally:
            runtime.shutdown()
        # Shutdown must have cleaned up suspended bookkeeping.
        self.assertNotIn(ids[0], runtime._suspended)
        rec = runtime.ledger.records_snapshot()[ids[0]]
        self.assertEqual(rec.state, "cancelled")

    def test_suspended_resumes_on_task_completed(self) -> None:
        runtime = CausalRuntime(strict_nogil=False, max_workers=2)
        resumed_with: list[TaskResult] = []

        def child_handler(ctx):
            return TaskResult.completed("child-done", deliverable=False)

        child = AgentTaskSpec(kind="tool", goal="child", handler=child_handler, task_id="child-task")

        def resume(prev, ctx) -> TaskResult:
            resumed_with.append(prev)
            return TaskResult.completed("parent-resumed")

        def parent_handler(ctx):
            return TaskResult.suspended(
                resume_handler=resume,
                awaiting=[DependencyPredicate.task_completed("child-task")],
                suspend_token="tok-resume-1",
                snapshot={"phase": 1},
                summary="awaiting child",
            )

        parent = AgentTaskSpec(kind="tool", goal="parent", handler=parent_handler)

        runtime.start()
        try:
            runtime.submit([child, parent])
            deadline = time.monotonic() + 5.0
            parent_id = parent.task_id
            while time.monotonic() < deadline:
                rec = runtime.ledger.records_snapshot().get(parent_id)
                if rec and rec.state == "completed":
                    break
                time.sleep(0.01)
            rec = runtime.ledger.records_snapshot()[parent_id]
            self.assertEqual(rec.state, "completed")
            self.assertEqual(rec.summary, "parent-resumed")
            self.assertEqual(len(resumed_with), 1)
            self.assertEqual(resumed_with[0].status, "suspended")
            self.assertEqual(resumed_with[0].suspend_token, "tok-resume-1")
        finally:
            runtime.shutdown()

    def test_register_future_resumes_on_future_done(self) -> None:
        import concurrent.futures

        runtime = CausalRuntime(strict_nogil=False, max_workers=2)
        future: concurrent.futures.Future[str] = concurrent.futures.Future()
        token_holder: dict[str, str] = {}

        def resume(prev, ctx) -> TaskResult:
            return TaskResult.completed("after-future")

        def handler(ctx):
            token = ctx.runtime.register_future(future)
            token_holder["token"] = token
            return TaskResult.suspended(
                resume_handler=resume,
                awaiting=[DependencyPredicate.future_done(token)],
                suspend_token=token,
                summary="awaiting future",
            )

        spec = AgentTaskSpec(kind="tool", goal="future-test", handler=handler)
        runtime.start()
        try:
            ids = runtime.submit([spec])
            for _ in range(200):
                rec = runtime.ledger.records_snapshot().get(ids[0])
                if rec and rec.state == "suspended":
                    break
                time.sleep(0.01)
            self.assertEqual(runtime.ledger.records_snapshot()[ids[0]].state, "suspended")
            future.set_result("hello")
            for _ in range(200):
                rec = runtime.ledger.records_snapshot().get(ids[0])
                if rec and rec.state == "completed":
                    break
                time.sleep(0.01)
            rec = runtime.ledger.records_snapshot()[ids[0]]
            self.assertEqual(rec.state, "completed")
            self.assertEqual(rec.summary, "after-future")
        finally:
            runtime.shutdown()


if __name__ == "__main__":
    unittest.main()
