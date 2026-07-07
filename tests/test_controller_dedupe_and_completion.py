"""Regression tests for AgentRunController dedupe and completion-gate fixes.

Covers:
- Dedupe key no longer carries snapshot_seq, so identical (name, args)
  emitted by parallel planners on different snapshots collapse to one.
- A completed non-readonly call cannot be re-issued with the same args.
- Read-only repeats (e.g. read_file) are NOT blocked by completed-set,
  matching the planner's legitimate need to re-read after writes.
- final_candidate is accepted when only read-only progress happened
  after it was emitted (prior status_seq logic spuriously rejected it).
- final_candidate is restarted when a non-readonly task completed after it.
"""

from __future__ import annotations

import json
import sys
import threading
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from high_agent.agent.controller import (
    AgentRunController,
    _PlannerRequest,
    _tool_call_dedupe_key,
)
from high_agent.agent.tool_calls import NormalizedToolCall
from high_agent.llm.types import NormalizedResponse, ToolCall
from high_agent.runtime.resource_access import ResourceAccess
from high_agent.runtime.types import DeliveryEvent, TaskResult


def _make_call(name: str, args: dict[str, Any], call_id: str = "c0") -> NormalizedToolCall:
    return NormalizedToolCall(
        call=ToolCall(id=call_id, name=name, arguments=json.dumps(args)),
        original_name=name,
    )


class _FakeTrace:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def emit(self, event: str, **payload: Any) -> None:
        self.events.append((event, payload))

    def emit_typed(self, event: str, **payload: Any) -> None:
        self.emit(event, **payload)


@dataclass
class _FakeTask:
    metadata: dict[str, Any] = field(default_factory=dict)
    input: dict[str, Any] = field(default_factory=dict)


class _FakeStore:
    def __init__(self) -> None:
        self.writes: list[tuple[str, str]] = []

    def write(self, key: str, value: str) -> None:
        self.writes.append((key, value))


class _FakeRuntime:
    def __init__(self) -> None:
        self.trace = _FakeTrace()
        self.deliveries: list[Any] = []
        self._pending_count = 0
        self.store = _FakeStore()

    def pending_count(self) -> int:
        return self._pending_count

    def status_digest(self) -> Any:
        return type("Digest", (), {"text": "runtime idle", "seq": 0})()

    def cancel_stale_tasks(self, max_seconds: float = 120.0) -> list[str]:
        return []

    def wait_next_delivery(self, timeout: float | None = None) -> Any:
        if self.deliveries:
            return self.deliveries.pop(0)
        return None


class _FakeAgent:
    def __init__(self) -> None:
        self.runtime = _FakeRuntime()
        self._active_action_index: dict[tuple[str, str], int] = {}
        self._active_action_index_lock = threading.Lock()
        self.submitted: list[list[NormalizedToolCall]] = []
        self.last_usage = None

    def _normalize_tool_calls(self, tool_calls: list[ToolCall]) -> list[NormalizedToolCall]:
        return [_make_call(call.name, call.args_dict(), call.id) for call in tool_calls]

    def _submit_normalized_tool_calls(self, tool_calls: list[NormalizedToolCall]) -> list[str]:
        self.submitted.append(tool_calls)
        return [item.call.id or f"call-{index}" for index, item in enumerate(tool_calls)]

    def _append_delivery_messages(self, messages: list[dict[str, Any]], batch: Any) -> int:
        return 0


def _make_controller() -> AgentRunController:
    agent = _FakeAgent()
    controller = AgentRunController(
        agent=agent,
        objective="test",
        messages=[],
    )
    return controller


class DedupeKeyTests(unittest.TestCase):
    def test_dedupe_key_excludes_snapshot_seq(self) -> None:
        item = _make_call("write_file", {"path": "a.txt", "content": "hi"})
        key = _tool_call_dedupe_key(item)
        self.assertEqual(key[0], "write_file")
        # canonical args does not embed any snapshot index.
        self.assertNotIn("snapshot", key[1])
        self.assertEqual(json.loads(key[1]), {"path": "a.txt", "content": "hi"})

    def test_dedupe_key_canonicalizes_arg_order(self) -> None:
        a = _make_call("write_file", {"path": "a.txt", "content": "hi"})
        b = _make_call("write_file", {"content": "hi", "path": "a.txt"})
        self.assertEqual(_tool_call_dedupe_key(a), _tool_call_dedupe_key(b))

    def test_dedupe_key_strips_internal_underscore_args(self) -> None:
        # AgentLoop (formerly SubAgent) stamps `_depth` into
        # delegate_task args before submitting the worker AgentTaskSpec;
        # ToolRegistry.task_from_call also adds `_workspace_root` for handler
        # dispatch. Neither key is a model-visible argument, so the dedupe key
        # must ignore them. Otherwise the same delegate at different depths
        # counts as two distinct calls (recursive delegate guard relies on
        # depth-bumping
        # being idempotent for dedupe), and a model that echoes back
        # `_depth=0` could reset the recursion guard by appearing fresh.
        bare = _make_call("delegate_task", {"goal": "x", "tasks": [{"goal": "a"}]})
        with_depth = _make_call(
            "delegate_task",
            {"goal": "x", "tasks": [{"goal": "a"}], "_depth": 1},
        )
        with_workspace = _make_call(
            "delegate_task",
            {"goal": "x", "tasks": [{"goal": "a"}], "_workspace_root": "/tmp"},
        )
        self.assertEqual(_tool_call_dedupe_key(bare), _tool_call_dedupe_key(with_depth))
        self.assertEqual(_tool_call_dedupe_key(bare), _tool_call_dedupe_key(with_workspace))


class DedupeAcrossSnapshotsTests(unittest.TestCase):
    def test_same_call_emitted_on_different_snapshots_collapses_to_one(self) -> None:
        controller = _make_controller()
        calls = [_make_call("write_file", {"path": "shared.txt", "content": "X"}, f"c{i}") for i in range(4)]
        # First emission keeps it.
        first, dropped = controller._dedupe_tool_calls(snapshot_seq=10, calls=[calls[0]])
        self.assertEqual(len(first), 1)
        self.assertEqual(dropped, [])
        # Subsequent emissions on later snapshots all drop.
        for i, snap in enumerate((11, 12, 13), start=1):
            out, dropped = controller._dedupe_tool_calls(snapshot_seq=snap, calls=[calls[i]])
            self.assertEqual(out, [])
            self.assertEqual(len(dropped), 1)
            self.assertEqual(dropped[0][1], "in_flight")
        events = [name for name, _ in controller.agent.runtime.trace.events]
        self.assertEqual(events.count("planner.ignored_duplicate"), 3)


class DedupeAgainstCompletedTests(unittest.TestCase):
    def test_completed_write_blocks_repeat(self) -> None:
        controller = _make_controller()
        canonical = json.dumps({"path": "a.txt", "content": "X"}, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        controller._completed_action_keys.add(("write_file", canonical))

        repeat = _make_call("write_file", {"path": "a.txt", "content": "X"})
        out, dropped = controller._dedupe_tool_calls(snapshot_seq=20, calls=[repeat])
        self.assertEqual(out, [])
        self.assertEqual(len(dropped), 1)
        self.assertEqual(dropped[0][1], "already_completed")
        names = [name for name, _ in controller.agent.runtime.trace.events]
        self.assertIn("planner.ignored_duplicate", names)
        # Reason should distinguish from in_flight.
        last_reason = controller.agent.runtime.trace.events[-1][1].get("reason")
        self.assertEqual(last_reason, "already_completed")

    def test_readonly_repeat_is_not_blocked_by_completed_set(self) -> None:
        controller = _make_controller()
        canonical = json.dumps({"path": "a.txt"}, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        # Even if read_file somehow leaked into the completed set, the dedupe
        # path must still allow re-reads — they are idempotent and necessary
        # after a write changes the file.
        controller._completed_action_keys.add(("read_file", canonical))

        repeat = _make_call("read_file", {"path": "a.txt"})
        out, dropped = controller._dedupe_tool_calls(snapshot_seq=30, calls=[repeat])
        self.assertEqual(len(out), 1)
        self.assertEqual(dropped, [])

    def test_command_tool_repeat_is_not_blocked_by_completed_set(self) -> None:
        controller = _make_controller()
        # Critical: run_tests after a fix must be allowed even though an
        # earlier run_tests with the same command already completed (failed).
        # This mirrors v040's "build → tests fail → patch → tests pass" loop.
        canonical = json.dumps({"command": "pytest"}, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        controller._completed_action_keys.add(("run_tests", canonical))

        repeat = _make_call("run_tests", {"command": "pytest"})
        out, dropped = controller._dedupe_tool_calls(snapshot_seq=40, calls=[repeat])
        self.assertEqual(len(out), 1)
        self.assertEqual(dropped, [])

    def test_mkdir_is_idempotent_and_not_in_completed_set(self) -> None:
        # Regression for the dead-loop discovered in
        # session-e8fc651f6acd/run-1779720182646.jsonl: the planner kept
        # emitting the same mkdir while no other progress was visible, the
        # dedupe layer dropped every retry, no task entered the runtime, and
        # the run hung. mkdir is exist_ok=True (idempotent), so even if it
        # leaks into _completed_action_keys, the dedupe path must allow it
        # through — only in_flight collapse remains.
        controller = _make_controller()
        canonical = json.dumps({"path": "foo"}, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        controller._completed_action_keys.add(("mkdir", canonical))

        repeat = _make_call("mkdir", {"path": "foo"})
        out, dropped = controller._dedupe_tool_calls(snapshot_seq=50, calls=[repeat])
        self.assertEqual(len(out), 1)
        self.assertEqual(dropped, [])


class NoteCompletedActionTests(unittest.TestCase):
    def test_only_file_mutating_tools_recorded(self) -> None:
        controller = _make_controller()
        write_task = _FakeTask(input={"name": "write_file", "args": {"path": "a.txt", "content": "X"}})
        controller._note_completed_action("write_file", write_task, succeeded=True)
        self.assertEqual(len(controller._completed_action_keys), 1)

        # Command-style tools (run_tests, terminal, etc.) must NOT be added —
        # the planner can legitimately re-issue them after intervening writes.
        run_task = _FakeTask(input={"name": "run_tests", "args": {"command": "pytest"}})
        controller._note_completed_action("run_tests", run_task, succeeded=True)
        self.assertEqual(len(controller._completed_action_keys), 1)

        # Read-only tools are likewise not stamped.
        read_task = _FakeTask(input={"name": "read_file", "args": {"path": "a.txt"}})
        controller._note_completed_action("read_file", read_task, succeeded=True)
        self.assertEqual(len(controller._completed_action_keys), 1)

    def test_failed_write_can_be_retried(self) -> None:
        controller = _make_controller()
        # Simulate a write that previously failed (e.g. permission denied);
        # the planner must be allowed to retry it.
        write_task = _FakeTask(input={"name": "write_file", "args": {"path": "a.txt", "content": "X"}})
        controller._submitted_action_keys.add(("write_file", json.dumps({"path": "a.txt", "content": "X"}, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)))
        controller._note_completed_action("write_file", write_task, succeeded=False)
        # Failed write: in-flight stamp cleared, but NOT added to completed.
        self.assertEqual(len(controller._submitted_action_keys), 0)
        self.assertEqual(len(controller._completed_action_keys), 0)

    def test_failed_run_tests_clears_in_flight_stamp(self) -> None:
        controller = _make_controller()
        run_task = _FakeTask(input={"name": "run_tests", "args": {"command": "pytest"}})
        controller._submitted_action_keys.add(("run_tests", json.dumps({"command": "pytest"}, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)))
        controller._note_completed_action("run_tests", run_task, succeeded=False)
        # In-flight cleared so a retry can be issued; nothing in completed.
        self.assertEqual(len(controller._submitted_action_keys), 0)
        self.assertEqual(len(controller._completed_action_keys), 0)


class DroppedToolCallFeedbackTests(unittest.TestCase):
    def test_record_dropped_tool_calls_writes_status_note(self) -> None:
        # Regression for session-e8fc651f6acd dead-loop: when every tool call
        # from a planner is dropped by the dedupe layer, no task enters the
        # runtime, no delivery fires, the ledger digest does not change, and
        # the same prompt would yield the same dropped calls forever. The
        # controller must surface a status note so the next planner sees
        # explicit feedback that breaks the cycle.
        controller = _make_controller()
        request = _PlannerRequest(
            request_id=7,
            snapshot_seq=3,
            digest_text="runtime idle",
            messages=[],
        )
        # Pre-stamp mkdir as in flight so dedupe rejects the next emission.
        canonical = json.dumps({"path": "foo"}, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        controller._submitted_action_keys.add(("mkdir", canonical))

        repeat = _make_call("mkdir", {"path": "foo"})
        out, dropped = controller._dedupe_tool_calls(snapshot_seq=3, calls=[repeat])
        self.assertEqual(out, [])
        self.assertEqual(len(dropped), 1)

        notes_before = list(controller._status_notes)
        controller._record_dropped_tool_calls(request, dropped)
        notes_after = controller._status_notes
        self.assertEqual(len(notes_after), len(notes_before) + 1)
        self.assertIn("dedupe", notes_after[-1])
        self.assertIn("mkdir", notes_after[-1])
        self.assertIn("in_flight", notes_after[-1])
        # A planner fact is also written so it shows up in the next refill.
        with controller._fact_lock:
            kinds = [fact.kind for fact in controller._planner_facts]
        self.assertIn("status", kinds)


class PlannerLifecycleTests(unittest.TestCase):
    def test_abandoned_streaming_planner_callback_does_not_submit_tool(self) -> None:
        controller = _make_controller()
        request = _PlannerRequest(
            request_id=1,
            snapshot_seq=0,
            digest_text="runtime idle",
            messages=[],
        )
        controller._register_planner_request(request)
        controller._abandon_planner_request(request, reason="test")

        captured: dict[str, Any] = {}

        class _StreamingModel:
            def complete_streaming(self, messages: list[dict], tools: list[dict] | None = None, on_tool_call: Any = None, **params: Any) -> NormalizedResponse:
                captured["on_tool_call"] = on_tool_call
                on_tool_call(ToolCall("late", "write_file", json.dumps({"path": "late.txt", "content": "x"})))
                return NormalizedResponse("", None, "stop")

        controller.agent.model_client = _StreamingModel()
        controller.agent.tools = type("Tools", (), {"definitions": lambda self: []})()

        result = controller._run_planner_request(request)
        self.assertEqual(result.early_dispatched_ids, frozenset())
        self.assertEqual(controller.agent.submitted, [])
        events = [name for name, _ in controller.agent.runtime.trace.events]
        self.assertIn("planner.ignored_late_output", events)

    def test_max_iterations_message_is_non_empty(self) -> None:
        controller = _make_controller()
        controller.max_iterations = 1
        self.assertIn("Stopped after 1 planner requests", controller._max_iterations_message())

    def test_abandon_synthesizes_assistant_message_for_early_dispatch(self) -> None:
        # when a planner is abandoned (e.g. _cancel_stale_planners
        # tripped) but tool calls were already dispatched mid-stream, the
        # runtime will still deliver tool results for those calls. If
        # self.messages does not contain an assistant entry carrying the
        # corresponding tool_call_ids, the next planner's
        # sanitize_tool_protocol_messages drops the tool results as orphans
        # and the model never sees what its work produced.
        controller = _make_controller()
        request = _PlannerRequest(
            request_id=42,
            snapshot_seq=0,
            digest_text="runtime idle",
            messages=[],
        )
        controller._register_planner_request(request)
        # Simulate that streaming dispatched two tool calls successfully.
        item_a = _make_call("write_file", {"path": "a.txt", "content": "A"}, "call-a")
        item_b = _make_call("write_file", {"path": "b.txt", "content": "B"}, "call-b")
        request._early_dispatched_calls.extend([item_a, item_b])

        controller._abandon_planner_request(request, reason="timeout")

        assistant_msgs = [m for m in controller.messages if m.get("role") == "assistant"]
        self.assertEqual(len(assistant_msgs), 1)
        ids = [tc["id"] for tc in assistant_msgs[0]["tool_calls"]]
        self.assertEqual(set(ids), {"call-a", "call-b"})
        # Calling abandon a second time must not duplicate the assistant entry.
        controller._abandon_planner_request(request, reason="timeout")
        assistant_msgs = [m for m in controller.messages if m.get("role") == "assistant"]
        self.assertEqual(len(assistant_msgs), 1)

    def test_abandon_without_dispatched_calls_does_not_inject_message(self) -> None:
        controller = _make_controller()
        request = _PlannerRequest(
            request_id=43,
            snapshot_seq=0,
            digest_text="runtime idle",
            messages=[],
        )
        controller._register_planner_request(request)
        controller._abandon_planner_request(request, reason="run_closed")
        self.assertEqual(controller.messages, [])

    def test_handle_planner_result_keeps_content_when_stop_with_tool_calls(self) -> None:
        # a model emitting `stop` finish_reason with non-empty
        # content alongside residual tool_calls must surface the content as
        # a final_candidate. Previously it was dropped.
        from high_agent.agent.controller import _PlannerResult

        controller = _make_controller()
        controller.agent.tools = type("Tools", (), {"definitions": lambda self: []})()
        controller.agent._record_assistant_message = lambda messages, response: [  # type: ignore[assignment]
            _make_call("write_file", {"path": "x.txt", "content": "X"}, "c-x"),
        ]
        request = _PlannerRequest(
            request_id=99,
            snapshot_seq=5,
            digest_text="runtime idle",
            messages=[],
        )
        controller._register_planner_request(request)
        result = _PlannerResult(
            response=NormalizedResponse(
                content="All done.",
                tool_calls=[ToolCall("c-x", "write_file", json.dumps({"path": "x.txt", "content": "X"}))],
                finish_reason="stop",
            ),
            prompt_estimate=0,
            completion_estimate=0,
        )
        candidate, candidate_seq, had_pending, effect_seq = controller._handle_planner_result(request, result)
        self.assertEqual(candidate, "All done.")
        self.assertEqual(candidate_seq, 5)
        events = [name for name, payload in controller.agent.runtime.trace.events]
        self.assertIn("planner.final_candidate", events)

    def test_handle_planner_result_drops_content_when_finish_is_tool_calls(self) -> None:
        # When finish_reason == "tool_calls", any narrative content is
        # mid-thought; do not surface it as a final.
        from high_agent.agent.controller import _PlannerResult

        controller = _make_controller()
        controller.agent.tools = type("Tools", (), {"definitions": lambda self: []})()
        controller.agent._record_assistant_message = lambda messages, response: [  # type: ignore[assignment]
            _make_call("write_file", {"path": "y.txt", "content": "Y"}, "c-y"),
        ]
        request = _PlannerRequest(
            request_id=100,
            snapshot_seq=6,
            digest_text="runtime idle",
            messages=[],
        )
        controller._register_planner_request(request)
        result = _PlannerResult(
            response=NormalizedResponse(
                content="I'll create y.txt next.",
                tool_calls=[ToolCall("c-y", "write_file", json.dumps({"path": "y.txt", "content": "Y"}))],
                finish_reason="tool_calls",
            ),
            prompt_estimate=0,
            completion_estimate=0,
        )
        candidate, _, _, _ = controller._handle_planner_result(request, result)
        self.assertEqual(candidate, "")


class DeliveryEffectTests(unittest.TestCase):
    def test_completed_readonly_tool_delivery_is_not_effect(self) -> None:
        controller = _make_controller()
        task = _FakeTask(metadata={"tool_name": "slow_read"}, input={"name": "slow_read", "args": {}})
        task.resource_access = ResourceAccess.read("file:/tmp/a.txt")  # type: ignore[attr-defined]
        event = DeliveryEvent(
            seq=1,
            task_id="t1",
            kind="tool",
            summary="ok",
            result=TaskResult.completed("ok"),
            metadata={"tool_name": "slow_read"},
        )
        self.assertFalse(controller._delivery_has_effect(event, task))

    def test_completed_write_tool_delivery_is_effect(self) -> None:
        controller = _make_controller()
        task = _FakeTask(metadata={"tool_name": "write_file"}, input={"name": "write_file", "args": {"path": "a.txt"}})
        task.resource_access = ResourceAccess.write("file:/tmp/a.txt")  # type: ignore[attr-defined]
        event = DeliveryEvent(
            seq=1,
            task_id="t1",
            kind="tool",
            summary="ok",
            result=TaskResult.completed("ok"),
            metadata={"tool_name": "write_file"},
        )
        self.assertTrue(controller._delivery_has_effect(event, task))


if __name__ == "__main__":
    unittest.main()
