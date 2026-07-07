"""feat-S1: OpenAI Responses transport early-dispatch (incremental tool_call emit).

Mirrors the contract verified by tests/test_openai_streaming_incremental.py
for the Chat Completions transport: a tool_call must fire as soon as its
accumulated arguments form valid JSON, and must not double-emit when the
arguments.done / output_item.done events later restate the same call.

Also pins the reorder edge case: deltas arriving before
output_item.added accumulate under the placeholder key, and the early-emit
ledger must migrate to the canonical key when the canonical key shows up,
so the canonical key cannot then re-emit a second time.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from high_agent.llm.client import ModelClient
from high_agent.llm.types import ToolCall


class FakeTC:
    def __init__(self, id: str, name: str, arguments: str) -> None:
        self.id = id
        self.name = name
        self.arguments = arguments


def _added(output_index: int, call_id: str, name: str, arguments: str = "") -> dict[str, Any]:
    return {
        "type": "response.output_item.added",
        "output_index": output_index,
        "item": {
            "type": "function_call",
            "call_id": call_id,
            "name": name,
            "arguments": arguments,
        },
    }


def _delta(output_index: int, delta: str, *, item_id: str | None = None,
           call_id: str | None = None) -> dict[str, Any]:
    data: dict[str, Any] = {
        "type": "response.function_call_arguments.delta",
        "output_index": output_index,
        "delta": delta,
    }
    if item_id is not None:
        data["item_id"] = item_id
    if call_id is not None:
        data["call_id"] = call_id
    return data


def _delta_done(output_index: int, arguments: str, *, item_id: str | None = None,
                call_id: str | None = None) -> dict[str, Any]:
    data: dict[str, Any] = {
        "type": "response.function_call_arguments.done",
        "output_index": output_index,
        "arguments": arguments,
    }
    if item_id is not None:
        data["item_id"] = item_id
    if call_id is not None:
        data["call_id"] = call_id
    return data


def _item_done(output_index: int, call_id: str, name: str, arguments: str) -> dict[str, Any]:
    return {
        "type": "response.output_item.done",
        "output_index": output_index,
        "item": {
            "type": "function_call",
            "call_id": call_id,
            "name": name,
            "arguments": arguments,
        },
    }


class ResponsesStreamingIncrementalDispatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.emitted: list[ToolCall] = []
        self.tool_parts: dict[int, dict[str, str]] = {}
        self.seen: set[int] = set()
        self.responses_parts: dict[str, dict[str, str]] = {}
        self.responses_index_to_key: dict[int, str] = {}
        self.responses_emitted: set[str] = set()

    def _feed(self, event: dict[str, Any]) -> None:
        ModelClient._check_tool_call_complete(
            event, self.tool_parts, self.seen, self.emitted.append, FakeTC,
            responses_parts=self.responses_parts,
            responses_index_to_key=self.responses_index_to_key,
            responses_emitted=self.responses_emitted,
        )

    def test_single_call_emits_on_json_close(self) -> None:
        self._feed(_added(0, "call_a", "write_file"))
        self.assertEqual(self.emitted, [])
        # Partial JSON — not yet closeable.
        self._feed(_delta(0, '{"path":"a.t', call_id="call_a"))
        self.assertEqual(self.emitted, [])
        # Closing fragment — must emit immediately.
        self._feed(_delta(0, 'xt","content":"A"}', call_id="call_a"))
        self.assertEqual(len(self.emitted), 1)
        self.assertEqual(self.emitted[0].id, "call_a")
        self.assertEqual(self.emitted[0].name, "write_file")
        self.assertEqual(self.emitted[0].arguments, '{"path":"a.txt","content":"A"}')
        # Subsequent .done must NOT re-emit.
        self._feed(_delta_done(0, '{"path":"a.txt","content":"A"}', call_id="call_a"))
        self._feed(_item_done(0, "call_a", "write_file",
                              '{"path":"a.txt","content":"A"}'))
        self.assertEqual(len(self.emitted), 1)

    def test_two_calls_emit_independently(self) -> None:
        self._feed(_added(0, "call_a", "write_file"))
        self._feed(_added(1, "call_b", "ls"))
        self._feed(_delta(0, '{"p":1}', call_id="call_a"))
        self.assertEqual(len(self.emitted), 1)
        self.assertEqual(self.emitted[0].id, "call_a")
        self._feed(_delta(1, '{}', call_id="call_b"))
        self.assertEqual(len(self.emitted), 2)
        self.assertEqual(self.emitted[1].id, "call_b")

    def test_invalid_json_only_emits_on_done_fallback(self) -> None:
        self._feed(_added(0, "call_a", "noop"))
        self._feed(_delta(0, "not json", call_id="call_a"))
        self.assertEqual(self.emitted, [])
        # arguments.done with valid JSON must flush.
        self._feed(_delta_done(0, '{"x":1}', call_id="call_a"))
        self.assertEqual(len(self.emitted), 1)
        self.assertEqual(self.emitted[0].arguments, '{"x":1}')

    def test_audit_l6_reorder_no_double_emit(self) -> None:
        # Deltas arrive *before* output_item.added (placeholder accumulation).
        self._feed(_delta(0, '{"path":"a.txt",'))
        self._feed(_delta(0, '"content":"A"}'))
        # Nothing should have emitted yet — placeholder lacks id+name.
        self.assertEqual(self.emitted, [])
        # Canonical key shows up. Migration must fold accumulated args in.
        self._feed(_added(0, "call_abc", "write_file"))
        self.assertEqual(len(self.emitted), 1)
        self.assertEqual(self.emitted[0].id, "call_abc")
        self.assertEqual(self.emitted[0].name, "write_file")
        self.assertEqual(self.emitted[0].arguments, '{"path":"a.txt","content":"A"}')
        # Subsequent restatements via item.done must not double-emit.
        self._feed(_item_done(0, "call_abc", "write_file",
                              '{"path":"a.txt","content":"A"}'))
        self.assertEqual(len(self.emitted), 1)

    def test_emit_locks_arguments_against_later_refinements(self) -> None:
        self._feed(_added(0, "call_a", "write_file"))
        self._feed(_delta(0, '{"path":"a.txt","content":"A"}', call_id="call_a"))
        self.assertEqual(len(self.emitted), 1)
        emitted_args = self.emitted[0].arguments
        # arguments.done arrives later carrying a "fuller" payload.
        self._feed(_delta_done(0, '{"path":"a.txt","content":"A","mode":"0644"}',
                               call_id="call_a"))
        self.assertEqual(len(self.emitted), 1)
        # The emitted ToolCall is what the runtime saw — the dispatch contract
        # locks arguments at first emit.
        self.assertEqual(self.emitted[0].arguments, emitted_args)

    def test_done_only_path_emits_once(self) -> None:
        # Some providers may collapse delta + done into a single .done event.
        self._feed(_item_done(0, "call_a", "ls", "{}"))
        self.assertEqual(len(self.emitted), 1)
        self.assertEqual(self.emitted[0].id, "call_a")
        self.assertEqual(self.emitted[0].arguments, "{}")


if __name__ == "__main__":
    unittest.main()
