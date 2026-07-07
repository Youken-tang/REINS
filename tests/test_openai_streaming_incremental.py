from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from high_agent.llm.client import ModelClient
from high_agent.llm.types import ToolCall


class FakeTC:
    """Drop-in stand-in for the ToolCall dataclass used by the streaming path."""

    def __init__(self, id: str, name: str, arguments: str) -> None:
        self.id = id
        self.name = name
        self.arguments = arguments


def _make_openai_chunk(
    *,
    index: int = 0,
    call_id: str | None = None,
    name: str | None = None,
    arguments: str | None = None,
    finish_reason: str | None = None,
) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    if call_id or name or arguments:
        fn: dict[str, Any] = {}
        if name is not None:
            fn["name"] = name
        if arguments is not None:
            fn["arguments"] = arguments
        call: dict[str, Any] = {"index": index}
        if call_id is not None:
            call["id"] = call_id
        if fn:
            call["function"] = fn
        delta["tool_calls"] = [call]
    return {"choices": [{"delta": delta, "finish_reason": finish_reason}]}


class OpenAIStreamingIncrementalDispatchTests(unittest.TestCase):
    """v8-P1: OpenAI tool_calls should fire as soon as their JSON closes,
    not wait for finish_reason='tool_calls'."""

    def test_single_tool_call_emits_on_json_close(self) -> None:
        emitted: list[ToolCall] = []

        def on_call(call: Any) -> None:
            emitted.append(call)

        tool_parts: dict[int, dict[str, str]] = {}
        seen: set[int] = set()

        # Header: id + name arrive first.
        ModelClient._check_tool_call_complete(
            _make_openai_chunk(index=0, call_id="call-a", name="write_file"),
            tool_parts, seen, on_call, FakeTC,
        )
        self.assertEqual(emitted, [])

        # Partial arguments — JSON not yet valid.
        ModelClient._check_tool_call_complete(
            _make_openai_chunk(index=0, arguments='{"path": "a.t'),
            tool_parts, seen, on_call, FakeTC,
        )
        self.assertEqual(emitted, [])

        # Closing chunk — JSON now valid, should emit immediately.
        ModelClient._check_tool_call_complete(
            _make_openai_chunk(index=0, arguments='xt", "content": "x"}'),
            tool_parts, seen, on_call, FakeTC,
        )
        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0].id, "call-a")
        self.assertEqual(emitted[0].name, "write_file")
        self.assertEqual(emitted[0].arguments, '{"path": "a.txt", "content": "x"}')

    def test_finish_reason_does_not_double_emit(self) -> None:
        emitted: list[ToolCall] = []

        def on_call(call: Any) -> None:
            emitted.append(call)

        tool_parts: dict[int, dict[str, str]] = {}
        seen: set[int] = set()

        ModelClient._check_tool_call_complete(
            _make_openai_chunk(index=0, call_id="call-a", name="ls", arguments="{}"),
            tool_parts, seen, on_call, FakeTC,
        )
        self.assertEqual(len(emitted), 1)

        ModelClient._check_tool_call_complete(
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
            tool_parts, seen, on_call, FakeTC,
        )
        self.assertEqual(len(emitted), 1)

    def test_two_calls_emit_independently(self) -> None:
        """First tool_call should dispatch even before the second is announced."""
        emitted: list[ToolCall] = []

        def on_call(call: Any) -> None:
            emitted.append(call)

        tool_parts: dict[int, dict[str, str]] = {}
        seen: set[int] = set()

        ModelClient._check_tool_call_complete(
            _make_openai_chunk(index=0, call_id="call-a", name="write_file", arguments='{"p":1}'),
            tool_parts, seen, on_call, FakeTC,
        )
        # Index 0 already emitted; index 1 just announced.
        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0].id, "call-a")

        ModelClient._check_tool_call_complete(
            _make_openai_chunk(index=1, call_id="call-b", name="ls"),
            tool_parts, seen, on_call, FakeTC,
        )
        ModelClient._check_tool_call_complete(
            _make_openai_chunk(index=1, arguments="{}"),
            tool_parts, seen, on_call, FakeTC,
        )
        self.assertEqual(len(emitted), 2)
        self.assertEqual(emitted[1].id, "call-b")

        # Closing finish_reason must not re-emit either.
        ModelClient._check_tool_call_complete(
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
            tool_parts, seen, on_call, FakeTC,
        )
        self.assertEqual(len(emitted), 2)

    def test_finish_reason_fallback_when_no_incremental_close(self) -> None:
        """If JSON parsing never succeeds mid-stream, finish_reason still flushes."""
        emitted: list[ToolCall] = []

        def on_call(call: Any) -> None:
            emitted.append(call)

        tool_parts: dict[int, dict[str, str]] = {}
        seen: set[int] = set()

        # Empty arguments — never a complete JSON object until finish_reason.
        ModelClient._check_tool_call_complete(
            _make_openai_chunk(index=0, call_id="call-a", name="noop", arguments=""),
            tool_parts, seen, on_call, FakeTC,
        )
        self.assertEqual(emitted, [])

        ModelClient._check_tool_call_complete(
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
            tool_parts, seen, on_call, FakeTC,
        )
        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0].arguments, "{}")


if __name__ == "__main__":
    unittest.main()
