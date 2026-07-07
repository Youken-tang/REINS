"""Regression test for OpenAIResponsesTransport SSE event reorder (audit L6).

Trace observation: if `response.function_call_arguments.delta` events
arrive before `response.output_item.added`, the deltas accumulate under
the placeholder key `output-{output_index}`. When the canonical key
(via call_id / item_id) shows up later, a naive implementation creates
a new call_parts entry, leaving the partial arguments orphaned in the
placeholder. The result is two ToolCalls in the final NormalizedResponse:
one with name+id but empty args, one with args but no name.

OpenAI's real SSE today emits `added` before any delta so the race
does not fire in practice, but the audit (L6) flagged the gap. This
test simulates the reordered sequence and asserts the transport merges
the placeholder accumulator into the canonical entry.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from high_agent.llm.client import ModelSettings
from high_agent.llm.transport import OpenAIResponsesTransport


class ResponsesStreamReorderTests(unittest.TestCase):
    def _transport(self) -> OpenAIResponsesTransport:
        return OpenAIResponsesTransport(
            ModelSettings("openai", "m", "https://api.openai.com/v1", "codex_responses", "k")
        )

    def test_delta_before_output_item_added_merges_into_canonical_key(self) -> None:
        transport = self._transport()
        # Provider sends args fragments *before* the output_item.added with
        # the canonical call_id. The transport must merge them.
        events = [
            {"event": "response.function_call_arguments.delta",
             "data": {"type": "response.function_call_arguments.delta", "output_index": 0,
                      "delta": '{"path":"a.txt",'}},
            {"event": "response.function_call_arguments.delta",
             "data": {"type": "response.function_call_arguments.delta", "output_index": 0,
                      "delta": '"content":"A"}'}},
            {"event": "response.output_item.added",
             "data": {"type": "response.output_item.added", "output_index": 0,
                      "item": {"type": "function_call", "call_id": "call_abc",
                               "name": "write_file", "arguments": ""}}},
            {"event": "response.output_item.done",
             "data": {"type": "response.output_item.done", "output_index": 0,
                      "item": {"type": "function_call", "call_id": "call_abc",
                               "name": "write_file",
                               "arguments": '{"path":"a.txt","content":"A"}'}}},
        ]
        normalized = transport.normalize_stream_events(events)
        self.assertIsNotNone(normalized.tool_calls)
        self.assertEqual(len(normalized.tool_calls), 1)
        call = normalized.tool_calls[0]
        self.assertEqual(call.id, "call_abc")
        self.assertEqual(call.name, "write_file")
        self.assertEqual(call.arguments, '{"path":"a.txt","content":"A"}')

    def test_delta_after_output_item_added_still_works(self) -> None:
        # Baseline: events arrive in the canonical OpenAI order. Behaviour
        # must remain identical to pre-fix.
        transport = self._transport()
        events = [
            {"event": "response.output_item.added",
             "data": {"type": "response.output_item.added", "output_index": 0,
                      "item": {"type": "function_call", "call_id": "call_xyz",
                               "name": "write_file", "arguments": ""}}},
            {"event": "response.function_call_arguments.delta",
             "data": {"type": "response.function_call_arguments.delta", "output_index": 0,
                      "delta": '{"path":"b.txt","content":"B"}'}},
            {"event": "response.function_call_arguments.done",
             "data": {"type": "response.function_call_arguments.done", "output_index": 0,
                      "arguments": '{"path":"b.txt","content":"B"}'}},
        ]
        normalized = transport.normalize_stream_events(events)
        self.assertEqual(len(normalized.tool_calls), 1)
        self.assertEqual(normalized.tool_calls[0].id, "call_xyz")
        self.assertEqual(normalized.tool_calls[0].arguments, '{"path":"b.txt","content":"B"}')


if __name__ == "__main__":
    unittest.main()
