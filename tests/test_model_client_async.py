"""Non-blocking ModelClient.complete_async returning a Future.

complete_async runs the existing sync complete() pipeline on a module-level
IO loop thread pool and returns a concurrent.futures.Future. The runtime
binds the future to a suspend_token via CausalRuntime.register_future(...) so
agent_loop_step can suspend on future_done(token) instead of blocking a
worker on LLM RTT.
"""
from __future__ import annotations

import concurrent.futures
import sys
import threading
import time
import unittest
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from high_agent.llm.client import ModelClient, ModelClientError
from high_agent.llm.providers import ModelSettings


class _FakeStreamResponse:
    def __init__(self, lines: list[str], *, status_code: int = 200,
                 content_type: str = "text/event-stream", delay: float = 0.0) -> None:
        self.lines = lines
        self.status_code = status_code
        self.headers = {"content-type": content_type}
        self.text = ""
        self.delay = delay

    def __enter__(self) -> "_FakeStreamResponse":
        if self.delay:
            time.sleep(self.delay)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def iter_lines(self):
        return iter(self.lines)

    def read(self) -> bytes:
        return "\n".join(self.lines).encode("utf-8")


class _FakeStreamHttpClient:
    def __init__(self, lines: list[str], *, status_code: int = 200,
                 content_type: str = "text/event-stream", delay: float = 0.0) -> None:
        self.lines = lines
        self.status_code = status_code
        self.content_type = content_type
        self.delay = delay
        self.stream_requests: list[dict] = []
        self.thread_names: list[str] = []

    def stream(self, method: str, url: str, headers: dict, json: dict) -> _FakeStreamResponse:
        self.stream_requests.append({"method": method, "url": url, "headers": headers, "json": json})
        self.thread_names.append(threading.current_thread().name)
        return _FakeStreamResponse(
            self.lines, status_code=self.status_code,
            content_type=self.content_type, delay=self.delay,
        )


def _ok_chat_lines() -> list[str]:
    """Minimal valid Chat Completions stream: one tool_call closes JSON, finish_reason='tool_calls'."""
    return [
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call-1","function":{"name":"noop","arguments":"{\\"x\\":1}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}\n',
        "\n",
        "data: [DONE]\n",
        "\n",
    ]


def _settings() -> ModelSettings:
    return ModelSettings("custom", "m", "https://example.test/v1", "chat_completions", "key")


class CompleteAsyncReturnsFutureTests(unittest.TestCase):
    def test_complete_async_returns_concurrent_future_resolving_with_response(self) -> None:
        http = _FakeStreamHttpClient(_ok_chat_lines())
        client = ModelClient(_settings(), http_client=http)
        fut = client.complete_async([{"role": "user", "content": "hi"}])
        self.assertIsInstance(fut, concurrent.futures.Future)
        response = fut.result(timeout=5.0)
        self.assertEqual(response.tool_calls[0].id, "call-1")
        self.assertEqual(response.tool_calls[0].name, "noop")
        self.assertEqual(response.usage.total_tokens, 4)

    def test_complete_async_runs_off_caller_thread(self) -> None:
        http = _FakeStreamHttpClient(_ok_chat_lines(), delay=0.05)
        client = ModelClient(_settings(), http_client=http)
        caller_thread = threading.current_thread().name
        fut = client.complete_async([{"role": "user", "content": "hi"}])
        # Future is not done yet — caller thread is free.
        self.assertFalse(fut.done())
        response = fut.result(timeout=5.0)
        self.assertEqual(response.tool_calls[0].id, "call-1")
        # The HTTP work executed on an IO worker thread, not the caller thread.
        self.assertTrue(http.thread_names)
        self.assertNotIn(caller_thread, http.thread_names)
        self.assertTrue(http.thread_names[0].startswith("high-agent-io-worker"))

    def test_complete_async_propagates_errors_to_future(self) -> None:
        # Empty stream lines → normalize_stream_events raises.
        http = _FakeStreamHttpClient(["data: {\"error\":{\"message\":\"boom\"}}\n", "\n"])
        client = ModelClient(_settings(), http_client=http)
        fut = client.complete_async([{"role": "user", "content": "hi"}])
        with self.assertRaises(ModelClientError):
            fut.result(timeout=5.0)

    def test_sync_complete_unaffected(self) -> None:
        """The sync complete() entry point keeps its existing semantics."""
        http = _FakeStreamHttpClient(_ok_chat_lines())
        response = ModelClient(_settings(), http_client=http).complete(
            [{"role": "user", "content": "hi"}]
        )
        self.assertEqual(response.tool_calls[0].id, "call-1")


class CompleteStreamingAsyncIncrementalDispatchTests(unittest.TestCase):
    """feat-S1 incremental on_tool_call dispatch is preserved in the async path."""

    def test_streaming_async_invokes_on_tool_call_before_future_resolves(self) -> None:
        # JSON-close on the first chunk should fire on_tool_call immediately.
        lines = [
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call-1","function":{"name":"ls","arguments":"{}"}}]}}]}\n',
            "\n",
            'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}\n',
            "\n",
            "data: [DONE]\n",
            "\n",
        ]
        http = _FakeStreamHttpClient(lines)
        client = ModelClient(_settings(), http_client=http)
        emitted: list[Any] = []

        def on_call(call: Any) -> None:
            emitted.append(call)

        fut = client.complete_streaming_async(
            [{"role": "user", "content": "hi"}], None, on_call,
        )
        response = fut.result(timeout=5.0)
        # Single emit, no double-dispatch from the finish_reason fallback.
        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0].id, "call-1")
        self.assertEqual(emitted[0].name, "ls")
        self.assertEqual(emitted[0].arguments, "{}")
        # The normalized response also carries the tool_call (transport-level).
        self.assertEqual(response.tool_calls[0].id, "call-1")


class IOLoopLifecycleTests(unittest.TestCase):
    """The IO loop thread is daemonized and shared across ModelClient instances."""

    def test_io_loop_thread_is_daemon(self) -> None:
        http = _FakeStreamHttpClient(_ok_chat_lines())
        ModelClient(_settings(), http_client=http).complete_async(
            [{"role": "user", "content": "hi"}]
        ).result(timeout=5.0)
        from high_agent.llm import client as client_module

        loop_thread = client_module._IO_LOOP_THREAD
        self.assertIsNotNone(loop_thread)
        self.assertTrue(loop_thread.daemon)

    def test_executor_shared_across_instances(self) -> None:
        from high_agent.llm import client as client_module

        http_a = _FakeStreamHttpClient(_ok_chat_lines())
        http_b = _FakeStreamHttpClient(_ok_chat_lines())
        ModelClient(_settings(), http_client=http_a).complete_async(
            [{"role": "user", "content": "a"}]
        ).result(timeout=5.0)
        executor_a = client_module._ASYNC_EXECUTOR
        ModelClient(_settings(), http_client=http_b).complete_async(
            [{"role": "user", "content": "b"}]
        ).result(timeout=5.0)
        executor_b = client_module._ASYNC_EXECUTOR
        self.assertIs(executor_a, executor_b)


if __name__ == "__main__":
    unittest.main()
