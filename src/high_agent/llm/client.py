"""HTTP model client."""

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import json
import threading
from dataclasses import dataclass, field
from typing import Any

try:
    import httpx
except ModuleNotFoundError:  # pragma: no cover - minimal env guidance path
    httpx = None

try:
    from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential, wait_fixed
except ModuleNotFoundError:  # pragma: no cover - minimal env guidance path
    retry = None
    retry_if_exception_type = None
    stop_after_attempt = None
    wait_exponential = None
    wait_fixed = None

from high_agent.llm.providers import ModelSettings
from high_agent.llm.transport import (
    HttpModelTransport,
    _responses_call_key,
    _responses_delta_key,
    create_transport,
)
from high_agent.llm.types import NormalizedResponse


# IO loop singleton for non-blocking HTTP dispatch (v11-C6).
#
# The IO loop runs on its own daemon thread and exposes a thread pool executor
# that the actual blocking HTTP work runs on. The contract that matters to the
# runtime is "complete_async returns a concurrent.futures.Future that resolves
# off the caller thread" — the runtime can register the future with
# CausalRuntime.register_future(...) and suspend the agent_loop_step task until
# future_done(token) fires.
#
# The loop thread itself is NOT a noGIL worker thread; ensure_nogil(strict=True)
# is only checked at CausalRuntime construction time, so introducing an asyncio
# loop here does not violate the noGIL invariant. The loop is started lazily on
# first complete_async call, and torn down via atexit. See
# for the contract.
_io_loop_lock = threading.Lock()
_IO_LOOP: asyncio.AbstractEventLoop | None = None
_IO_LOOP_THREAD: threading.Thread | None = None
_ASYNC_EXECUTOR: concurrent.futures.ThreadPoolExecutor | None = None


def _ensure_io_loop() -> asyncio.AbstractEventLoop:
    global _IO_LOOP, _IO_LOOP_THREAD, _ASYNC_EXECUTOR
    loop = _IO_LOOP
    if loop is not None and loop.is_running():
        return loop
    with _io_loop_lock:
        loop = _IO_LOOP
        if loop is not None and loop.is_running():
            return loop
        loop = asyncio.new_event_loop()
        thread = threading.Thread(
            target=loop.run_forever,
            name="high-agent-io-loop",
            daemon=True,
        )
        thread.start()
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=16,
            thread_name_prefix="high-agent-io-worker",
        )
        _IO_LOOP = loop
        _IO_LOOP_THREAD = thread
        _ASYNC_EXECUTOR = executor
        atexit.register(_shutdown_io_loop)
        return loop


def _shutdown_io_loop() -> None:
    global _IO_LOOP, _IO_LOOP_THREAD, _ASYNC_EXECUTOR
    executor = _ASYNC_EXECUTOR
    loop = _IO_LOOP
    _ASYNC_EXECUTOR = None
    _IO_LOOP = None
    _IO_LOOP_THREAD = None
    if executor is not None:
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except Exception:  # pragma: no cover - best-effort teardown
            pass
    if loop is not None and loop.is_running():
        try:
            loop.call_soon_threadsafe(loop.stop)
        except Exception:  # pragma: no cover - best-effort teardown
            pass


class ModelClientError(RuntimeError):
    pass


@dataclass
class ModelClient:
    settings: ModelSettings
    http_client: Any = None
    timeout: float = 600.0
    transport: HttpModelTransport | None = None
    _owns_client: bool = False
    # round-robin counter for multi-key pools (settings.api_keys)
    _key_idx: int = field(default=0, init=False)
    _key_lock: Any = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.transport is None:
            self.transport = create_transport(self.settings)
        if self.http_client is None and httpx is not None:
            self.http_client = httpx.Client(timeout=self.timeout)
            self._owns_client = True
        # threading.Lock for round-robin key rotation across concurrent planners
        import threading as _threading
        self._key_lock = _threading.Lock()

    def _next_api_key(self) -> str:
        """Pick the next api_key from the pool (round-robin), or fall back
        to settings.api_key if no pool is configured."""
        keys = self.settings.api_keys
        if not keys:
            return self.settings.api_key
        if self._key_lock is None:
            return keys[0]
        with self._key_lock:
            k = keys[self._key_idx % len(keys)]
            self._key_idx += 1
            return k

    def _override_auth_header(self, headers: dict[str, str]) -> dict[str, str]:
        """Replace the Authorization (Bearer) header with the next pool key.

        Transport.build_stream_http_request() bakes settings.api_key into
        the headers; for multi-key pools we swap it per-request so each
        concurrent planner uses a different key, side-stepping single-key
        provider rate limits / EngineCore bottlenecks observed under
        max_planner_requests > 1.
        """
        if not self.settings.api_keys:
            return headers
        out = dict(headers)
        key = self._next_api_key()
        out["authorization"] = f"Bearer {key}"
        return out

    def close(self) -> None:
        if self._owns_client and self.http_client is not None:
            try:
                self.http_client.close()
            except Exception:
                pass
            self.http_client = None
            self._owns_client = False

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
                 **params: Any) -> NormalizedResponse:
        if httpx is None and self.http_client is None:
            raise ModelClientError("httpx is required for real model calls; install project dependencies first")
        if not self.settings.model:
            raise ModelClientError("model is not configured")
        request = self.transport.build_stream_http_request(
            model=self.settings.model,
            messages=messages,
            tools=tools,
            **params,
        )
        headers = self._override_auth_header(request.headers)
        events = self._post_stream(request.url, headers, request.payload)
        return self.transport.normalize_stream_events(events)

    def complete_async(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **params: Any,
    ) -> concurrent.futures.Future[NormalizedResponse]:
        """Schedule complete() on the IO loop thread pool, return a Future.

        Used by ``agent_loop_step`` (v11-C7) to suspend on ``future_done(token)``
        instead of blocking a worker thread for the LLM RTT. The future
        resolves with the same NormalizedResponse that ``complete()`` would
        have returned synchronously, so callers can pass it through
        ``runtime.register_future(...)`` and continue once the suspend wakes.
        """
        _ensure_io_loop()
        executor = _ASYNC_EXECUTOR
        if executor is None:  # pragma: no cover - shutdown race
            raise ModelClientError("ModelClient async executor is not running")
        return executor.submit(self.complete, messages, tools, **params)

    def complete_streaming_async(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        on_tool_call: Any = None,
        **params: Any,
    ) -> concurrent.futures.Future[NormalizedResponse]:
        """Async variant of complete_streaming; preserves feat-S1 incremental dispatch.

        ``on_tool_call`` fires from the IO worker thread as each tool call's
        JSON arguments close (Anthropic content_block_stop, OpenAI Chat
        finish_reason / JSON-close, OpenAI Responses arguments.delta JSON-close).
        Callbacks must be thread-safe — they typically post to a runtime queue
        (e.g. via ``CausalRuntime.submit_threadsafe`` indirection) rather than
        mutating shared state directly.
        """
        _ensure_io_loop()
        executor = _ASYNC_EXECUTOR
        if executor is None:  # pragma: no cover - shutdown race
            raise ModelClientError("ModelClient async executor is not running")
        return executor.submit(
            self.complete_streaming, messages, tools, on_tool_call, **params
        )

    def complete_streaming(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        on_tool_call: Any = None,
        **params: Any,
    ) -> NormalizedResponse:
        """Like complete(), but fires on_tool_call(ToolCall) as each call completes in the stream."""
        if httpx is None and self.http_client is None:
            raise ModelClientError("httpx is required for real model calls; install project dependencies first")
        if not self.settings.model:
            raise ModelClientError("model is not configured")
        request = self.transport.build_stream_http_request(
            model=self.settings.model,
            messages=messages,
            tools=tools,
            **params,
        )
        headers = self._override_auth_header(request.headers)
        events = self._post_stream_incremental(request.url, headers, request.payload, on_tool_call)
        return self.transport.normalize_stream_events(events)

    def _post_stream_incremental(
        self, url: str, headers: dict[str, str], payload: dict[str, Any], on_tool_call: Any
    ) -> list[dict[str, Any]]:
        from high_agent.llm.types import ToolCall as TC
        if self.http_client is None:
            raise ModelClientError("httpx is required")
        client = self.http_client
        if not hasattr(client, "stream"):
            return [{"event": "json_response", "data": self._post_json_once(url, headers, payload)}]
        return self._post_stream_incremental_retryable(url, headers, payload, on_tool_call, client, TC)

    def _post_stream_incremental_once(
        self, url: str, headers: dict[str, str], payload: dict[str, Any],
        on_tool_call: Any, client: Any, TC: Any,
    ) -> list[dict[str, Any]]:
        with client.stream("POST", url, headers=headers, json=payload) as response:
            if response.status_code >= 500:
                text = _read_response_text(response)
                if httpx is not None:
                    raise httpx.TransportError(f"model stream failed {response.status_code}: {text[:500]}")
                raise ModelClientError(f"model stream failed {response.status_code}: {text[:2000]}")
            if response.status_code >= 400:
                text = _read_response_text(response)
                raise ModelClientError(f"model stream failed {response.status_code}: {text[:2000]}")
            content_type = str(response.headers.get("content-type", "") if hasattr(response, "headers") else "")
            if "text/event-stream" not in content_type.lower():
                text = _read_response_text(response)
                try:
                    data = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ModelClientError("model stream response was not SSE or JSON") from exc
                return [{"event": "json_response", "data": data}]

            events: list[dict[str, Any]] = []
            tool_parts: dict[int, dict[str, str]] = {}
            emitted_indices: set[int] = set()
            responses_parts: dict[str, dict[str, str]] = {}
            responses_index_to_key: dict[int, str] = {}
            responses_emitted: set[str] = set()
            for raw_event in _iter_sse(response.iter_lines()):
                if raw_event.data == "[DONE]":
                    continue
                try:
                    data = json.loads(raw_event.data)
                except json.JSONDecodeError as exc:
                    raise ModelClientError(f"invalid model stream event: {raw_event.data[:200]}") from exc
                if isinstance(data, dict) and data.get("error"):
                    raise ModelClientError(f"model stream error: {str(data.get('error'))[:2000]}")
                events.append({"event": raw_event.event, "data": data})
                if on_tool_call and isinstance(data, dict):
                    self._check_tool_call_complete(
                        data, tool_parts, emitted_indices, on_tool_call, TC,
                        responses_parts=responses_parts,
                        responses_index_to_key=responses_index_to_key,
                        responses_emitted=responses_emitted,
                    )
            return events

    @staticmethod
    def _check_tool_call_complete(
        data: dict[str, Any], tool_parts: dict[int, dict[str, str]],
        emitted: set[int], on_tool_call: Any, TC: type,
        *,
        responses_parts: dict[str, dict[str, str]] | None = None,
        responses_index_to_key: dict[int, str] | None = None,
        responses_emitted: set[str] | None = None,
    ) -> None:
        event_type = str(data.get("type") or "")
        # Anthropic: content_block_start / content_block_delta / content_block_stop
        if event_type == "content_block_start":
            index = int(data.get("index") or 0)
            block = data.get("content_block") or {}
            if block.get("type") == "tool_use":
                initial_input = block.get("input")
                args = json.dumps(initial_input, ensure_ascii=False) if initial_input else ""
                tool_parts[index] = {
                    "id": str(block.get("id") or f"toolu-{index}"),
                    "name": str(block.get("name") or ""),
                    "arguments": args,
                }
        elif event_type == "content_block_delta":
            index = int(data.get("index") or 0)
            delta = data.get("delta") or {}
            if delta.get("type") == "input_json_delta" and index in tool_parts:
                tool_parts[index]["arguments"] += str(delta.get("partial_json") or "")
        elif event_type == "content_block_stop":
            index = int(data.get("index") or 0)
            if index in tool_parts and index not in emitted:
                emitted.add(index)
                rec = tool_parts[index]
                on_tool_call(TC(id=rec["id"], name=rec["name"], arguments=rec["arguments"] or "{}"))
        # OpenAI Responses: response.output_item.added / .arguments.delta / .arguments.done / .output_item.done
        # feat-S1: stream-level early dispatch for the Codex transport. Mirrors
        # the placeholder-key migration done by in transport.py so that
        # delta events arriving before output_item.added still resolve to the
        # canonical call_id once it shows up. Gate is the same JSON-closure
        # check used by OpenAI Chat: emit as soon as the accumulated arguments
        # parse, otherwise fall through to the fallback emit at .done events.
        if (
            responses_parts is not None
            and responses_index_to_key is not None
            and responses_emitted is not None
            and event_type.startswith("response.")
        ):
            ModelClient._handle_responses_event(
                data, event_type, responses_parts, responses_index_to_key,
                responses_emitted, on_tool_call, TC,
            )
        # OpenAI Chat: detect via choices[0].finish_reason or tool_calls delta
        choices = data.get("choices") or []
        if choices:
            choice = choices[0] or {}
            delta = choice.get("delta") or {}
            for call in delta.get("tool_calls") or []:
                idx = int(call.get("index") or 0)
                record = tool_parts.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                if call.get("id"):
                    record["id"] = str(call.get("id"))
                fn = call.get("function") or {}
                if fn.get("name"):
                    record["name"] = str(fn.get("name"))
                record["arguments"] += str(fn.get("arguments") or "")
                # Try emitting as soon as the arguments JSON closes; falls back
                # to the finish_reason batch path below if the JSON happens not
                # to be valid yet (e.g. mid-string chunk boundary).
                ModelClient._try_emit_openai_tool_call(idx, record, emitted, on_tool_call, TC)
            if choice.get("finish_reason") == "tool_calls":
                for idx, rec in sorted(tool_parts.items()):
                    if idx not in emitted:
                        emitted.add(idx)
                        on_tool_call(TC(id=rec["id"], name=rec["name"], arguments=rec["arguments"] or "{}"))

    @staticmethod
    def _handle_responses_event(
        data: dict[str, Any],
        event_type: str,
        responses_parts: dict[str, dict[str, str]],
        responses_index_to_key: dict[int, str],
        responses_emitted: set[str],
        on_tool_call: Any,
        TC: type,
    ) -> None:
        if event_type == "response.output_item.added":
            item = data.get("item") or {}
            if item.get("type") not in {"function_call", "tool_call"}:
                return
            key = _responses_call_key(item, data)
            index = data.get("output_index")
            placeholder_key: str | None = None
            if index is not None:
                placeholder_key = responses_index_to_key.get(int(index))
                if placeholder_key is None:
                    placeholder_key = f"output-{int(index)}"
                responses_index_to_key[int(index)] = key
            accumulated_args = ""
            if placeholder_key is not None and placeholder_key != key:
                prior = responses_parts.pop(placeholder_key, None)
                if prior is not None:
                    accumulated_args = prior.get("arguments") or ""
                # Emit dedupe ledger must follow the placeholder migration so
                # an early emit under the placeholder doesn't double-fire.
                if placeholder_key in responses_emitted:
                    responses_emitted.discard(placeholder_key)
                    responses_emitted.add(key)
            existing = responses_parts.get(key)
            arguments_from_added = str(item.get("arguments") or "")
            merged_arguments = arguments_from_added
            if len(accumulated_args) > len(merged_arguments):
                merged_arguments = accumulated_args
            if existing and len(existing.get("arguments") or "") > len(merged_arguments):
                merged_arguments = existing["arguments"]
            responses_parts[key] = {
                "id": str(item.get("call_id") or item.get("id") or key),
                "name": str(item.get("name") or (existing.get("name") if existing else "") or ""),
                "arguments": merged_arguments,
            }
            ModelClient._try_emit_responses_tool_call(
                key, responses_parts[key], responses_emitted, on_tool_call, TC,
            )
        elif event_type == "response.function_call_arguments.delta":
            key = _responses_delta_key(data, responses_index_to_key)
            record = responses_parts.setdefault(key, {"id": key, "name": "", "arguments": ""})
            record["arguments"] += str(data.get("delta") or "")
            ModelClient._try_emit_responses_tool_call(
                key, record, responses_emitted, on_tool_call, TC,
            )
        elif event_type == "response.function_call_arguments.done":
            key = _responses_delta_key(data, responses_index_to_key)
            record = responses_parts.setdefault(key, {"id": key, "name": "", "arguments": ""})
            if data.get("arguments") is not None and key not in responses_emitted:
                # Only overwrite when we haven't emitted yet — once emit has
                # happened, the dispatch contract says later refinements are
                # discarded so the runtime can't observe two different argument
                # payloads for the same tool_call_id.
                record["arguments"] = str(data.get("arguments") or "")
            ModelClient._try_emit_responses_tool_call(
                key, record, responses_emitted, on_tool_call, TC,
            )
        elif event_type == "response.output_item.done":
            item = data.get("item") or {}
            if item.get("type") not in {"function_call", "tool_call"}:
                return
            key = _responses_call_key(item, data)
            existing = responses_parts.get(key, {"id": key, "name": "", "arguments": ""})
            if key not in responses_emitted:
                responses_parts[key] = {
                    "id": str(item.get("call_id") or item.get("id") or key),
                    "name": str(item.get("name") or existing.get("name") or ""),
                    "arguments": str(item.get("arguments") or existing.get("arguments") or "{}"),
                }
                ModelClient._try_emit_responses_tool_call(
                    key, responses_parts[key], responses_emitted, on_tool_call, TC,
                    require_json_close=False,
                )

    @staticmethod
    def _try_emit_responses_tool_call(
        key: str, record: dict[str, str], emitted: set[str], on_tool_call: Any, TC: type,
        *, require_json_close: bool = True,
    ) -> None:
        """Emit Responses tool_call once arguments form valid JSON.

        Mirrors `_try_emit_openai_tool_call` but keys on the canonical call_id
        string rather than the integer choice index, because Responses uses
        opaque call_ids and the placeholder-key migration in
        """
        if key in emitted:
            return
        if not record.get("id") or not record.get("name"):
            return
        args = record.get("arguments") or ""
        if require_json_close:
            stripped = args.lstrip()
            if not stripped.startswith("{"):
                return
            try:
                json.loads(args)
            except json.JSONDecodeError:
                return
        emitted.add(key)
        on_tool_call(TC(id=record["id"], name=record["name"], arguments=args or "{}"))

    @staticmethod
    def _try_emit_openai_tool_call(
        idx: int, record: dict[str, str], emitted: set[int], on_tool_call: Any, TC: type,
    ) -> None:
        """Emit OpenAI tool_call as soon as accumulated arguments form valid JSON."""
        if idx in emitted:
            return
        if not record.get("id") or not record.get("name"):
            return
        args = record.get("arguments") or ""
        stripped = args.lstrip()
        if not stripped.startswith("{"):
            return
        try:
            json.loads(args)
        except json.JSONDecodeError:
            return
        emitted.add(idx)
        on_tool_call(TC(id=record["id"], name=record["name"], arguments=args))

    def _post_stream(self, url: str, headers: dict[str, str], payload: dict[str, Any]) -> list[dict[str, Any]]:
        try:
            return self._post_stream_retryable(url, headers, payload)
        except Exception as exc:
            if httpx is not None and isinstance(exc, httpx.TimeoutException):
                raise ModelClientError(
                    "model stream timed out after "
                    f"{self.timeout:g}s. For large project-build runs, increase "
                    "`--model-timeout`, `HIGH_AGENT_MODEL_TIMEOUT_SECONDS`, "
                    "or `model.timeout_seconds` in config.yaml."
                ) from exc
            if httpx is not None and isinstance(exc, httpx.TransportError):
                raise ModelClientError(f"model stream transport error: {exc}") from exc
            raise

    def _post_json(self, url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return self._post_json_retryable(url, headers, payload)
        except Exception as exc:
            if httpx is not None and isinstance(exc, httpx.TimeoutException):
                raise ModelClientError(
                    "model request timed out after "
                    f"{self.timeout:g}s. For large project-build runs, increase "
                    "`--model-timeout`, `HIGH_AGENT_MODEL_TIMEOUT_SECONDS`, "
                    "or `model.timeout_seconds` in config.yaml."
                ) from exc
            if httpx is not None and isinstance(exc, httpx.TransportError):
                raise ModelClientError(f"model transport error: {exc}") from exc
            raise

    if retry is not None and httpx is not None:
        @retry(
            stop=stop_after_attempt(5),
            wait=wait_fixed(10),
            retry=retry_if_exception_type((httpx.TimeoutException, httpx.TransportError)),
            reraise=True,
        )
        def _post_json_retryable(self, url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
            return self._post_json_once(url, headers, payload)
    else:
        def _post_json_retryable(self, url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
            return self._post_json_once(url, headers, payload)

    if retry is not None and httpx is not None:
        @retry(
            stop=stop_after_attempt(5),
            wait=wait_fixed(10),
            retry=retry_if_exception_type((httpx.TimeoutException, httpx.TransportError)),
            reraise=True,
        )
        def _post_stream_retryable(self, url: str, headers: dict[str, str], payload: dict[str, Any]) -> list[dict[str, Any]]:
            return self._post_stream_once(url, headers, payload)

        @retry(
            stop=stop_after_attempt(5),
            wait=wait_fixed(10),
            retry=retry_if_exception_type((httpx.TimeoutException, httpx.TransportError)),
            reraise=True,
        )
        def _post_stream_incremental_retryable(
            self, url: str, headers: dict[str, str], payload: dict[str, Any],
            on_tool_call: Any, client: Any, TC: Any,
        ) -> list[dict[str, Any]]:
            return self._post_stream_incremental_once(url, headers, payload, on_tool_call, client, TC)
    else:
        def _post_stream_retryable(self, url: str, headers: dict[str, str], payload: dict[str, Any]) -> list[dict[str, Any]]:
            return self._post_stream_once(url, headers, payload)

        def _post_stream_incremental_retryable(
            self, url: str, headers: dict[str, str], payload: dict[str, Any],
            on_tool_call: Any, client: Any, TC: Any,
        ) -> list[dict[str, Any]]:
            return self._post_stream_incremental_once(url, headers, payload, on_tool_call, client, TC)

    def _post_json_once(self, url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        if self.http_client is None:
            raise ModelClientError("httpx is required for real model calls; install project dependencies first")
        client = self.http_client
        response = client.post(url, headers=headers, json=payload)
        if response.status_code >= 500:
            if httpx is not None:
                raise httpx.TransportError(f"model request failed {response.status_code}: {response.text[:500]}")
            raise ModelClientError(f"model request failed {response.status_code}: {response.text[:2000]}")
        if response.status_code >= 400:
            raise ModelClientError(f"model request failed {response.status_code}: {response.text[:2000]}")
        data = response.json()
        if not isinstance(data, dict):
            raise ModelClientError("model response must be a JSON object")
        return data

    def _post_stream_once(self, url: str, headers: dict[str, str], payload: dict[str, Any]) -> list[dict[str, Any]]:
        if self.http_client is None:
            raise ModelClientError("httpx is required for real model calls; install project dependencies first")
        client = self.http_client
        if not hasattr(client, "stream"):
            return [{"event": "json_response", "data": self._post_json_once(url, headers, payload)}]
        with client.stream("POST", url, headers=headers, json=payload) as response:
            if response.status_code >= 500:
                text = _read_response_text(response)
                if httpx is not None:
                    raise httpx.TransportError(f"model stream failed {response.status_code}: {text[:500]}")
                raise ModelClientError(f"model stream failed {response.status_code}: {text[:2000]}")
            if response.status_code >= 400:
                text = _read_response_text(response)
                raise ModelClientError(f"model stream failed {response.status_code}: {text[:2000]}")
            content_type = str(response.headers.get("content-type", "") if hasattr(response, "headers") else "")
            if "text/event-stream" not in content_type.lower():
                text = _read_response_text(response)
                try:
                    data = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ModelClientError("model stream response was not SSE or JSON") from exc
                if not isinstance(data, dict):
                    raise ModelClientError("model response must be a JSON object")
                return [{"event": "json_response", "data": data}]

            events: list[dict[str, Any]] = []
            for raw_event in _iter_sse(response.iter_lines()):
                if raw_event.data == "[DONE]":
                    continue
                try:
                    data = json.loads(raw_event.data)
                except json.JSONDecodeError as exc:
                    raise ModelClientError(f"invalid model stream event: {raw_event.data[:200]}") from exc
                if isinstance(data, dict) and data.get("error"):
                    raise ModelClientError(f"model stream error: {str(data.get('error'))[:2000]}")
                events.append({"event": raw_event.event, "data": data})
            return events


@dataclass(frozen=True)
class _SSEEvent:
    event: str
    data: str


def _iter_sse(lines: Any) -> list[_SSEEvent]:
    events: list[_SSEEvent] = []
    event_name = "message"
    data_lines: list[str] = []

    def flush() -> None:
        nonlocal event_name, data_lines
        if data_lines:
            events.append(_SSEEvent(event=event_name, data="\n".join(data_lines)))
        event_name = "message"
        data_lines = []

    for raw_line in lines:
        line = raw_line.decode("utf-8", errors="replace") if isinstance(raw_line, bytes) else str(raw_line)
        line = line.rstrip("\r\n")
        if not line:
            flush()
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_name = line[6:].strip() or "message"
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
        else:
            data_lines.append(line)
    flush()
    return events


def _read_response_text(response: Any) -> str:
    try:
        body = response.read()
    except Exception:
        return str(getattr(response, "text", "") or "")
    if isinstance(body, bytes):
        return body.decode("utf-8", errors="replace")
    return str(body)
