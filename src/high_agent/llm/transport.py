"""Provider transport adapters.

The runtime core only sees NormalizedResponse and ToolCall. HTTP details stay
inside these adapters.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from high_agent.llm.providers import ModelSettings
from high_agent.llm.types import NormalizedResponse, ToolCall, Usage


@dataclass(frozen=True)
class HttpRequest:
    url: str
    headers: dict[str, str]
    payload: dict[str, Any]


class ModelTransport(ABC):
    @abstractmethod
    def build_request(self, *, model: str, messages: list[dict[str, Any]],
                      tools: list[dict[str, Any]] | None = None,
                      **params: Any) -> dict[str, Any]:
        ...

    @abstractmethod
    def normalize_response(self, response: Any) -> NormalizedResponse:
        ...


class EchoTransport(ModelTransport):
    """Test transport that returns supplied response payloads unchanged."""

    def build_request(self, *, model: str, messages: list[dict[str, Any]],
                      tools: list[dict[str, Any]] | None = None,
                      **params: Any) -> dict[str, Any]:
        return {"model": model, "messages": messages, "tools": tools or [], **params}

    def normalize_response(self, response: Any) -> NormalizedResponse:
        if isinstance(response, NormalizedResponse):
            return response
        raise TypeError("EchoTransport expects a NormalizedResponse")


class HttpModelTransport(ModelTransport):
    def __init__(self, settings: ModelSettings) -> None:
        self.settings = settings

    @abstractmethod
    def build_http_request(self, *, model: str, messages: list[dict[str, Any]],
                           tools: list[dict[str, Any]] | None = None,
                           **params: Any) -> HttpRequest:
        ...

    def build_stream_request(self, *, model: str, messages: list[dict[str, Any]],
                             tools: list[dict[str, Any]] | None = None,
                             **params: Any) -> dict[str, Any]:
        payload = self.build_request(model=model, messages=messages, tools=tools, **params)
        payload["stream"] = True
        return payload

    def build_stream_http_request(self, *, model: str, messages: list[dict[str, Any]],
                                  tools: list[dict[str, Any]] | None = None,
                                  **params: Any) -> HttpRequest:
        request = self.build_http_request(model=model, messages=messages, tools=tools, **params)
        payload = dict(request.payload)
        payload["stream"] = True
        return HttpRequest(url=request.url, headers=request.headers, payload=payload)

    def normalize_stream_events(self, events: list[dict[str, Any]]) -> NormalizedResponse:
        if len(events) == 1 and events[0].get("event") == "json_response":
            return self.normalize_response(events[0].get("data") or {})
        raise NotImplementedError(f"{type(self).__name__} does not support streaming")

    def _auth_headers(self) -> dict[str, str]:
        headers = {"content-type": "application/json", **self.settings.extra_headers}
        if self.settings.api_key:
            headers["authorization"] = f"Bearer {self.settings.api_key}"
        return headers


class OpenAIChatCompletionsTransport(HttpModelTransport):
    def build_request(self, *, model: str, messages: list[dict[str, Any]],
                      tools: list[dict[str, Any]] | None = None,
                      **params: Any) -> dict[str, Any]:
        payload = {"model": model, "messages": _openai_messages(messages), **params}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        return payload

    def build_http_request(self, *, model: str, messages: list[dict[str, Any]],
                           tools: list[dict[str, Any]] | None = None,
                           **params: Any) -> HttpRequest:
        return HttpRequest(
            url=_join_endpoint(self.settings.base_url, "chat/completions"),
            headers=self._auth_headers(),
            payload=self.build_request(model=model, messages=messages, tools=tools, **params),
        )

    def normalize_response(self, response: Any) -> NormalizedResponse:
        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        calls = [
            ToolCall(
                id=call.get("id"),
                name=(call.get("function") or {}).get("name", ""),
                arguments=(call.get("function") or {}).get("arguments", "{}"),
                provider_data=call,
            )
            for call in message.get("tool_calls") or []
        ]
        return NormalizedResponse(
            content=message.get("content"),
            tool_calls=calls or None,
            finish_reason=choice.get("finish_reason") or "",
            usage=_usage(response.get("usage")),
            provider_data=response,
        )

    def normalize_stream_events(self, events: list[dict[str, Any]]) -> NormalizedResponse:
        if len(events) == 1 and events[0].get("event") == "json_response":
            return self.normalize_response(events[0].get("data") or {})
        text_parts: list[str] = []
        tool_parts: dict[int, dict[str, str]] = {}
        finish_reason = ""
        usage: Usage | None = None
        chunks: list[dict[str, Any]] = []
        for event in events:
            data = event.get("data") or {}
            if not isinstance(data, dict):
                continue
            chunks.append(data)
            if isinstance(data.get("usage"), dict):
                usage = _usage(data.get("usage"))
            choices = data.get("choices") or []
            if not choices:
                continue
            choice = choices[0] or {}
            if choice.get("finish_reason"):
                finish_reason = str(choice.get("finish_reason") or "")
            delta = choice.get("delta") or choice.get("message") or {}
            content = delta.get("content")
            if isinstance(content, str):
                text_parts.append(content)
            for call in delta.get("tool_calls") or []:
                index = int(call.get("index") or 0)
                record = tool_parts.setdefault(index, {"id": "", "name": "", "arguments": ""})
                if call.get("id"):
                    record["id"] = str(call.get("id"))
                function = call.get("function") or {}
                if function.get("name"):
                    record["name"] = str(function.get("name"))
                if function.get("arguments") is not None:
                    record["arguments"] += str(function.get("arguments") or "")
        calls = [
            ToolCall(
                id=record["id"] or f"call-{index}",
                name=record["name"],
                arguments=record["arguments"] or "{}",
                provider_data={"index": index, **record},
            )
            for index, record in sorted(tool_parts.items())
        ]
        return NormalizedResponse(
            content="".join(text_parts) or None,
            tool_calls=calls or None,
            finish_reason=finish_reason,
            usage=usage,
            provider_data={"stream": chunks},
        )


class AnthropicMessagesTransport(HttpModelTransport):
    def _auth_headers(self) -> dict[str, str]:
        headers = {"content-type": "application/json", "anthropic-version": "2023-06-01", **self.settings.extra_headers}
        if self.settings.api_key:
            headers["x-api-key"] = self.settings.api_key
        return headers

    def build_request(self, *, model: str, messages: list[dict[str, Any]],
                      tools: list[dict[str, Any]] | None = None,
                      **params: Any) -> dict[str, Any]:
        system, converted = _anthropic_messages(messages)
        payload = {
            "model": model,
            "messages": converted,
            "max_tokens": int(params.pop("max_tokens", 4096)),
            **params,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = [_anthropic_tool(tool) for tool in tools]
        return payload

    def build_http_request(self, *, model: str, messages: list[dict[str, Any]],
                           tools: list[dict[str, Any]] | None = None,
                           **params: Any) -> HttpRequest:
        return HttpRequest(
            url=_join_endpoint(self.settings.base_url, "messages"),
            headers=self._auth_headers(),
            payload=self.build_request(model=model, messages=messages, tools=tools, **params),
        )

    def normalize_response(self, response: Any) -> NormalizedResponse:
        content = response.get("content") or []
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in content:
            if block.get("type") == "text":
                text_parts.append(str(block.get("text") or ""))
            if block.get("type") == "tool_use":
                calls.append(
                    ToolCall(
                        id=block.get("id"),
                        name=block.get("name", ""),
                        arguments=json.dumps(block.get("input") or {}, ensure_ascii=False),
                        provider_data=block,
                    )
                )
        usage_raw = response.get("usage") or {}
        usage = Usage(
            prompt_tokens=int(usage_raw.get("input_tokens") or 0),
            completion_tokens=int(usage_raw.get("output_tokens") or 0),
            total_tokens=int(usage_raw.get("input_tokens") or 0) + int(usage_raw.get("output_tokens") or 0),
        )
        return NormalizedResponse(
            content="\n".join(part for part in text_parts if part) or None,
            tool_calls=calls or None,
            finish_reason=response.get("stop_reason") or "",
            usage=usage,
            provider_data=response,
        )

    def normalize_stream_events(self, events: list[dict[str, Any]]) -> NormalizedResponse:
        if len(events) == 1 and events[0].get("event") == "json_response":
            return self.normalize_response(events[0].get("data") or {})
        text_parts: list[str] = []
        tool_parts: dict[int, dict[str, str]] = {}
        input_tokens = 0
        output_tokens = 0
        finish_reason = ""
        provider_events: list[dict[str, Any]] = []
        for event in events:
            data = event.get("data") or {}
            if not isinstance(data, dict):
                continue
            provider_events.append(data)
            event_type = str(data.get("type") or event.get("event") or "")
            if event_type == "message_start":
                usage_raw = ((data.get("message") or {}).get("usage") or {})
                input_tokens = int(usage_raw.get("input_tokens") or input_tokens or 0)
                output_tokens = int(usage_raw.get("output_tokens") or output_tokens or 0)
            elif event_type == "content_block_start":
                index = int(data.get("index") or 0)
                block = data.get("content_block") or {}
                if block.get("type") == "text" and block.get("text"):
                    text_parts.append(str(block.get("text") or ""))
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
                if delta.get("type") == "text_delta":
                    text_parts.append(str(delta.get("text") or ""))
                if delta.get("type") == "input_json_delta":
                    record = tool_parts.setdefault(index, {"id": f"toolu-{index}", "name": "", "arguments": ""})
                    record["arguments"] += str(delta.get("partial_json") or "")
            elif event_type == "message_delta":
                delta = data.get("delta") or {}
                if delta.get("stop_reason"):
                    finish_reason = str(delta.get("stop_reason") or "")
                usage_raw = data.get("usage") or {}
                output_tokens = int(usage_raw.get("output_tokens") or output_tokens or 0)
        calls = [
            ToolCall(
                id=record["id"],
                name=record["name"],
                arguments=record["arguments"] or "{}",
                provider_data={"index": index, **record},
            )
            for index, record in sorted(tool_parts.items())
        ]
        return NormalizedResponse(
            content="".join(text_parts) or None,
            tool_calls=calls or None,
            finish_reason=finish_reason,
            usage=Usage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            ),
            provider_data={"stream": provider_events},
        )


class OpenAIResponsesTransport(HttpModelTransport):
    def build_request(self, *, model: str, messages: list[dict[str, Any]],
                      tools: list[dict[str, Any]] | None = None,
                      **params: Any) -> dict[str, Any]:
        payload = {"model": model, "input": _responses_input(messages), **params}
        if tools:
            payload["tools"] = [_responses_tool(tool) for tool in tools]
        return payload

    def build_http_request(self, *, model: str, messages: list[dict[str, Any]],
                           tools: list[dict[str, Any]] | None = None,
                           **params: Any) -> HttpRequest:
        return HttpRequest(
            url=_join_endpoint(self.settings.base_url, "responses"),
            headers=self._auth_headers(),
            payload=self.build_request(model=model, messages=messages, tools=tools, **params),
        )

    def normalize_response(self, response: Any) -> NormalizedResponse:
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for item in response.get("output") or []:
            item_type = item.get("type")
            if item_type == "message":
                for block in item.get("content") or []:
                    if block.get("type") in {"output_text", "text"}:
                        text_parts.append(str(block.get("text") or ""))
            if item_type in {"function_call", "tool_call"}:
                calls.append(
                    ToolCall(
                        id=item.get("call_id") or item.get("id"),
                        name=item.get("name", ""),
                        arguments=item.get("arguments") or "{}",
                        provider_data=item,
                    )
                )
        return NormalizedResponse(
            content="\n".join(part for part in text_parts if part) or response.get("output_text"),
            tool_calls=calls or None,
            finish_reason=response.get("status") or "",
            usage=_usage(response.get("usage")),
            provider_data=response,
        )

    def normalize_stream_events(self, events: list[dict[str, Any]]) -> NormalizedResponse:
        if len(events) == 1 and events[0].get("event") == "json_response":
            return self.normalize_response(events[0].get("data") or {})
        text_parts: list[str] = []
        call_parts: dict[str, dict[str, str]] = {}
        output_index_to_key: dict[int, str] = {}
        finish_reason = ""
        usage: Usage | None = None
        provider_events: list[dict[str, Any]] = []
        for event in events:
            data = event.get("data") or {}
            if not isinstance(data, dict):
                continue
            provider_events.append(data)
            event_type = str(data.get("type") or event.get("event") or "")
            completed = data.get("response")
            if event_type == "response.completed" and isinstance(completed, dict) and completed.get("output"):
                return self.normalize_response(completed)
            if event_type == "response.completed" and isinstance(completed, dict):
                finish_reason = str(completed.get("status") or finish_reason)
                usage = _usage(completed.get("usage")) or usage
            if event_type == "response.output_text.delta":
                text_parts.append(str(data.get("delta") or ""))
            if event_type == "response.output_item.added":
                item = data.get("item") or {}
                if item.get("type") in {"function_call", "tool_call"}:
                    key = _responses_call_key(item, data)
                    index = data.get("output_index")
                    # if `response.function_call_arguments.delta`
                    # arrived before `output_item.added` (SSE reorder /
                    # buffered fragments), it accumulated into the
                    # placeholder key `output-{index}`. When the canonical
                    # key shows up here we must MERGE the placeholder into
                    # the canonical entry, otherwise `sorted(call_parts)`
                    # later emits two ToolCalls — one with name/id but no
                    # args, one with args but no name — for the same logical
                    # call. Same migration applies to output_index_to_key
                    # so subsequent deltas land in the canonical bucket.
                    placeholder_key: str | None = None
                    if index is not None:
                        placeholder_key = output_index_to_key.get(int(index))
                        if placeholder_key is None:
                            placeholder_key = f"output-{int(index)}"
                        output_index_to_key[int(index)] = key
                    accumulated_args = ""
                    if placeholder_key is not None and placeholder_key != key:
                        prior = call_parts.pop(placeholder_key, None)
                        if prior is not None:
                            accumulated_args = prior.get("arguments") or ""
                    existing = call_parts.get(key)
                    arguments_from_added = str(item.get("arguments") or "")
                    # Prefer the longer arguments string: prior accumulation
                    # under the placeholder may already hold more than the
                    # `added` event's snapshot (which often is empty).
                    merged_arguments = arguments_from_added
                    if len(accumulated_args) > len(merged_arguments):
                        merged_arguments = accumulated_args
                    if existing and len(existing.get("arguments") or "") > len(merged_arguments):
                        merged_arguments = existing["arguments"]
                    call_parts[key] = {
                        "id": str(item.get("call_id") or item.get("id") or key),
                        "name": str(item.get("name") or (existing.get("name") if existing else "") or ""),
                        "arguments": merged_arguments,
                    }
            if event_type == "response.function_call_arguments.delta":
                key = _responses_delta_key(data, output_index_to_key)
                record = call_parts.setdefault(key, {"id": key, "name": "", "arguments": ""})
                record["arguments"] += str(data.get("delta") or "")
            if event_type in {"response.function_call_arguments.done", "response.output_item.done"}:
                item = data.get("item") or {}
                if event_type == "response.function_call_arguments.done":
                    key = _responses_delta_key(data, output_index_to_key)
                    record = call_parts.setdefault(key, {"id": key, "name": "", "arguments": ""})
                    if data.get("arguments") is not None:
                        record["arguments"] = str(data.get("arguments") or "")
                elif item.get("type") in {"function_call", "tool_call"}:
                    key = _responses_call_key(item, data)
                    call_parts[key] = {
                        "id": str(item.get("call_id") or item.get("id") or key),
                        "name": str(item.get("name") or ""),
                        "arguments": str(item.get("arguments") or call_parts.get(key, {}).get("arguments") or "{}"),
                    }
        calls = [
            ToolCall(id=record["id"], name=record["name"], arguments=record["arguments"] or "{}", provider_data=record)
            for _, record in sorted(call_parts.items())
        ]
        return NormalizedResponse(
            content="".join(text_parts) or None,
            tool_calls=calls or None,
            finish_reason=finish_reason,
            usage=usage,
            provider_data={"stream": provider_events},
        )


def create_transport(settings: ModelSettings) -> HttpModelTransport:
    if settings.api_mode == "chat_completions":
        return OpenAIChatCompletionsTransport(settings)
    if settings.api_mode == "anthropic_messages":
        return AnthropicMessagesTransport(settings)
    if settings.api_mode == "codex_responses":
        return OpenAIResponsesTransport(settings)
    raise ValueError(f"unsupported api_mode: {settings.api_mode}")


def _join_endpoint(base_url: str, endpoint: str) -> str:
    base = (base_url or "").rstrip("/")
    if base.endswith(f"/{endpoint}"):
        return base
    if endpoint == "messages" and base.endswith("/v1"):
        return f"{base}/messages"
    if endpoint == "messages" and not base.endswith("/v1") and "anthropic" in base:
        return f"{base}/v1/messages"
    # If base_url already carries an explicit version segment (/v1, /v2,
    # /v3, /api/v1, ...) — i.e. anything beyond the bare host — append the
    # endpoint directly. The "/v1 fallback" below is only for hosts that
    # supplied just a hostname (e.g. https://api.openai.com).
    from urllib.parse import urlparse
    path = urlparse(base).path.strip("/")
    if path:
        return f"{base}/{endpoint}"
    return f"{base}/v1/{endpoint}"


def _usage(raw: Any) -> Usage | None:
    if not isinstance(raw, dict):
        return None
    prompt = int(raw.get("prompt_tokens") or raw.get("input_tokens") or 0)
    completion = int(raw.get("completion_tokens") or raw.get("output_tokens") or 0)
    return Usage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=int(raw.get("total_tokens") or prompt + completion),
    )


def _openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        item = {"role": role, "content": message.get("content") or ""}
        if role == "assistant" and message.get("tool_calls"):
            item["tool_calls"] = message["tool_calls"]
        if role == "tool":
            item["tool_call_id"] = message.get("tool_call_id")
        out.append(item)
    return out


def _anthropic_messages(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    system_parts: list[str] = []
    out: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role == "system":
            system_parts.append(str(message.get("content") or ""))
        elif role == "assistant":
            content: list[dict[str, Any]] = []
            if message.get("content"):
                content.append({"type": "text", "text": str(message["content"])})
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                try:
                    args = json.loads(function.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {"raw": function.get("arguments") or ""}
                content.append({"type": "tool_use", "id": call.get("id"), "name": function.get("name"), "input": args})
            out.append({"role": "assistant", "content": content or [{"type": "text", "text": ""}]})
        elif role == "tool":
            out.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": message.get("tool_call_id"),
                            "content": str(message.get("content") or ""),
                        }
                    ],
                }
            )
        else:
            out.append({"role": "user", "content": str(message.get("content") or "")})
    return "\n\n".join(part for part in system_parts if part), out


def _anthropic_tool(tool: dict[str, Any]) -> dict[str, Any]:
    function = tool.get("function") or tool
    return {
        "name": function["name"],
        "description": function.get("description", ""),
        "input_schema": function.get("parameters") or {"type": "object", "properties": {}},
    }


def _responses_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role == "tool":
            out.append(
                {
                    "type": "function_call_output",
                    "call_id": message.get("tool_call_id"),
                    "output": str(message.get("content") or ""),
                }
            )
        elif role == "assistant" and message.get("tool_calls"):
            if message.get("content"):
                out.append({"role": "assistant", "content": str(message.get("content") or "")})
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                out.append(
                    {
                        "type": "function_call",
                        "call_id": call.get("id"),
                        "name": function.get("name", ""),
                        "arguments": function.get("arguments") or "{}",
                    }
                )
        else:
            out.append({"role": role, "content": str(message.get("content") or "")})
    return out


def _responses_tool(tool: dict[str, Any]) -> dict[str, Any]:
    function = tool.get("function") or tool
    return {
        "type": "function",
        "name": function["name"],
        "description": function.get("description", ""),
        "parameters": function.get("parameters") or {"type": "object", "properties": {}},
    }


def _responses_call_key(item: dict[str, Any], data: dict[str, Any]) -> str:
    if item.get("id"):
        return str(item.get("id"))
    if item.get("call_id"):
        return str(item.get("call_id"))
    if data.get("output_index") is not None:
        return f"output-{int(data.get('output_index'))}"
    return "output-0"


def _responses_delta_key(data: dict[str, Any], output_index_to_key: dict[int, str]) -> str:
    if data.get("item_id"):
        return str(data.get("item_id"))
    if data.get("call_id"):
        return str(data.get("call_id"))
    if data.get("output_index") is not None:
        index = int(data.get("output_index"))
        return output_index_to_key.get(index, f"output-{index}")
    return "output-0"
