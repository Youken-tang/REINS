"""Tool-call normalization for provider and Hermes/Codex wrappers."""

from __future__ import annotations

import json
import copy
import re
from dataclasses import dataclass
from typing import Any

from reins.llm_types import ToolCall
from reins.runtime.types import new_id


VALID_API_ROLES = frozenset({"system", "user", "assistant", "tool", "function", "developer"})
TOOL_ARGUMENTS_REPAIRED_MARKER = "[Tool call arguments were malformed and were sanitized before replay.]"


@dataclass(frozen=True)
class NormalizedToolCall:
    call: ToolCall
    original_name: str
    parent_call_id: str | None = None


class ToolCallNormalizer:
    """Normalize tool names and expand multi-tool wrapper calls."""

    def normalize(self, calls: list[ToolCall]) -> list[NormalizedToolCall]:
        normalized: list[NormalizedToolCall] = []
        for call in calls:
            call_id = call.id or new_id("call")
            call.id = call_id
            repair_current_tool_call(call)
            canonical_name = normalize_tool_name(call.name)
            if canonical_name == "multi_tool_use.parallel":
                normalized.extend(self._expand_parallel(call, call_id))
                continue
            if canonical_name == "write_many_files":
                expanded = self._expand_write_many_files(call, call_id)
                if expanded:
                    normalized.extend(expanded)
                    continue
            original = call.name
            call.name = canonical_name
            normalized.append(NormalizedToolCall(call=call, original_name=original))
        return normalized

    def _expand_parallel(self, call: ToolCall, call_id: str) -> list[NormalizedToolCall]:
        try:
            payload = call.args_dict()
        except Exception:
            return [NormalizedToolCall(call=call, original_name=call.name)]
        out: list[NormalizedToolCall] = []
        for index, item in enumerate(payload.get("tool_uses") or []):
            if not isinstance(item, dict):
                continue
            original = str(item.get("recipient_name") or item.get("name") or "")
            parameters = item.get("parameters") or item.get("arguments") or {}
            if not isinstance(parameters, dict):
                parameters = {"value": parameters}
            out.append(
                NormalizedToolCall(
                    call=ToolCall(
                        id=f"{call_id}-{index}",
                        name=normalize_tool_name(original),
                        arguments=json.dumps(parameters, ensure_ascii=False),
                        provider_data={"parent": call.provider_data, "wrapper": "multi_tool_use.parallel"},
                    ),
                    original_name=original,
                    parent_call_id=call_id,
                )
            )
        if out:
            return out
        return [NormalizedToolCall(call=call, original_name=call.name)]

    def _expand_write_many_files(self, call: ToolCall, call_id: str) -> list[NormalizedToolCall]:
        try:
            payload = call.args_dict()
        except Exception:
            return []
        files = payload.get("files")
        if not isinstance(files, list):
            return []
        out: list[NormalizedToolCall] = []
        for index, item in enumerate(files):
            if not isinstance(item, dict) or "path" not in item:
                return []
            parameters = {"path": item.get("path"), "content": item.get("content") or ""}
            out.append(
                NormalizedToolCall(
                    call=ToolCall(
                        id=f"{call_id}-{index}",
                        name="write_file",
                        arguments=json.dumps(parameters, ensure_ascii=False),
                        provider_data={"parent": call.provider_data, "lowered_from": "write_many_files"},
                    ),
                    original_name=call.name,
                    parent_call_id=call_id,
                )
            )
        return out


def repair_current_tool_call(call: ToolCall) -> bool:
    """Repair a freshly returned model tool call before runtime lowering.

    Hermes applies tool-call argument validation before execution as well as
    before API replay. high-agent keeps the same boundary: malformed current
    response arguments are fixed before `args_dict()` is asked to parse them,
    so the runtime receives either a valid object or a normal tool-lowering
    failure that can be delivered back to the model.
    """
    name = str(call.name or "?")
    arguments = call.arguments
    repaired = False
    if arguments is None or arguments == "":
        fixed = "{}"
        repaired = True
    elif not isinstance(arguments, str):
        fixed = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
        repaired = True
    else:
        fixed = repair_tool_call_arguments(arguments, name)
        repaired = fixed != arguments

    try:
        parsed = json.loads(fixed)
    except (json.JSONDecodeError, TypeError, ValueError):
        fixed = "{}"
        repaired = True
    else:
        if not isinstance(parsed, dict):
            fixed = "{}"
            repaired = True

    call.arguments = fixed
    if repaired:
        data = dict(call.provider_data or {})
        data["arguments_repaired"] = True
        call.provider_data = data
    return repaired


def repair_current_tool_calls(calls: list[ToolCall]) -> int:
    """Repair current response tool calls in-place and return repair count."""
    repaired = 0
    for call in calls:
        if repair_current_tool_call(call):
            repaired += 1
    return repaired


def normalize_tool_name(name: str) -> str:
    raw = str(name or "").strip()
    for prefix in ("functions.", "function.", "tools.", "tool."):
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
            break
    aliases = {
        "write": "write_file",
        "read": "read_file",
        "patch": "patch_file",
        "execute_code": "run_python",
        "run_command": "terminal",
        "bash": "terminal",
    }
    return aliases.get(raw, raw)


def assistant_tool_calls(calls: list[NormalizedToolCall]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for item in calls:
        call = item.call
        payload.append(
            {
                "id": call.id or new_id("call"),
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": call.arguments or "{}",
                },
            }
        )
    return payload


def assistant_tool_calls_from_provider(calls: list[ToolCall]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for call in calls:
        call.id = call.id or new_id("call")
        payload.append(
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": call.arguments or "{}",
                },
            }
        )
    return payload


def sanitize_tool_protocol_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Hermes-style pre-call repair for assistant tool_calls and tool results.

    The model provider sees a strict OpenAI-compatible transcript: every
    assistant tool_call id has exactly one role=tool result, orphan tool results
    are removed, missing results receive a stub, and malformed argument JSON is
    repaired on the API copy without mutating the run ledger.
    """
    copied = [copy.deepcopy(message) for message in messages if isinstance(message, dict)]
    copied = [message for message in copied if message.get("role") in VALID_API_ROLES]
    _sanitize_assistant_tool_calls(copied)

    surviving_call_ids: set[str] = set()
    for message in copied:
        if message.get("role") != "assistant":
            continue
        for tool_call in message.get("tool_calls") or []:
            call_id = _tool_call_id(tool_call)
            if call_id:
                surviving_call_ids.add(call_id)

    result_by_id: dict[str, dict[str, Any]] = {}
    for message in copied:
        if message.get("role") == "tool":
            call_id = str(message.get("tool_call_id") or "")
            if not call_id or call_id not in surviving_call_ids or call_id in result_by_id:
                continue
            result_by_id[call_id] = message

    patched: list[dict[str, Any]] = []
    for message in copied:
        if message.get("role") == "tool":
            continue
        patched.append(message)
        if message.get("role") != "assistant":
            continue
        for tool_call in message.get("tool_calls") or []:
            call_id = _tool_call_id(tool_call)
            if not call_id:
                continue
            patched.append(
                result_by_id.get(call_id)
                or {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": "[Result unavailable - see runtime context summary above]",
                }
            )
    return patched


def _sanitize_assistant_tool_calls(messages: list[dict[str, Any]]) -> None:
    for message in messages:
        if message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        sanitized: list[dict[str, Any]] = []
        for index, tool_call in enumerate(tool_calls):
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if not isinstance(function, dict):
                continue
            call_id = str(tool_call.get("id") or new_id(f"call-{index}"))
            name = str(function.get("name") or "")
            arguments = function.get("arguments")
            repaired = False
            if arguments is None or arguments == "":
                arguments = "{}"
                repaired = True
            elif not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
                repaired = True
            else:
                fixed = repair_tool_call_arguments(arguments, name)
                repaired = fixed != arguments
                arguments = fixed
            sanitized.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": arguments,
                    },
                }
            )
            if repaired:
                _mark_matching_tool_result(messages, call_id)
        message["tool_calls"] = sanitized


def repair_tool_call_arguments(raw_args: str, tool_name: str = "?") -> str:
    raw = raw_args.strip() if isinstance(raw_args, str) else ""
    if not raw or raw == "None":
        return "{}"
    try:
        parsed = json.loads(raw, strict=False)
        if not isinstance(parsed, dict):
            return "{}"
        return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    fixed = re.sub(r",\s*([}\]])", r"\1", raw)
    open_curly = fixed.count("{") - fixed.count("}")
    open_bracket = fixed.count("[") - fixed.count("]")
    if open_curly > 0:
        fixed += "}" * open_curly
    if open_bracket > 0:
        fixed += "]" * open_bracket
    for _ in range(50):
        try:
            parsed = json.loads(fixed)
            if not isinstance(parsed, dict):
                return "{}"
            return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
        except json.JSONDecodeError:
            if fixed.endswith("}") and fixed.count("}") > fixed.count("{"):
                fixed = fixed[:-1]
            elif fixed.endswith("]") and fixed.count("]") > fixed.count("["):
                fixed = fixed[:-1]
            else:
                break
    escaped = _escape_invalid_chars_in_json_strings(fixed)
    try:
        parsed = json.loads(escaped)
        if not isinstance(parsed, dict):
            return "{}"
        return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
    except (json.JSONDecodeError, TypeError, ValueError):
        return "{}"


def _tool_call_id(tool_call: Any) -> str:
    if isinstance(tool_call, dict):
        return str(tool_call.get("id") or "")
    return str(getattr(tool_call, "id", "") or "")


def _mark_matching_tool_result(messages: list[dict[str, Any]], call_id: str) -> None:
    for message in messages:
        if message.get("role") != "tool" or message.get("tool_call_id") != call_id:
            continue
        content = message.get("content")
        if isinstance(content, str):
            if not content.startswith(TOOL_ARGUMENTS_REPAIRED_MARKER):
                message["content"] = f"{TOOL_ARGUMENTS_REPAIRED_MARKER}\n{content}"
        else:
            message["content"] = TOOL_ARGUMENTS_REPAIRED_MARKER
        return


def _escape_invalid_chars_in_json_strings(raw: str) -> str:
    out: list[str] = []
    in_string = False
    i = 0
    while i < len(raw):
        char = raw[i]
        if in_string:
            if char == "\\" and i + 1 < len(raw):
                out.append(char)
                out.append(raw[i + 1])
                i += 2
                continue
            if char == '"':
                in_string = False
                out.append(char)
            elif ord(char) < 0x20:
                out.append(f"\\u{ord(char):04x}")
            else:
                out.append(char)
        else:
            if char == '"':
                in_string = True
            out.append(char)
        i += 1
    return "".join(out)
