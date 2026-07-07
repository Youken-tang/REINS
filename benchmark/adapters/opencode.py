"""OpenCode adapter for benchmark.

Invokes the `opencode` CLI in non-interactive mode with --format json,
parses the structured event stream to extract tool calls, results, and
final answer into a TaskTrace.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from typing import Any

from benchmark.adapters import TaskTrace, ToolCall, ToolResult
from benchmark.adapters.base import AgentAdapter, TaskInput

# OpenCode tool names → benchmark canonical names
_TOOL_NAME_MAP: dict[str, str] = {
    "read": "read_file",
    "write": "write_file",
    "edit": "replace_in_file",
    "patch": "patch_file",
    "apply_patch": "patch_file",
    "shell": "terminal",
    "bash": "terminal",
    "glob": "list_tree",
    "grep": "search_files",
    "task": "delegate_task",
    "todo": "todo_write",
    "fetch": "web_fetch",
    "search": "web_search",
}


def _map_tool_name(opencode_name: str) -> str:
    return _TOOL_NAME_MAP.get(opencode_name, opencode_name)


class OpenCodeAdapter(AgentAdapter):
    """Adapter wrapping OpenCode CLI (v1.15.5+) for benchmark execution."""

    def __init__(self) -> None:
        self._model = ""
        self._base_url = ""
        self._opencode_bin = ""

    @property
    def name(self) -> str:
        return "opencode"

    def setup(self, model: str, base_url: str, **kwargs: Any) -> None:
        # opencode CLI requires `provider/model` form; auto-prepend a default
        # provider id when the caller passes a bare model name. The provider
        # entry must already exist in ~/.config/opencode/opencode.json.
        if model and "/" not in model:
            default_provider = kwargs.get("opencode_provider", "bobdong")
            self._model = f"{default_provider}/{model}"
        else:
            self._model = model
        self._base_url = base_url
        self._opencode_bin = kwargs.get("opencode_bin", "") or shutil.which("opencode") or "opencode"

    def run_task(self, task: TaskInput) -> TaskTrace:
        trace = TaskTrace(
            task_id=task.task_id,
            agent_name=self.name,
            start_time=time.time(),
        )

        cmd = [
            self._opencode_bin,
            "run",
            task.prompt,
            "--format", "json",
            "--dir", task.workspace,
        ]
        if self._model:
            cmd.extend(["--model", self._model])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=task.timeout,
                cwd=task.workspace,
            )
            self._parse_events(result.stdout, trace)
            if result.returncode != 0 and not trace.error:
                stderr_tail = (result.stderr or "").strip()[-500:]
                if stderr_tail:
                    trace.error = f"exit {result.returncode}: {stderr_tail}"

        except subprocess.TimeoutExpired:
            trace.error = f"timeout after {task.timeout}s"
        except FileNotFoundError:
            trace.error = f"opencode binary not found: {self._opencode_bin}"
        except Exception as exc:
            trace.error = str(exc)
        finally:
            trace.end_time = time.time()

        return trace

    def _parse_events(self, stdout: str, trace: TaskTrace) -> None:
        """Parse newline-delimited JSON events from opencode --format json."""
        text_parts: list[str] = []

        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue

            event_type = event.get("type", "")

            if event_type == "tool_use":
                self._handle_tool_event(event, trace)

            elif event_type == "text":
                part = event.get("part", {})
                text = part.get("text", "").strip()
                if text:
                    text_parts.append(text)

            elif event_type == "error":
                error_data = event.get("error", {})
                msg = ""
                if isinstance(error_data, dict):
                    data = error_data.get("data", {})
                    if isinstance(data, dict):
                        msg = data.get("message", "")
                    if not msg:
                        msg = error_data.get("name", str(error_data))
                else:
                    msg = str(error_data)
                if msg and not trace.error:
                    trace.error = msg

            elif event_type == "step_start":
                trace.model_calls += 1

            elif event_type == "step_finish":
                part = event.get("part", {})
                if isinstance(part, dict):
                    tokens = part.get("tokens", {})
                    if isinstance(tokens, dict):
                        total = tokens.get("total")
                        if isinstance(total, (int, float)):
                            trace.total_tokens = max(trace.total_tokens, int(total))

        if text_parts:
            trace.final_answer = "\n".join(text_parts)

    def _handle_tool_event(self, event: dict[str, Any], trace: TaskTrace) -> None:
        """Extract tool call and result from a tool_use event."""
        part = event.get("part", {})
        if not isinstance(part, dict):
            return

        tool_name = part.get("tool", "unknown")
        mapped_name = _map_tool_name(tool_name)
        call_id = part.get("id", "") or part.get("callID", "")
        state = part.get("state", {})

        if not isinstance(state, dict):
            return

        # Extract input arguments
        tool_input = state.get("input", {})
        if isinstance(tool_input, str):
            try:
                tool_input = json.loads(tool_input)
            except (json.JSONDecodeError, ValueError):
                tool_input = {"raw": tool_input}
        if not isinstance(tool_input, dict):
            tool_input = {"raw": str(tool_input)}

        timestamp = event.get("timestamp", 0)
        if timestamp > 1e12:
            timestamp = timestamp / 1000.0

        trace.tool_calls.append(ToolCall(
            name=mapped_name,
            arguments=tool_input,
            call_id=call_id,
            timestamp=timestamp,
        ))

        # Extract output/result
        status = state.get("status", "")
        output = state.get("output", "")
        if isinstance(output, dict):
            output = json.dumps(output, ensure_ascii=False)[:500]
        elif isinstance(output, str):
            output = output[:500]
        else:
            output = str(output)[:500]

        trace.tool_results.append(ToolResult(
            call_id=call_id,
            output=output,
            success=(status == "completed"),
        ))

    def teardown(self) -> None:
        pass

    def supports_parallel(self) -> bool:
        return True

    def supports_delegation(self) -> bool:
        return True
