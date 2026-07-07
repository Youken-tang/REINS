"""Drop-in replacement for hermes' ``_execute_tool_calls``

This is the ``loop.py`` half of the bridge described in
```` and ````
 step 3. The static resource-access table and dispatcher live in
:mod:`benchmark.adapters.hermes_reins`; this module wraps them in
the call-site shape hermes uses at ``run_agent.py:9098``:

* input: ``assistant_message`` (anything with ``.tool_calls``) + a mutable
  ``messages: list`` to append to
* per-call invocation: caller-supplied ``invoke_tool(name, args) -> str`` so
  hermes' own ``_invoke_tool`` keeps owning tool semantics, plugin hooks,
  approval policy, etc.
* output: appends ``{"role": "tool", "content": str, "tool_call_id": id}``
  to ``messages`` *in original tool-call order*

Per CLAUDE.md, ``hermes-agent/`` is reference code and not part of this
project's commits. The function below is exactly what would land in
``hermes-agent/reins_bridge/loop.py`` once the bridge is wired in-tree;
keeping the source under ``benchmark/adapters/`` lets the unit test suite
(:mod:`tests.test_hermes_reins_loop`) and the 5-prompt smoke harness
(:mod:`benchmark.adapters.smoke`) exercise the swap without modifying
hermes-agent.

Env-flag fallback
-----------------

``HERMES_REINS=0`` falls back to sequential dispatch through the same
``invoke_tool`` callback. This mirrors the spike's recommendation that the
bridge keep an off-switch so runs marked ``hermes-vanilla`` in the
table can share the same call site as ``hermes-REINS``.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

from reins import CausalRuntime, TaskResult

from benchmark.adapters.hermes_reins_runtime import (
    HermesToolCall,
    dispatch_tool_calls,
)

logger = logging.getLogger(__name__)


# ``invoke_tool(name, args) -> str``. Caller-side: hermes' ``_invoke_tool``
# already returns a JSON/plain string; we keep the same contract so the
# bridge has zero coupling to hermes' tool registry shape.
InvokeTool = Callable[[str, dict[str, Any]], str]


class _AssistantMessage(Protocol):
    """Duck type for hermes' ``assistant_message``.

    Hermes uses ``openai.types.chat.ChatCompletionMessage``; we don't
    import openai-python here. Anything with a ``tool_calls`` attribute
    that yields :class:`HermesToolCall`-shaped objects works.
    """
    tool_calls: Sequence[HermesToolCall]


# Env-flag name. Defaulted in code for tests; documented at module level
# so the smoke harness and W3 sweep harness can flip it without import
# tricks.
ENV_FLAG = "HERMES_REINS"


def _format_failure(name: str, result: TaskResult) -> str:
    """Render a failed REINS task as the same string hermes' worker thread
    would have produced when ``_invoke_tool`` raised:

        Error executing tool '<name>': <reason>

    Matches ``run_agent.py:9395`` so downstream parsers (including
    ``_detect_tool_failure`` and the assistant-side error explainers) keep
    working.
    """
    summary = result.summary or "tool execution failed"
    return f"Error executing tool {name!r}: {summary}"


def _format_value(name: str, value: Any) -> str:
    """Best-effort coerce handler return to a string.

    Hermes tools already return strings (most return ``json.dumps(...)``);
    if a caller plugs in a non-string for testing, fall back to JSON or
    repr instead of crashing the bridge.
    """
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return repr(value)


def _sequential_dispatch(
    tool_calls: Sequence[HermesToolCall],
    invoke_tool: InvokeTool,
    messages: list,
) -> None:
    """Vanilla path: invoke each tool in order on the calling thread.

    Used when ``HERMES_REINS=0`` is set or when the batch is too small to
    benefit from REINS (single call). Bypasses ``CausalRuntime`` entirely
    so the ↔ ``hermes-REINS`` pair shares one
    call site with one branch.
    """
    for tc in tool_calls:
        name = tc.function.name
        try:
            args = json.loads(tc.function.arguments) if tc.function.arguments else {}
        except json.JSONDecodeError:
            args = {}
        if not isinstance(args, dict):
            args = {}
        try:
            content: str = _format_value(name, invoke_tool(name, args))
        except Exception as exc:  # noqa: BLE001 — same shape as hermes worker
            logger.exception("sequential invoke_tool(%r) raised", name)
            content = f"Error executing tool {name!r}: {exc}"
        messages.append({
            "role": "tool",
            "content": content,
            "tool_call_id": tc.id,
        })


def _reins_enabled(env_flag: str) -> bool:
    """REINS is on by default; ``HERMES_REINS=0`` (or ``false``/``no``)
    flips the bridge into vanilla pass-through. Anything else (including
    unset) keeps it on.
    """
    raw = os.environ.get(env_flag)
    if raw is None:
        return True
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def execute_tool_calls(
    assistant_message: _AssistantMessage,
    messages: list,
    *,
    invoke_tool: InvokeTool,
    runtime: CausalRuntime | None = None,
    root: str | Path | None = None,
    timeout_seconds: float = 600.0,
    env_flag: str = ENV_FLAG,
) -> None:
    """Drop-in replacement for hermes' ``_execute_tool_calls``.

    Hermes call site (run_agent.py:9098) reduces to:

    .. code-block:: python

        execute_tool_calls(
            assistant_message,
            messages,
            invoke_tool=lambda n, a: self._invoke_tool(
                n, a, effective_task_id, tool_call_id=..., messages=messages,
            ),
            runtime=self._reins,        # constructed once at startup
            root=self._workspace_root,
        )

    The bridge:

    1. Reads ``assistant_message.tool_calls`` (duck-typed).
    2. If the batch is empty → no-op (matches hermes' behaviour).
    3. If ``HERMES_REINS=0`` or no runtime is supplied → sequential
       dispatch on the calling thread.
    4. Else lower the batch to REINS via :func:`dispatch_tool_calls`,
       which assigns each call a :class:`ResourceAccess` from
       :data:`HERMES_TOOL_RESOURCE_TABLE` and submits to ``runtime``.
       Results come back in original call order.
    5. Append ``role=tool`` messages to ``messages`` in original order.
       Failed :class:`TaskResult` instances are rendered as the same
       error string hermes' worker thread would produce, so downstream
       failure detection (``_detect_tool_failure``) keeps working.

    Errors from :func:`dispatch_tool_calls` itself (timeout, runtime
    refused submission, …) propagate. Per-call handler exceptions do
    *not* propagate — they're surfaced as failed :class:`TaskResult`
    entries by :func:`dispatch_tool_calls`, then formatted here.
    """
    tool_calls = list(getattr(assistant_message, "tool_calls", None) or ())
    if not tool_calls:
        return

    # the prior "Group 7" optimization that bypassed
    # CausalRuntime for len(tool_calls) <= 1 was harmful — showed >50% of
    # W_fix planner rounds are 1-call, so on those rounds hermes-vanilla and
    # hermes-REINS executed identical code and the A/B was tautological.
    # Always go through CausalRuntime when REINS is enabled so the ledger /
    # trace / dispatch path is uniform across the sweep.
    if runtime is None or not _reins_enabled(env_flag):
        _sequential_dispatch(tool_calls, invoke_tool, messages)
        return

    def _handler(name: str, args: dict[str, Any]) -> str:
        return _format_value(name, invoke_tool(name, args))

    results = dispatch_tool_calls(
        tool_calls,
        handler=_handler,
        runtime=runtime,
        root=root,
        timeout_seconds=timeout_seconds,
    )

    for r in results:
        if r.result.status == "completed":
            content = _format_value(r.tool_name, r.result.value)
        else:
            content = _format_failure(r.tool_name, r.result)
        messages.append({
            "role": "tool",
            "content": content,
            "tool_call_id": r.tool_call_id,
        })


__all__ = [
    "ENV_FLAG",
    "InvokeTool",
    "execute_tool_calls",
]
