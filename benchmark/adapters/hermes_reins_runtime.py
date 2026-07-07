"""REINS scheduler bridge for hermes-agent tool execution.

See package docstring (:mod:`benchmark.adapters`) for the contract.
Implementation references are in
````

The adapter is *static*: it does not import hermes-agent. Tests pass
duck-typed call objects shaped like hermes' ``ChatCompletionMessageToolCall``
(``.id``, ``.function.name``, ``.function.arguments`` JSON string).

Tool-tier triage (per spike

* **Read-only safe** — ported verbatim from hermes' ``_PARALLEL_SAFE_TOOLS``
  (``run_agent.py:311-323``). 11 tools.
* **Path-scoped file mutators** — ``read_file`` / ``write_file`` / ``patch``
  (``_PATH_SCOPED_TOOLS`` at ``run_agent.py:326``). 2 mutators + the read-only
  ``read_file`` overlap.
* **Terminal / shell** — single tool ``terminal``; access is inferred via the
  high_agent shell-classification ladder
  (:func:`high_agent.tools.core._process_resource_access`) so that ``ls``,
  ``cat``, ``grep`` etc. land on read-only access while ``rm``, ``mv``,
  ``sed -i`` land on writes.
* **Agent-internal trivial** — ``clarify``, ``todo_*``, ``memory_*``,
  ``skill_*``, ``delegate``, etc. None of them touch shared workspace state
  across calls; mark as ``empty()``.
* **Domain-specific** — browser_*, feishu_*, mcp_*, discord, homeassistant,
  image_generation, neutts_synth, send_message, vision_*, voice_*, web_*,
  xai_*, yuanbao_*. ~40 tools. Default to
  :py:meth:`ResourceAccess.unknown_workspace` per spike risk plan
  they serialise globally until annotated.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from reins import AgentTaskSpec, CausalRuntime, TaskResult
from reins.runtime.resource_access import ResourceAccess, normalize_component

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool-call object protocol
# ---------------------------------------------------------------------------


class _Function(Protocol):
    name: str
    arguments: str


class HermesToolCall(Protocol):
    """Duck type matching hermes' OpenAI-shaped tool-call object.

    Hermes uses ``openai.types.chat.ChatCompletionMessageToolCall`` directly;
    we don't import it here so the adapter stays decoupled from openai-python.
    Anything with ``.id``, ``.function.name``, ``.function.arguments`` (JSON
    string) satisfies this protocol.
    """

    id: str
    function: _Function


# ---------------------------------------------------------------------------
# Static tool table
# ---------------------------------------------------------------------------


_PARALLEL_SAFE_HERMES_TOOLS: frozenset[str] = frozenset({
    # Source: hermes-agent/run_agent.py:311-323 (_PARALLEL_SAFE_TOOLS).
    "ha_get_state",
    "ha_list_entities",
    "ha_list_services",
    "read_file",
    "search_files",
    "session_search",
    "skill_view",
    "skills_list",
    "vision_analyze",
    "web_extract",
    "web_search",
})

_PATH_SCOPED_HERMES_TOOLS: frozenset[str] = frozenset({
    # Source: hermes-agent/run_agent.py:326 (_PATH_SCOPED_TOOLS).
    "read_file",
    "write_file",
    "patch",
})

_AGENT_INTERNAL_HERMES_TOOLS: frozenset[str] = frozenset({
    # Spike tier "agent-internal trivial". None of these reach shared
    # workspace state, so they always parallelise with anything except
    # themselves+interactive (clarify is sequential by hermes' own list).
    "clarify",
    "delegate",
    "memory_add",
    "memory_search",
    "memory_view",
    "skill_run",
    "skills_view",
    "todo_add",
    "todo_complete",
    "todo_list",
    "todo_remove",
    "todo_update",
    "interrupt",
    "checkpoint_create",
})

# Tools that must serialise (interactive / user-facing). Mirrors hermes'
# _NEVER_PARALLEL_TOOLS at run_agent.py:308 plus the spike's broader
# interactive bucket.
_INTERACTIVE_HERMES_TOOLS: frozenset[str] = frozenset({"clarify"})


_TOOL_PATH_ARG_KEYS: dict[str, tuple[str, ...]] = {
    # Map hermes tool name → ordered tuple of argument keys to probe for the
    # workspace path. ``read_file`` / ``write_file`` use ``"path"``; ``patch``
    # carries ``"target"`` in some hermes builds and ``"path"`` in others.
    "read_file": ("path",),
    "write_file": ("path",),
    "patch": ("path", "target"),
}


def _file_resource_access_for_path_arg(
    tool_name: str,
    args: Mapping[str, Any],
    root: str | None,
) -> ResourceAccess | None:
    """Return ResourceAccess for path-scoped hermes tools, or None when the
    path isn't statically extractable (caller falls back to ``unknown``)."""
    keys = _TOOL_PATH_ARG_KEYS.get(tool_name, ())
    raw_path: str | None = None
    for key in keys:
        candidate = args.get(key)
        if isinstance(candidate, str) and candidate.strip():
            raw_path = candidate.strip()
            break
    if raw_path is None:
        return None
    component = f"file:{raw_path}"
    if tool_name == "read_file":
        return ResourceAccess(
            reads=frozenset({normalize_component(component, root)}),
        )
    return ResourceAccess(
        writes=frozenset({normalize_component(component, root)}),
        side_effect_level="local",
    )


def _terminal_resource_access(
    args: Mapping[str, Any],
    root: str | None,
) -> ResourceAccess:
    """Defer to high_agent's shell classifier.

    The function is imported lazily so importing this module never pulls in
    the full ``high_agent.tools`` surface for adapter consumers that only
    care about the static table.
    """
    from high_agent.tools.core import _process_resource_access

    return _process_resource_access(dict(args), root, kind="terminal")


def _empty_resource_access(_args: Mapping[str, Any], _root: str | None) -> ResourceAccess:
    return ResourceAccess.empty()


def _unknown_resource_access(_args: Mapping[str, Any], _root: str | None) -> ResourceAccess:
    return ResourceAccess.unknown_workspace()


def _read_only_resource_access(_args: Mapping[str, Any], _root: str | None) -> ResourceAccess:
    """For tools in _PARALLEL_SAFE_HERMES_TOOLS that don't need a path probe.

    Hermes treats them as read-only with no path scoping; REINS gets the same
    coarse signal: an empty ``ResourceAccess`` with no writes. Two read-only
    tools never conflict per ``access_conflicts`` rules.
    """
    return ResourceAccess.empty()


_ResourceAccessFn = Callable[[Mapping[str, Any], str | None], ResourceAccess]


HERMES_TOOL_RESOURCE_TABLE: dict[str, _ResourceAccessFn] = {
    # --- path-scoped file mutators (overlap with read-only `read_file`) ---
    "read_file": lambda args, root: (
        _file_resource_access_for_path_arg("read_file", args, root)
        or ResourceAccess.empty()
    ),
    "write_file": lambda args, root: (
        _file_resource_access_for_path_arg("write_file", args, root)
        or ResourceAccess.unknown_workspace()
    ),
    "patch": lambda args, root: (
        _file_resource_access_for_path_arg("patch", args, root)
        or ResourceAccess.unknown_workspace()
    ),

    # --- terminal: full shell classifier ---
    "terminal": _terminal_resource_access,

    # --- read-only safe (per hermes _PARALLEL_SAFE_TOOLS, minus read_file
    #     which is path-scoped above) ---
    "ha_get_state": _read_only_resource_access,
    "ha_list_entities": _read_only_resource_access,
    "ha_list_services": _read_only_resource_access,
    "search_files": _read_only_resource_access,
    "session_search": _read_only_resource_access,
    "skill_view": _read_only_resource_access,
    "skills_list": _read_only_resource_access,
    "vision_analyze": _read_only_resource_access,
    "web_extract": _read_only_resource_access,
    "web_search": _read_only_resource_access,

    # --- agent-internal trivial (no shared workspace state) ---
    **{name: _empty_resource_access for name in _AGENT_INTERNAL_HERMES_TOOLS},
}


# — name-pattern fallback for the ~40 hermes tools that
# were not individually annotated. Previously they all fell through to
# `unknown_workspace()` and serialised globally. By tagging them as
# `external_read` (network reads) or `external_write` (network mutations) we
# let them parallelise with each other and with `empty()`-class tools.
#
# Patterns matched in order; first hit wins. Explicit overrides go in
# `_NAMED_EXTERNAL_OVERRIDES`. Anything still unmatched falls through to the
# old `unknown_workspace()` baseline.

_EXTERNAL_WRITE_NAME_TOKENS: tuple[str, ...] = (
    "_post", "_send", "_create", "_generate", "_write", "_update", "_delete",
    "_remove", "_publish", "_dispatch", "_emit", "_invoke",
    "send_", "post_", "create_", "generate_", "write_", "update_", "delete_",
    "publish_", "dispatch_", "emit_", "invoke_",
)

_EXTERNAL_READ_NAME_TOKENS: tuple[str, ...] = (
    "_search", "_lookup", "_get", "_list", "_view", "_fetch", "_query",
    "_read", "_inspect", "_describe",
    "search_", "lookup_", "get_", "list_", "view_", "fetch_", "query_",
    "read_", "inspect_", "describe_",
)

_EXTERNAL_NAME_PREFIXES: tuple[str, ...] = (
    "mcp_", "feishu_", "browser_", "web_", "discord", "homeassistant",
    "image_generation", "voice_", "vision_", "xai_", "yuanbao_", "neutts_",
    "send_message",
)

_NAMED_EXTERNAL_OVERRIDES: dict[str, str] = {
    # name → "read" | "write"
    "send_message": "write",
    "image_generation": "write",
}


def _classify_external_by_name(tool_name: str) -> ResourceAccess | None:
    """Return external_read / external_write for a name-pattern hit, else None."""
    override = _NAMED_EXTERNAL_OVERRIDES.get(tool_name)
    if override == "write":
        return ResourceAccess(side_effect_level="external_write")
    if override == "read":
        return ResourceAccess(side_effect_level="external_read")
    matches_external = any(tool_name.startswith(p) for p in _EXTERNAL_NAME_PREFIXES)
    if not matches_external:
        return None
    lname = tool_name.lower()
    if any(tok in lname for tok in _EXTERNAL_WRITE_NAME_TOKENS):
        return ResourceAccess(side_effect_level="external_write")
    if any(tok in lname for tok in _EXTERNAL_READ_NAME_TOKENS):
        return ResourceAccess(side_effect_level="external_read")
    # External-prefixed but no read/write hint — default conservatively to
    # external_read (most "service_action" tools are read-leaning lookups).
    return ResourceAccess(side_effect_level="external_read")


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------


def resource_access_for(
    tool_name: str,
    args: Mapping[str, Any] | None = None,
    *,
    root: str | None = None,
) -> ResourceAccess:
    """Return the declared ``ResourceAccess`` for a hermes tool call.

    Falls back through three layers:
      1. ``HERMES_TOOL_RESOURCE_TABLE`` static map (exact-name).
      2. ``_classify_external_by_name`` name-pattern dispatcher (Group 6).
      3. ``ResourceAccess.unknown_workspace()`` — the safe baseline that
         serialises against everything.
    """
    args = args or {}
    if tool_name in _INTERACTIVE_HERMES_TOOLS:
        # Hermes' own gate forces these to run sequentially. Bake the same
        # signal into ResourceAccess so REINS' scheduler sees it too. Must
        # take precedence over any other declarator.
        return ResourceAccess(unknown=True, side_effect_level="interactive")
    fn = HERMES_TOOL_RESOURCE_TABLE.get(tool_name)
    if fn is None:
        external = _classify_external_by_name(tool_name)
        if external is not None:
            return external
        return ResourceAccess.unknown_workspace()
    try:
        return fn(args, root)
    except Exception:  # noqa: BLE001 — annotation must never crash dispatch
        logger.exception(
            "resource_access_for(%r) raised; falling back to unknown_workspace",
            tool_name,
        )
        return ResourceAccess.unknown_workspace()


@dataclass(slots=True)
class DispatchResult:
    """Per-call result returned by :func:`dispatch_tool_calls`.

    Re-orders tool-call output to match the input call order so callers can
    append ``role=tool`` messages in the same sequence the model emitted.
    """
    tool_call_id: str
    tool_name: str
    task_id: str
    result: TaskResult


def _parse_arguments(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


ToolHandler = Callable[[str, dict[str, Any]], Any]
"""Caller-supplied executor: ``(tool_name, args) -> hermes-shaped result``.

The bridge does not know how hermes runs its tools; the caller (i.e. the
swap-target replacing ``_execute_tool_calls``) supplies the handler that
already exists in hermes' codebase. Whatever the handler returns is wrapped
in ``TaskResult.completed(value=...)`` so REINS can deliver it.
"""


def _make_task(
    call: HermesToolCall,
    handler: ToolHandler,
    *,
    root: str | None,
) -> AgentTaskSpec:
    name = call.function.name
    args = _parse_arguments(call.function.arguments)
    access = resource_access_for(name, args, root=root)

    def _run(_ctx: Any) -> TaskResult:
        try:
            value = handler(name, args)
        except Exception as exc:  # noqa: BLE001 — surface as failed task
            return TaskResult.failed(
                f"hermes tool {name!r} raised: {exc}",
                error_type=type(exc).__name__,
            )
        return TaskResult.completed(summary=f"hermes:{name}", value=value)

    return AgentTaskSpec(
        kind="tool",
        goal=f"hermes:{name}",
        input={"tool_name": name, "args": args, "tool_call_id": call.id},
        resource_access=access,
        handler=_run,
        metadata={"hermes_tool_call_id": call.id, "hermes_tool_name": name},
    )


def dispatch_tool_calls(
    tool_calls: Sequence[HermesToolCall],
    *,
    handler: ToolHandler,
    runtime: CausalRuntime,
    root: str | Path | None = None,
    timeout_seconds: float = 600.0,
) -> list[DispatchResult]:
    """Lower a hermes tool-call batch to REINS and return ordered results.

    This is the function intended to replace
    ``hermes-agent/run_agent.py:9098`` ``_execute_tool_calls`` once the
    bridge is wired in-tree (see spike Inputs are duck-typed; outputs
    preserve the original call order so the caller can append per-call
    ``role=tool`` messages without re-sorting.

    Errors from the handler are surfaced as ``TaskResult.failed`` so the
    scheduler still emits a delivery for the call — the caller decides
    whether to convert that into a hermes-shaped error string.
    """
    root_str = str(root) if root is not None else None
    specs: list[AgentTaskSpec] = []
    submissions: list[tuple[str, str, str]] = []  # (task_id, call_id, name)
    for call in tool_calls:
        spec = _make_task(call, handler, root=root_str)
        specs.append(spec)
        submissions.append((spec.task_id, call.id, call.function.name))
    if specs:
        runtime.submit(specs)

    expected = {task_id for task_id, _, _ in submissions}
    collected: dict[str, TaskResult] = {}
    deadline = time.monotonic() + timeout_seconds

    while expected - collected.keys():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            missing = expected - collected.keys()
            raise TimeoutError(
                f"hermes_reins: timed out waiting for {len(missing)} "
                f"tool result(s) after {timeout_seconds}s"
            )
        batch = runtime.wait_next_delivery(timeout=remaining)
        if batch is None:
            continue
        for event in batch.events:
            if event.task_id in expected:
                collected[event.task_id] = event.result

    return [
        DispatchResult(
            tool_call_id=call_id,
            tool_name=name,
            task_id=task_id,
            result=collected[task_id],
        )
        for task_id, call_id, name in submissions
    ]
