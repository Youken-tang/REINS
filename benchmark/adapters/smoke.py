"""Offline smoke harness for the REINS-into-hermes bridge

 W2 calls for 5 smoke prompts on ``hermes-agent/mini_swe_runner.py``.
Per CLAUDE.md ``hermes-agent/`` is reference code, not a project commit
target; the actual end-to-end smoke against a live model lives in the
hermes side of the bridge once it's wired in-tree.

This module is the *offline* half of that smoke: 5 fixed tool-call batches
that exercise the bridge call site (:func:`benchmark.adapters.hermes_reins_loop.execute_tool_calls`)
without booting a model. Each batch maps to a category from the spike's
tool-tier triage so the smoke covers every scheduling decision the live
runner would hit.

For each batch the harness runs the same call twice — once with
``HERMES_REINS=0`` (sequential) and once with ``HERMES_REINS=1`` (REINS)
— and reports per-batch wall-clock so a regression in scheduler fan-out
shows up as the REINS row losing its lead. Tool work is simulated with
``time.sleep`` so wall-clock differences are deterministic and don't
depend on a model provider.

Usage::

    PYTHONPATH=src .venv/bin/python -m benchmark.adapters.smoke
    PYTHONPATH=src .venv/bin/python -m benchmark.adapters.smoke --json

The non-JSON output is a small table that's easy to eyeball; ``--json``
emits machine-parseable output that the W3 sweep harness can consume.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    # Allow `python benchmark/adapters/smoke.py` style invocation by
    # bolting on the src/ path. For the documented `python -m` form
    # PYTHONPATH=src takes care of this.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from reins import CausalRuntime  # noqa: E402

from benchmark.adapters import hermes_reins_loop as loop_mod  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic tool-call objects
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _Function:
    name: str
    arguments: str


@dataclass(slots=True)
class _Call:
    id: str
    function: _Function


@dataclass(slots=True)
class _AssistantMessage:
    tool_calls: list[_Call]


def _call(call_id: str, name: str, args: dict[str, Any]) -> _Call:
    return _Call(id=call_id, function=_Function(name=name, arguments=json.dumps(args)))


# ---------------------------------------------------------------------------
# Synthetic invoke_tool: simulates IO via sleep so wall-clock is deterministic
# ---------------------------------------------------------------------------


# Per-tool simulated latency (seconds). Values picked so that a 4-fan-out
# batch shows clear speedup without making the smoke take more than a
# few seconds total.
_SIM_LATENCY = {
    "read_file": 0.05,
    "write_file": 0.05,
    "patch": 0.06,
    "search_files": 0.05,
    "session_search": 0.05,
    "web_search": 0.05,
    "vision_analyze": 0.05,
    "skill_view": 0.04,
    "skills_list": 0.04,
    "ha_get_state": 0.04,
    "ha_list_entities": 0.04,
    "ha_list_services": 0.04,
    "web_extract": 0.05,
    "terminal": 0.06,
    # default for unknown tools (browser_*, mcp_*, …)
    "_default": 0.05,
}


def _make_invoke_tool() -> tuple[callable, list[tuple[str, dict, float]]]:
    """Return ``(invoke_tool, log)`` where the log captures per-call
    (name, args, duration) so callers can sanity-check what ran."""
    log: list[tuple[str, dict, float]] = []

    def invoke(name: str, args: dict[str, Any]) -> str:
        latency = _SIM_LATENCY.get(name, _SIM_LATENCY["_default"])
        start = time.monotonic()
        time.sleep(latency)
        duration = time.monotonic() - start
        log.append((name, args, duration))
        return f"{name}::{json.dumps(args, sort_keys=True)}"

    return invoke, log


# ---------------------------------------------------------------------------
# Smoke prompts (5 batches covering every tier)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SmokePrompt:
    """One smoke 'turn'.

    ``label`` shows up in the report so a regression points at a single
    batch. ``calls`` is what the model would have emitted; the bridge
    handles dispatch from there.
    """
    label: str
    description: str
    calls: list[_Call]


def _smoke_prompts() -> list[SmokePrompt]:
    return [
        SmokePrompt(
            label="read-fanout",
            description="4 read_file on distinct paths — should fully parallelise",
            calls=[
                _call("c1", "read_file", {"path": "src/main.py"}),
                _call("c2", "read_file", {"path": "src/utils.py"}),
                _call("c3", "read_file", {"path": "tests/test_main.py"}),
                _call("c4", "read_file", {"path": "README.md"}),
            ],
        ),
        SmokePrompt(
            label="write-fanout",
            description="3 write_file on distinct paths — should fully parallelise",
            calls=[
                _call("c1", "write_file", {"path": "out/a.txt", "content": "a"}),
                _call("c2", "write_file", {"path": "out/b.txt", "content": "b"}),
                _call("c3", "write_file", {"path": "out/c.txt", "content": "c"}),
            ],
        ),
        SmokePrompt(
            label="readonly-mix",
            description="search_files + web_search + session_search — read-only triple",
            calls=[
                _call("c1", "search_files", {"pattern": "TODO"}),
                _call("c2", "web_search", {"query": "REINS scheduler"}),
                _call("c3", "session_search", {"query": "previous bug"}),
            ],
        ),
        SmokePrompt(
            label="write-then-read-same-path",
            description="write_file followed by read_file on the *same* path — must serialise",
            calls=[
                _call("c1", "write_file", {"path": "shared.txt", "content": "x"}),
                _call("c2", "read_file", {"path": "shared.txt"}),
            ],
        ),
        SmokePrompt(
            label="unknown-fallback",
            description="2 browser_navigate (unknown_workspace fallback) — must serialise",
            calls=[
                _call("c1", "browser_navigate", {"url": "https://a"}),
                _call("c2", "browser_navigate", {"url": "https://b"}),
            ],
        ),
    ]


# ---------------------------------------------------------------------------
# Run a single batch under both modes
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BatchResult:
    label: str
    description: str
    n_calls: int
    sequential_seconds: float
    reins_seconds: float
    speedup: float
    sequential_ok: bool
    reins_ok: bool
    sequential_messages: list[dict] = field(default_factory=list)
    reins_messages: list[dict] = field(default_factory=list)


def _run_one(
    prompt: SmokePrompt,
    runtime: CausalRuntime,
    timeout_seconds: float,
) -> BatchResult:
    invoke_seq, _ = _make_invoke_tool()
    invoke_reins, _ = _make_invoke_tool()
    msg = _AssistantMessage(tool_calls=prompt.calls)

    # ---- sequential leg ----
    seq_messages: list[dict] = []
    os.environ[loop_mod.ENV_FLAG] = "0"
    try:
        seq_start = time.monotonic()
        loop_mod.execute_tool_calls(
            msg, seq_messages,
            invoke_tool=invoke_seq,
            runtime=runtime,
            root=".",
            timeout_seconds=timeout_seconds,
        )
        seq_duration = time.monotonic() - seq_start
    finally:
        os.environ.pop(loop_mod.ENV_FLAG, None)

    # ---- REINS leg ----
    reins_messages: list[dict] = []
    os.environ[loop_mod.ENV_FLAG] = "1"
    try:
        reins_start = time.monotonic()
        loop_mod.execute_tool_calls(
            msg, reins_messages,
            invoke_tool=invoke_reins,
            runtime=runtime,
            root=".",
            timeout_seconds=timeout_seconds,
        )
        reins_duration = time.monotonic() - reins_start
    finally:
        os.environ.pop(loop_mod.ENV_FLAG, None)

    seq_ok = (
        [m["tool_call_id"] for m in seq_messages]
        == [c.id for c in prompt.calls]
    )
    reins_ok = (
        [m["tool_call_id"] for m in reins_messages]
        == [c.id for c in prompt.calls]
    )

    return BatchResult(
        label=prompt.label,
        description=prompt.description,
        n_calls=len(prompt.calls),
        sequential_seconds=seq_duration,
        reins_seconds=reins_duration,
        speedup=(seq_duration / reins_duration) if reins_duration > 0 else float("inf"),
        sequential_ok=seq_ok,
        reins_ok=reins_ok,
        sequential_messages=seq_messages,
        reins_messages=reins_messages,
    )


def run_smoke(timeout_seconds: float = 30.0) -> list[BatchResult]:
    """Run the 5-prompt smoke and return per-batch results.

    A fresh ``CausalRuntime`` is constructed per call so smoke runs are
    independent. ``strict_nogil=False`` keeps the smoke runnable on either
    GIL or noGIL CPython for artifact reproducibility; the sweep
    itself ran on 3.13t noGIL (see "v1 → v2 偏差记录").
    """
    runtime = CausalRuntime(
        max_workers=8,
        workspace_root=".",
        delivery_debounce=0.0,
        strict_nogil=False,
    )
    try:
        return [_run_one(p, runtime, timeout_seconds) for p in _smoke_prompts()]
    finally:
        runtime.shutdown()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _format_table(results: list[BatchResult]) -> str:
    cols = (
        ("label", 32),
        ("calls", 6),
        ("seq_s", 8),
        ("reins_s", 8),
        ("speedup", 8),
        ("ok", 6),
    )
    header = "  ".join(name.rjust(width) for name, width in cols)
    rows = [header, "-" * len(header)]
    for r in results:
        ok_flag = "yes" if (r.sequential_ok and r.reins_ok) else "NO"
        rows.append("  ".join([
            r.label.rjust(32),
            str(r.n_calls).rjust(6),
            f"{r.sequential_seconds:.3f}".rjust(8),
            f"{r.reins_seconds:.3f}".rjust(8),
            f"{r.speedup:.2f}x".rjust(8),
            ok_flag.rjust(6),
        ]))
    return "\n".join(rows)


def _to_jsonable(results: list[BatchResult]) -> list[dict]:
    return [
        {
            "label": r.label,
            "description": r.description,
            "n_calls": r.n_calls,
            "sequential_seconds": r.sequential_seconds,
            "reins_seconds": r.reins_seconds,
            "speedup": r.speedup,
            "sequential_ok": r.sequential_ok,
            "reins_ok": r.reins_ok,
        }
        for r in results
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true",
                        help="emit JSON instead of a human-readable table")
    parser.add_argument("--timeout-seconds", type=float, default=30.0,
                        help="per-batch dispatch timeout (default: 30)")
    args = parser.parse_args(argv)

    results = run_smoke(timeout_seconds=args.timeout_seconds)

    if args.json:
        json.dump(_to_jsonable(results), sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print(_format_table(results))
        all_ok = all(r.sequential_ok and r.reins_ok for r in results)
        print("\nall batches ordered correctly:", "yes" if all_ok else "NO")

    failures = [r for r in results if not (r.sequential_ok and r.reins_ok)]
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
