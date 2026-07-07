"""Trace JSONL post-processing.

Reads a trace.jsonl emitted by ``high_agent.runtime.trace.TraceWriter`` and
returns three families of metrics:

1. ``compute_basic_metrics(trace)`` — wall/task seconds, parallelism
   efficiency, worker idle ratio, planner_to_tool_ratio.

2. ``compute_F1_F8(trace)`` — eight scheduler-quality findings
   (F1 explicit-dep sparsity, F2 conflict density, F3 ledger sufficiency,
   F4 ready→running queue, F5 critical-path stall, F6 LLM RTT share,
   F7 build-like completion, F8 re-call dedupe rate). Output is
   JSON-friendly so ``--out metrics.json`` can be fed straight into
   figure scripts.

3. ``replay_tool_calls(trace, runner)`` — extracts the ordered tool-call
   sequence from ``planner.lowered`` + ``task.submitted`` events and
   feeds it back into a runner callable. Used by ablations to compare
   alternative scheduler configurations on the *same* model output.

The schema this script consumes is defined in
``high_agent.runtime.trace.TRACE_EVENT_SCHEMA``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator


TraceEvent = dict[str, Any]


# ───────────────────────────── loader ──────────────────────────────────────


def iter_trace(path: str | Path) -> Iterator[TraceEvent]:
    """Yield trace events from a JSONL file. Bad lines are skipped silently."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_trace(path: str | Path) -> list[TraceEvent]:
    return list(iter_trace(path))


def _by_event(trace: Iterable[TraceEvent]) -> dict[str, list[TraceEvent]]:
    out: dict[str, list[TraceEvent]] = defaultdict(list)
    for ev in trace:
        name = ev.get("event")
        if isinstance(name, str):
            out[name].append(ev)
    return out


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * pct
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


# ─────────────────────── basic metrics #1) ───────────────────────────


@dataclass
class BasicMetrics:
    wall_seconds: float
    task_seconds: float
    parallelism_efficiency: float
    worker_idle_ratio: float
    planner_to_tool_ratio: float
    task_count: int
    planner_count: int
    tool_count: int
    max_workers_observed: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "wall_seconds": round(self.wall_seconds, 6),
            "task_seconds": round(self.task_seconds, 6),
            "parallelism_efficiency": round(self.parallelism_efficiency, 4),
            "worker_idle_ratio": round(self.worker_idle_ratio, 4),
            "planner_to_tool_ratio": round(self.planner_to_tool_ratio, 4),
            "task_count": self.task_count,
            "planner_count": self.planner_count,
            "tool_count": self.tool_count,
            "max_workers_observed": self.max_workers_observed,
        }


def compute_basic_metrics(
    trace: Iterable[TraceEvent],
    *,
    max_workers: int | None = None,
) -> BasicMetrics:
    """Aggregate top-line wall/task/parallelism numbers.

    ``max_workers`` is the configured worker pool size; if not given we
    estimate it from the peak overlap of ``task.started``/``task.completed``
    pairs.
    """
    trace = list(trace)
    grouped = _by_event(trace)

    if not trace:
        return BasicMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, max_workers or 0)

    timestamps = [float(ev["ts"]) for ev in trace if "ts" in ev]
    wall_seconds = max(timestamps) - min(timestamps) if timestamps else 0.0

    task_seconds = sum(
        float(ev.get("duration_seconds") or 0.0)
        for ev in grouped.get("task.completed", [])
    )

    started = grouped.get("task.started", [])
    completed = grouped.get("task.completed", [])
    completion_lookup: dict[str, float] = {
        ev["task_id"]: float(ev["ts"])
        for ev in completed
        if "task_id" in ev and "ts" in ev
    }
    intervals: list[tuple[float, float]] = []
    for ev in started:
        tid = ev.get("task_id")
        ts = ev.get("ts")
        if tid in completion_lookup and ts is not None:
            intervals.append((float(ts), completion_lookup[tid]))

    peak_concurrency = _peak_concurrency(intervals)
    workers = max_workers or peak_concurrency or 1

    if wall_seconds > 0 and workers > 0:
        parallelism_efficiency = task_seconds / (wall_seconds * workers)
        worker_idle_ratio = max(
            0.0, 1.0 - (task_seconds / (wall_seconds * workers))
        )
    else:
        parallelism_efficiency = 0.0
        worker_idle_ratio = 0.0

    tool_count = sum(
        1 for ev in grouped.get("task.submitted", []) if ev.get("kind") == "tool"
    )
    planner_count = len(grouped.get("planner.responded", []))
    if not planner_count:
        planner_count = len(grouped.get("planner.completed", []))
    if tool_count > 0:
        planner_to_tool_ratio = planner_count / tool_count
    else:
        planner_to_tool_ratio = 0.0

    return BasicMetrics(
        wall_seconds=wall_seconds,
        task_seconds=task_seconds,
        parallelism_efficiency=min(parallelism_efficiency, 1.0),
        worker_idle_ratio=min(worker_idle_ratio, 1.0),
        planner_to_tool_ratio=planner_to_tool_ratio,
        task_count=len(grouped.get("task.submitted", [])),
        planner_count=planner_count,
        tool_count=tool_count,
        max_workers_observed=peak_concurrency,
    )


def _peak_concurrency(intervals: list[tuple[float, float]]) -> int:
    if not intervals:
        return 0
    edges: list[tuple[float, int]] = []
    for start, end in intervals:
        edges.append((start, +1))
        edges.append((end, -1))
    edges.sort()
    cur = 0
    peak = 0
    for _, delta in edges:
        cur += delta
        if cur > peak:
            peak = cur
    return peak


# ─────────────────────── F1–F8 findings #2) ──────────────────────────


def compute_F1_F8(trace: Iterable[TraceEvent]) -> dict[str, Any]:
    """Return scalar metrics for findings F1 through F8 plus their inputs."""
    trace = list(trace)
    grouped = _by_event(trace)

    submitted = grouped.get("task.submitted", [])
    submitted_count = len(submitted)
    tool_count = sum(1 for ev in submitted if ev.get("kind") == "tool")

    # F1: explicit-dependency sparsity. submitted.deps lists the predicates
    # the planner declared at submit time; ratio = explicit / total submits.
    explicit_dep_count = 0
    total_with_deps_window = 0
    for ev in submitted:
        deps = ev.get("deps") or []
        if isinstance(deps, list) and deps:
            explicit_dep_count += 1
        total_with_deps_window += 1
    f1_explicit_dep_ratio = (
        explicit_dep_count / total_with_deps_window
        if total_with_deps_window
        else 0.0
    )

    # F2: conflict density per submitted task.
    conflict_count = len(grouped.get("resource.conflict", []))
    f2_conflict_density = (
        conflict_count / submitted_count if submitted_count else 0.0
    )

    # F3: ledger digest size distribution + planner pass-through rate.
    digest_sizes = [
        int(ev.get("ledger_digest_size") or 0)
        for ev in grouped.get("planner.requested", [])
        if ev.get("ledger_digest_size") is not None
    ]
    planner_responded = grouped.get("planner.responded", [])
    planner_requested = grouped.get("planner.requested", [])
    f3_planner_pass_rate = (
        len(planner_responded) / len(planner_requested)
        if planner_requested
        else 0.0
    )
    f3_ledger_size_p50 = _percentile([float(x) for x in digest_sizes], 0.5)
    f3_ledger_size_p95 = _percentile([float(x) for x in digest_sizes], 0.95)

    # F4: ready→running queue depth. wait_seconds is logged on task.ready.
    wait_seconds = [
        float(ev.get("wait_seconds") or 0.0)
        for ev in grouped.get("task.ready", [])
    ]
    f4_ready_to_run_p50 = _percentile(wait_seconds, 0.5)
    f4_ready_to_run_p95 = _percentile(wait_seconds, 0.95)

    # F5: critical-path stall — gaps between consecutive planner.requested
    # events (a long gap indicates a single critical-path tool blocked the
    # next planner round).
    planner_ts = sorted(
        float(ev.get("ts") or 0.0) for ev in planner_requested if ev.get("ts")
    )
    gaps: list[float] = []
    for i in range(1, len(planner_ts)):
        gaps.append(planner_ts[i] - planner_ts[i - 1])
    f5_planner_gap_p50 = _percentile(gaps, 0.5)
    f5_planner_gap_p95 = _percentile(gaps, 0.95)

    # F6: LLM RTT share of wall.
    rtt_total_ms = sum(
        float(ev.get("rtt_ms") or 0.0)
        for ev in grouped.get("model.future.resumed", [])
    )
    timestamps = [float(ev["ts"]) for ev in trace if "ts" in ev]
    wall_seconds = max(timestamps) - min(timestamps) if timestamps else 0.0
    f6_llm_rtt_share = (
        (rtt_total_ms / 1000.0) / wall_seconds if wall_seconds > 0 else 0.0
    )

    # F7: build-like completion gate firings.
    gate_events = grouped.get("controller.completion_gate", [])
    gate_accepted = sum(1 for ev in gate_events if ev.get("accepted"))
    gate_build_like = sum(1 for ev in gate_events if ev.get("build_like"))
    gate_reasons = Counter(
        str(ev.get("reason") or "") for ev in gate_events if ev.get("reason")
    )
    f7_completion_gate_accept_rate = (
        gate_accepted / len(gate_events) if gate_events else 0.0
    )

    # F8: re-call dedupe rate from planner.lowered.
    lowered = grouped.get("planner.lowered", [])
    deduped = sum(int(ev.get("deduped") or 0) for ev in lowered)
    submitted_lowered = sum(int(ev.get("submitted") or 0) for ev in lowered)
    dropped_lowered = sum(int(ev.get("dropped") or 0) for ev in lowered)
    total_lowered = submitted_lowered + dropped_lowered
    f8_dedupe_rate = deduped / total_lowered if total_lowered else 0.0

    return {
        "F1": {
            "explicit_dep_ratio": round(f1_explicit_dep_ratio, 4),
            "submits_with_explicit_deps": explicit_dep_count,
            "submits_total": total_with_deps_window,
        },
        "F2": {
            "conflict_density": round(f2_conflict_density, 4),
            "conflict_count": conflict_count,
            "submitted_count": submitted_count,
        },
        "F3": {
            "planner_pass_rate": round(f3_planner_pass_rate, 4),
            "ledger_digest_size_p50": round(f3_ledger_size_p50, 2),
            "ledger_digest_size_p95": round(f3_ledger_size_p95, 2),
            "planner_requests": len(planner_requested),
            "planner_responses": len(planner_responded),
        },
        "F4": {
            "ready_to_run_p50": round(f4_ready_to_run_p50, 6),
            "ready_to_run_p95": round(f4_ready_to_run_p95, 6),
            "samples": len(wait_seconds),
        },
        "F5": {
            "planner_gap_p50": round(f5_planner_gap_p50, 6),
            "planner_gap_p95": round(f5_planner_gap_p95, 6),
            "samples": len(gaps),
        },
        "F6": {
            "llm_rtt_share": round(f6_llm_rtt_share, 4),
            "rtt_total_ms": round(rtt_total_ms, 2),
            "wall_seconds": round(wall_seconds, 4),
            "future_count": len(grouped.get("model.future.resumed", [])),
        },
        "F7": {
            "completion_gate_accept_rate": round(f7_completion_gate_accept_rate, 4),
            "build_like_count": gate_build_like,
            "gate_events": len(gate_events),
            "reasons": dict(gate_reasons),
        },
        "F8": {
            "dedupe_rate": round(f8_dedupe_rate, 4),
            "deduped": deduped,
            "submitted": submitted_lowered,
            "dropped": dropped_lowered,
            "total_lowered": total_lowered,
        },
        "_basic": compute_basic_metrics(trace).as_dict(),
        "tool_count": tool_count,
    }


# ─────────────────────── replay #3) ──────────────────────────────────


@dataclass
class ReplayCall:
    """A single tool call lifted from a trace, ready to feed an alt runner."""

    task_id: str
    kind: str
    parent_id: str | None
    deps: list[Any]
    resource_access: Any
    submitted_at: float


def extract_tool_calls(trace: Iterable[TraceEvent]) -> list[ReplayCall]:
    """Return the ordered tool-call sequence from a trace.

    Order is by ``ts`` of the ``task.submitted`` event. Used by-2/3/5
    ablations that need to feed the *same* planner output into a different
    scheduler configuration.
    """
    calls: list[ReplayCall] = []
    for ev in _by_event(trace).get("task.submitted", []):
        if ev.get("kind") not in {"tool", "agent_loop"}:
            continue
        calls.append(
            ReplayCall(
                task_id=str(ev.get("task_id") or ""),
                kind=str(ev.get("kind") or ""),
                parent_id=ev.get("parent_id"),
                deps=list(ev.get("deps") or []),
                resource_access=ev.get("resource_access"),
                submitted_at=float(ev.get("ts") or 0.0),
            )
        )
    calls.sort(key=lambda c: c.submitted_at)
    return calls


def replay_tool_calls(
    trace: Iterable[TraceEvent],
    runner: Callable[[ReplayCall], Any],
) -> list[Any]:
    """Walk extracted tool calls and feed each into ``runner``.

    Returns the list of runner return values. ``runner`` is responsible for
    deciding whether to actually re-execute the tool, drop a no-op, or just
    record the call shape — keep this layer minimal so-3 (debounce
    sweep) and-5 (suspend/resume sweep) can swap the runner without
    touching extraction.
    """
    out: list[Any] = []
    for call in extract_tool_calls(trace):
        out.append(runner(call))
    return out


# ─────────────────────────── CLI ───────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="parse_trace",
        description="Compute basic + F1–F8 metrics from a trace.jsonl file.",
    )
    parser.add_argument("trace", type=Path, help="path to trace.jsonl")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output path for metrics.json (default: stdout)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="configured worker pool size (improves parallelism_efficiency)",
    )
    parser.add_argument(
        "--mode",
        choices=("basic", "f1f8", "both"),
        default="both",
    )
    args = parser.parse_args(argv)

    events = load_trace(args.trace)
    metrics: dict[str, Any] = {}
    if args.mode in ("basic", "both"):
        metrics["basic"] = compute_basic_metrics(
            events, max_workers=args.max_workers
        ).as_dict()
    if args.mode in ("f1f8", "both"):
        metrics["findings"] = compute_F1_F8(events)

    payload = json.dumps(metrics, indent=2, sort_keys=True, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
    else:
        sys.stdout.write(payload + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
