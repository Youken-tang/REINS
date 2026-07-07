"""Enhanced metrics extraction from runtime trace and ledger."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RuntimeMetrics:
    """Comprehensive metrics extracted from a single benchmark run."""

    parallelism_timeline: list[tuple[float, int]] = field(default_factory=list)
    peak_parallelism: int = 0
    avg_parallelism: float = 0.0
    conflict_count: int = 0
    conflict_serialization_seconds: float = 0.0
    planning_stall_seconds: float = 0.0
    streaming_dispatch_count: int = 0
    total_dispatch_count: int = 0
    batch_count: int = 0
    task_seconds: float = 0.0
    wall_seconds: float = 0.0
    context_tokens_per_request: list[int] = field(default_factory=list)
    speedup: float = 1.0
    efficiency: float = 0.0

    @property
    def streaming_dispatch_ratio(self) -> float:
        if self.total_dispatch_count == 0:
            return 0.0
        return self.streaming_dispatch_count / self.total_dispatch_count

    @property
    def conflict_rate(self) -> float:
        if self.total_dispatch_count == 0:
            return 0.0
        return self.conflict_count / self.total_dispatch_count

    @property
    def planning_stall_ratio(self) -> float:
        if self.wall_seconds == 0:
            return 0.0
        return self.planning_stall_seconds / self.wall_seconds

    def as_dict(self) -> dict[str, Any]:
        return {
            "peak_parallelism": self.peak_parallelism,
            "avg_parallelism": round(self.avg_parallelism, 3),
            "conflict_count": self.conflict_count,
            "conflict_serialization_seconds": round(self.conflict_serialization_seconds, 4),
            "planning_stall_seconds": round(self.planning_stall_seconds, 4),
            "streaming_dispatch_ratio": round(self.streaming_dispatch_ratio, 3),
            "batch_count": self.batch_count,
            "task_seconds": round(self.task_seconds, 3),
            "wall_seconds": round(self.wall_seconds, 3),
            "speedup": round(self.speedup, 3),
            "efficiency": round(self.efficiency, 3),
            "conflict_rate": round(self.conflict_rate, 3),
            "planning_stall_ratio": round(self.planning_stall_ratio, 3),
        }


def extract_from_runtime(runtime: Any, controller: Any = None) -> RuntimeMetrics:
    """Extract metrics directly from runtime and controller objects after a run."""
    metrics = RuntimeMetrics()

    timing = runtime.ledger.timing()
    metrics.wall_seconds = timing.wall_seconds
    metrics.task_seconds = timing.task_seconds
    metrics.peak_parallelism = timing.running_tasks

    if metrics.wall_seconds > 0 and metrics.task_seconds > 0:
        metrics.speedup = metrics.task_seconds / metrics.wall_seconds
        metrics.efficiency = metrics.speedup / max(1, runtime.max_workers)

    metrics.conflict_count = len(getattr(runtime.ledger, "_conflicts", []))
    metrics.batch_count = getattr(runtime, "_batch_seq", 0)

    if controller is not None:
        usage = getattr(controller, "usage", None)
        if usage:
            metrics.total_dispatch_count = getattr(usage, "model_calls", 0)

    return metrics


def extract_from_trace(trace_path: str | Path) -> RuntimeMetrics:
    """Extract metrics by parsing a JSONL trace file."""
    path = Path(trace_path)
    if not path.exists():
        return RuntimeMetrics()

    metrics = RuntimeMetrics()
    events: list[dict[str, Any]] = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not events:
        return metrics

    metrics.parallelism_timeline = _build_parallelism_timeline(events)
    if metrics.parallelism_timeline:
        metrics.peak_parallelism = max(count for _, count in metrics.parallelism_timeline)
        total_time = 0.0
        weighted_sum = 0.0
        for i in range(len(metrics.parallelism_timeline) - 1):
            t0, count = metrics.parallelism_timeline[i]
            t1, _ = metrics.parallelism_timeline[i + 1]
            dt = t1 - t0
            weighted_sum += count * dt
            total_time += dt
        if total_time > 0:
            metrics.avg_parallelism = weighted_sum / total_time

    metrics.conflict_count = sum(1 for e in events if e.get("event") == "resource.conflict")
    metrics.planning_stall_seconds = _compute_planning_stall(events)
    metrics.streaming_dispatch_count = _count_streaming_dispatches(events)
    metrics.total_dispatch_count = sum(1 for e in events if e.get("event") == "task.started")
    metrics.batch_count = sum(1 for e in events if e.get("event") == "delivery.delivered")

    task_durations = [
        e.get("duration_seconds", 0.0)
        for e in events
        if e.get("event") == "task.completed" and e.get("status") == "completed"
    ]
    metrics.task_seconds = sum(task_durations)

    started_events = [e for e in events if e.get("event") == "runtime.started"]
    shutdown_events = [e for e in events if e.get("event") == "runtime.shutdown"]
    if started_events and shutdown_events:
        metrics.wall_seconds = shutdown_events[-1].get("ts", 0) - started_events[0].get("ts", 0)

    if metrics.wall_seconds > 0 and metrics.task_seconds > 0:
        metrics.speedup = metrics.task_seconds / metrics.wall_seconds

    return metrics


def _build_parallelism_timeline(events: list[dict[str, Any]]) -> list[tuple[float, int]]:
    """Reconstruct running task count over time from trace events."""
    timeline: list[tuple[float, int]] = []
    running = 0

    for event in events:
        ts = event.get("ts", 0.0)
        event_type = event.get("event", "")

        if event_type == "task.started":
            running += 1
            timeline.append((ts, running))
        elif event_type == "task.completed":
            running = max(0, running - 1)
            timeline.append((ts, running))

    return timeline


def _compute_planning_stall(events: list[dict[str, Any]]) -> float:
    """Compute total time planners spent waiting for delivery."""
    planner_starts: dict[int, float] = {}
    total_stall = 0.0

    for event in events:
        event_type = event.get("event", "")
        ts = event.get("ts", 0.0)

        if event_type == "planner.started":
            seq = event.get("planner_seq", 0)
            planner_starts[seq] = ts
        elif event_type == "planner.completed":
            seq = event.get("planner_seq", 0)
            start = planner_starts.pop(seq, None)
            if start is not None:
                # Planner duration itself is not stall; stall is time between
                # delivery.delivered and next planner.started
                pass

    # Alternative: measure gaps between delivery and next planner start
    delivery_times = [
        e.get("ts", 0.0) for e in events if e.get("event") == "delivery.delivered"
    ]
    planner_start_times = sorted(
        e.get("ts", 0.0) for e in events if e.get("event") == "planner.started"
    )

    for d_time in delivery_times:
        next_planner = None
        for p_time in planner_start_times:
            if p_time > d_time:
                next_planner = p_time
                break
        if next_planner is not None:
            stall = next_planner - d_time
            if stall > 0.001:
                total_stall += stall

    return total_stall


def _count_streaming_dispatches(events: list[dict[str, Any]]) -> int:
    """Count tool calls that were dispatched via streaming early dispatch."""
    count = 0
    for event in events:
        if event.get("event") == "planner.completed":
            # Early dispatched IDs are tracked in the planner result
            # but not directly in trace. Count task.started events that
            # occur before their planner.completed event.
            pass
    # Heuristic: count task.started events that occur between
    # planner.started and planner.completed for the same planner_seq
    planner_windows: list[tuple[float, float]] = []
    starts: dict[int, float] = {}
    for event in events:
        ts = event.get("ts", 0.0)
        if event.get("event") == "planner.started":
            starts[event.get("planner_seq", 0)] = ts
        elif event.get("event") == "planner.completed":
            seq = event.get("planner_seq", 0)
            start = starts.pop(seq, None)
            if start is not None:
                planner_windows.append((start, ts))

    for event in events:
        if event.get("event") != "task.started":
            continue
        ts = event.get("ts", 0.0)
        for window_start, window_end in planner_windows:
            if window_start < ts < window_end:
                count += 1
                break

    return count
