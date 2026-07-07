"""Category A: Parallel Scheduling Efficiency evaluator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmark.adapters import TaskTrace
from benchmark.evaluators import Evaluator, MetricResult, TaskEvaluation


class ParallelEvaluator(Evaluator):
    """Evaluates parallel scheduling efficiency.

    Measures how well the runtime exploits task-level parallelism
    given a known dependency graph.
    """

    @property
    def category(self) -> str:
        return "parallel"

    def evaluate_task(self, trace: TaskTrace, task_spec: dict[str, Any]) -> TaskEvaluation:
        metrics: list[MetricResult] = []

        serial_baseline = float(task_spec.get("serial_baseline_seconds", 0))
        optimal_parallel = float(task_spec.get("optimal_parallel_seconds", 0))
        expected_parallelism = float(task_spec.get("expected_max_parallelism", 1))
        total_task_count = int(task_spec.get("total_subtasks", 1))

        wall_time = trace.wall_time

        # Wall-clock — the only ground-truth comparable across frameworks.
        # Cross-framework speedups are computed downstream from this.
        metrics.append(MetricResult(
            name="wall_time_seconds",
            value=round(wall_time, 3),
            unit="s",
            higher_is_better=False,
        ))

        # Speedup vs synthetic serial baseline (kept for backward compatibility,
        # but note it under-counts real LLM latency — use cross-framework speedup
        # for rigorous comparisons).
        if wall_time > 0 and serial_baseline > 0:
            speedup = serial_baseline / wall_time
            metrics.append(MetricResult(
                name="speedup_ratio",
                value=round(speedup, 3),
                unit="x",
                higher_is_better=True,
                details={"serial_baseline": serial_baseline, "actual": wall_time},
            ))

        # Parallelism utilization: actual_speedup / theoretical_max_speedup
        if optimal_parallel > 0 and serial_baseline > 0:
            theoretical_max_speedup = serial_baseline / optimal_parallel
            actual_speedup = serial_baseline / wall_time if wall_time > 0 else 0
            utilization = min(1.0, actual_speedup / theoretical_max_speedup) if theoretical_max_speedup > 0 else 0
            metrics.append(MetricResult(
                name="parallelism_utilization",
                value=round(utilization, 3),
                unit="ratio",
                higher_is_better=True,
                details={"theoretical_max": theoretical_max_speedup, "actual": actual_speedup},
            ))

        # Scheduling overhead: (actual - optimal) / actual
        if wall_time > 0 and optimal_parallel > 0:
            overhead = max(0, (wall_time - optimal_parallel) / wall_time)
            metrics.append(MetricResult(
                name="scheduling_overhead",
                value=round(overhead, 3),
                unit="ratio",
                higher_is_better=False,
            ))

        # Conflict detection accuracy (from metadata if available)
        expected_conflicts = task_spec.get("expected_conflicts", [])
        actual_conflicts = trace.metadata.get("detected_conflicts", [])
        if expected_conflicts:
            expected_set = set(tuple(c) for c in expected_conflicts)
            actual_set = set(tuple(c) for c in actual_conflicts) if actual_conflicts else set()
            tp = len(expected_set & actual_set)
            fp = len(actual_set - expected_set)
            fn = len(expected_set - actual_set)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            metrics.append(MetricResult(
                name="conflict_detection_f1",
                value=round(f1, 3),
                unit="f1",
                higher_is_better=True,
            ))

        # Task completion correctness
        expected_outputs = task_spec.get("expected_outputs", {})
        passed = trace.success
        if expected_outputs:
            passed = self._check_outputs(trace, expected_outputs)

        return TaskEvaluation(
            task_id=trace.task_id,
            category=self.category,
            metrics=metrics,
            passed=passed,
        )

    def _check_outputs(self, trace: TaskTrace, expected: dict[str, Any]) -> bool:
        if not trace.success:
            return False
        # Prefer the workspace path injected by the runner — it's the only
        # cross-framework reliable anchor (high_agent's tool_calls don't
        # surface in the trace, hermes/opencode use different field names).
        workspace = trace.metadata.get("workspace")
        if not workspace:
            for tc in trace.tool_calls:
                args = tc.arguments if isinstance(tc.arguments, dict) else {}
                for key in ("path", "filePath", "file_path"):
                    p = args.get(key)
                    if isinstance(p, str) and "/" in p:
                        workspace = str(Path(p).parent)
                        break
                if workspace:
                    break

        for key, value in expected.items():
            if key == "files_created":
                missing = []
                for fname in value:
                    fpath = Path(fname)
                    if fpath.is_absolute():
                        if not fpath.exists():
                            missing.append(fname)
                    elif workspace and (Path(workspace) / fname).exists():
                        continue
                    else:
                        # Fallback: scan tool_calls for matching filename.
                        found = False
                        for tc in trace.tool_calls:
                            args = tc.arguments if isinstance(tc.arguments, dict) else {}
                            for k in ("path", "filePath", "file_path"):
                                p = args.get(k)
                                if isinstance(p, str) and Path(p).name == Path(fname).name:
                                    found = True
                                    break
                            if found:
                                break
                        if not found:
                            missing.append(fname)
                if missing:
                    return False
            elif key == "final_contains":
                if not all(v in trace.final_answer for v in value):
                    return False
        return True
