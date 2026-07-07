"""Category: Scheduling micro-benchmark evaluator.

Evaluates pure scheduling efficiency using deterministic workloads
with known optimal parallelism and dependency structures.
"""

from __future__ import annotations

from typing import Any

from benchmark.adapters import TaskTrace
from benchmark.evaluators import Evaluator, MetricResult, TaskEvaluation


class SchedulingEvaluator(Evaluator):
    """Evaluates scheduling layer performance on micro-benchmarks.

    Focuses on: dispatch latency, parallelism exploitation,
    conflict detection accuracy, and scheduling overhead.
    """

    @property
    def category(self) -> str:
        return "scheduling"

    def evaluate_task(self, trace: TaskTrace, task_spec: dict[str, Any]) -> TaskEvaluation:
        metrics: list[MetricResult] = []

        serial_baseline = float(task_spec.get("serial_baseline_seconds", 0))
        optimal_parallel = float(task_spec.get("optimal_parallel_seconds", 0))
        expected_parallelism = int(task_spec.get("expected_max_parallelism", 1))
        pattern = task_spec.get("scheduling_pattern", "unknown")

        wall_time = trace.wall_time

        # Speedup vs serial baseline
        if wall_time > 0 and serial_baseline > 0:
            speedup = serial_baseline / wall_time
            metrics.append(MetricResult(
                name="speedup",
                value=round(speedup, 3),
                unit="x",
                higher_is_better=True,
                details={"serial_baseline": serial_baseline, "actual_wall": wall_time},
            ))

        # Efficiency: speedup / max_workers
        if wall_time > 0 and serial_baseline > 0:
            speedup = serial_baseline / wall_time
            max_workers = trace.metadata.get("max_concurrent", 8)
            efficiency = speedup / max(1, max_workers)
            metrics.append(MetricResult(
                name="efficiency",
                value=round(efficiency, 3),
                unit="ratio",
                higher_is_better=True,
            ))

        # Peak parallelism achieved vs expected
        if trace.peak_parallelism > 0:
            parallelism_ratio = trace.peak_parallelism / max(1, expected_parallelism)
            metrics.append(MetricResult(
                name="parallelism_achieved",
                value=trace.peak_parallelism,
                unit="tasks",
                higher_is_better=True,
                details={
                    "expected": expected_parallelism,
                    "ratio": round(parallelism_ratio, 3),
                },
            ))

        # Scheduling overhead: time beyond optimal
        if wall_time > 0 and optimal_parallel > 0:
            overhead_seconds = max(0, wall_time - optimal_parallel)
            overhead_ratio = overhead_seconds / wall_time
            metrics.append(MetricResult(
                name="scheduling_overhead",
                value=round(overhead_ratio, 3),
                unit="ratio",
                higher_is_better=False,
                details={"overhead_seconds": round(overhead_seconds, 3)},
            ))

        # Conflict detection: false positives cause unnecessary serialization
        if trace.conflict_count > 0:
            expected_conflicts = task_spec.get("expected_conflicts", [])
            expected_count = len(expected_conflicts)
            false_positive_rate = max(0, trace.conflict_count - expected_count) / max(1, trace.conflict_count)
            metrics.append(MetricResult(
                name="conflict_false_positive_rate",
                value=round(false_positive_rate, 3),
                unit="ratio",
                higher_is_better=False,
            ))

        # Batch count (fewer batches = better pipelining)
        if trace.batch_count > 0:
            metrics.append(MetricResult(
                name="batch_count",
                value=trace.batch_count,
                unit="batches",
                higher_is_better=False,
            ))

        # Streaming dispatch ratio
        if trace.total_dispatch_count > 0:
            streaming_ratio = trace.streaming_dispatch_count / trace.total_dispatch_count
            metrics.append(MetricResult(
                name="streaming_dispatch_ratio",
                value=round(streaming_ratio, 3),
                unit="ratio",
                higher_is_better=True,
            ))

        # Task-seconds (total CPU work)
        if trace.task_seconds > 0:
            metrics.append(MetricResult(
                name="task_seconds",
                value=round(trace.task_seconds, 3),
                unit="seconds",
                higher_is_better=False,
            ))

        # Correctness check
        passed = self._check_correctness(trace, task_spec)

        return TaskEvaluation(
            task_id=trace.task_id,
            category=self.category,
            metrics=metrics,
            passed=passed,
            notes=f"pattern={pattern}",
        )

    def _check_correctness(self, trace: TaskTrace, task_spec: dict[str, Any]) -> bool:
        if not trace.success:
            return False

        expected_outputs = task_spec.get("expected_outputs", {})
        if not expected_outputs:
            return True

        files_created_calls = {
            tc.arguments.get("path", tc.arguments.get("file_path", ""))
            for tc in trace.tool_calls
            if tc.name in ("write_file", "create_file")
        }

        if "files_created" in expected_outputs:
            expected_files = set(expected_outputs["files_created"])
            if not expected_files.issubset(files_created_calls):
                return False

        if "files_not_created" in expected_outputs:
            forbidden_files = set(expected_outputs["files_not_created"])
            if forbidden_files & files_created_calls:
                return False

        if "file_contents" in expected_outputs:
            pass

        if "file_contains" in expected_outputs:
            pass

        return True
