"""Category: Realistic workload evaluator.

Evaluates agent performance on real-world tasks including
project scaffolding, TDD cycles, code analysis, and bug fixes.
"""

from __future__ import annotations

from typing import Any

from benchmark.adapters import TaskTrace
from benchmark.evaluators import Evaluator, MetricResult, TaskEvaluation


class RealisticEvaluator(Evaluator):
    """Evaluates realistic multi-step workloads.

    Measures: correctness, speedup, planning efficiency,
    and tool call economy.
    """

    @property
    def category(self) -> str:
        return "realistic"

    def evaluate_task(self, trace: TaskTrace, task_spec: dict[str, Any]) -> TaskEvaluation:
        metrics: list[MetricResult] = []

        serial_baseline = float(task_spec.get("serial_baseline_seconds", 0))
        workload_type = task_spec.get("workload_type", "unknown")
        wall_time = trace.wall_time

        # Speedup
        if wall_time > 0 and serial_baseline > 0:
            speedup = serial_baseline / wall_time
            metrics.append(MetricResult(
                name="speedup",
                value=round(speedup, 3),
                unit="x",
                higher_is_better=True,
            ))

        # Task-level speedup from trace
        if trace.task_seconds > 0 and wall_time > 0:
            measured_speedup = trace.task_seconds / wall_time
            metrics.append(MetricResult(
                name="measured_speedup",
                value=round(measured_speedup, 3),
                unit="x",
                higher_is_better=True,
            ))

        # Planning stall ratio
        if trace.planning_stall_seconds > 0 and wall_time > 0:
            stall_ratio = trace.planning_stall_seconds / wall_time
            metrics.append(MetricResult(
                name="planning_stall_ratio",
                value=round(stall_ratio, 3),
                unit="ratio",
                higher_is_better=False,
            ))

        # Tool call economy: fewer calls for same result is better
        total_subtasks = int(task_spec.get("total_subtasks", 0))
        if total_subtasks > 0 and trace.step_count > 0:
            call_overhead = trace.step_count / total_subtasks
            metrics.append(MetricResult(
                name="tool_call_overhead",
                value=round(call_overhead, 3),
                unit="ratio",
                higher_is_better=False,
                details={"actual_calls": trace.step_count, "expected_min": total_subtasks},
            ))

        # Model call efficiency
        if trace.model_calls > 0:
            metrics.append(MetricResult(
                name="model_calls",
                value=trace.model_calls,
                unit="calls",
                higher_is_better=False,
            ))

        # Token usage
        if trace.total_tokens > 0:
            metrics.append(MetricResult(
                name="total_tokens",
                value=trace.total_tokens,
                unit="tokens",
                higher_is_better=False,
            ))

        # Peak parallelism
        if trace.peak_parallelism > 0:
            metrics.append(MetricResult(
                name="peak_parallelism",
                value=trace.peak_parallelism,
                unit="tasks",
                higher_is_better=True,
            ))

        # Correctness
        passed = self._check_correctness(trace, task_spec)

        return TaskEvaluation(
            task_id=trace.task_id,
            category=self.category,
            metrics=metrics,
            passed=passed,
            notes=f"type={workload_type}",
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

        if "tests_pass" in expected_outputs:
            test_results = [
                tr for tr in trace.tool_results
                if "pytest" in tr.output or "test" in tr.output.lower()
            ]
            if expected_outputs["tests_pass"]:
                if not any(tr.success for tr in test_results):
                    return False

        if "file_contains" in expected_outputs:
            pass

        return True
