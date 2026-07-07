"""Category C: Complex Planning & Coordination evaluator."""

from __future__ import annotations

from typing import Any

from benchmark.adapters import TaskTrace
from benchmark.evaluators import Evaluator, MetricResult, TaskEvaluation


class PlanningEvaluator(Evaluator):
    """Evaluates complex planning, decomposition, and coordination.

    Measures plan quality, delegation efficiency, error recovery,
    and end-to-end completion under complex multi-step goals.
    """

    @property
    def category(self) -> str:
        return "planning"

    def evaluate_task(self, trace: TaskTrace, task_spec: dict[str, Any]) -> TaskEvaluation:
        metrics: list[MetricResult] = []

        expected_subtasks = task_spec.get("expected_subtasks", [])
        has_error_injection = task_spec.get("has_error_injection", False)
        expected_outputs = task_spec.get("expected_outputs", {})
        max_expected_time = float(task_spec.get("max_expected_time_seconds", 300))

        passed = self._check_completion(trace, expected_outputs)

        # Plan decomposition coverage
        if expected_subtasks:
            actual_tools = [tc.name for tc in trace.tool_calls]
            actual_args_flat = " ".join(
                str(v) for tc in trace.tool_calls for v in tc.arguments.values()
            )
            covered = 0
            for subtask in expected_subtasks:
                subtask_indicators = subtask.get("indicators", [])
                if any(ind in actual_tools or ind in actual_args_flat for ind in subtask_indicators):
                    covered += 1
            coverage = covered / len(expected_subtasks) if expected_subtasks else 0
            metrics.append(MetricResult(
                name="subtask_coverage",
                value=round(coverage, 3),
                unit="ratio",
                higher_is_better=True,
                details={"covered": covered, "total": len(expected_subtasks)},
            ))

        # Delegation efficiency (high_agent specific)
        delegate_calls = [tc for tc in trace.tool_calls if tc.name == "delegate_task"]
        if delegate_calls:
            total_delegated = sum(
                len(tc.arguments.get("tasks", [])) for tc in delegate_calls
            )
            metrics.append(MetricResult(
                name="delegation_count",
                value=total_delegated,
                unit="tasks",
                higher_is_better=True,
            ))

        # Error recovery (if error injection is present)
        if has_error_injection:
            failed_results = [r for r in trace.tool_results if not r.success]
            if failed_results:
                recovery = 1.0 if passed else 0.0
                metrics.append(MetricResult(
                    name="error_recovery",
                    value=recovery,
                    unit="binary",
                    higher_is_better=True,
                    details={"errors_encountered": len(failed_results)},
                ))

        # Completion time vs budget
        if max_expected_time > 0:
            time_ratio = trace.wall_time / max_expected_time
            metrics.append(MetricResult(
                name="time_budget_ratio",
                value=round(min(time_ratio, 2.0), 3),
                unit="ratio",
                higher_is_better=False,
                details={"actual": trace.wall_time, "budget": max_expected_time},
            ))

        # Model call efficiency
        if trace.model_calls > 0:
            metrics.append(MetricResult(
                name="model_calls",
                value=trace.model_calls,
                unit="calls",
                higher_is_better=False,
            ))

        # Token efficiency
        if trace.total_tokens > 0:
            metrics.append(MetricResult(
                name="total_tokens",
                value=trace.total_tokens,
                unit="tokens",
                higher_is_better=False,
            ))

        return TaskEvaluation(
            task_id=trace.task_id,
            category=self.category,
            metrics=metrics,
            passed=passed,
        )

    def _check_completion(self, trace: TaskTrace, expected: dict[str, Any]) -> bool:
        if not trace.success:
            return False
        if not expected:
            return True

        for key, value in expected.items():
            if key == "files_created":
                created = set()
                for tc in trace.tool_calls:
                    if tc.name == "write_file":
                        created.add(tc.arguments.get("path"))
                    elif tc.name == "write_many_files":
                        for f in tc.arguments.get("files", []):
                            created.add(f.get("path"))
                if not set(value).issubset(created):
                    return False
            elif key == "final_contains":
                if not all(v in trace.final_answer for v in value):
                    return False
            elif key == "directory_structure":
                created_dirs = {tc.arguments.get("path") for tc in trace.tool_calls
                                if tc.name == "mkdir"}
                if not set(value).issubset(created_dirs):
                    return False
        return True
