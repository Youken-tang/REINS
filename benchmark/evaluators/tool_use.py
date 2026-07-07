"""Category B: Multi-Step Tool Use evaluator."""

from __future__ import annotations

from typing import Any

from benchmark.adapters import TaskTrace
from benchmark.evaluators import Evaluator, MetricResult, TaskEvaluation


class ToolUseEvaluator(Evaluator):
    """Evaluates multi-step tool use correctness and efficiency.

    Measures whether the agent selects the right tools in the right order,
    and how efficiently it reaches the goal.
    """

    @property
    def category(self) -> str:
        return "tool_use"

    def evaluate_task(self, trace: TaskTrace, task_spec: dict[str, Any]) -> TaskEvaluation:
        metrics: list[MetricResult] = []

        expected_tools = task_spec.get("expected_tool_sequence", [])
        optimal_steps = int(task_spec.get("optimal_steps", len(expected_tools) or 1))
        expected_outputs = task_spec.get("expected_outputs", {})

        # Task success
        passed = self._check_task_success(trace, expected_outputs)

        # Step efficiency: optimal_steps / actual_steps
        actual_steps = trace.step_count
        if actual_steps > 0:
            step_eff = min(1.0, optimal_steps / actual_steps)
            metrics.append(MetricResult(
                name="step_efficiency",
                value=round(step_eff, 3),
                unit="ratio",
                higher_is_better=True,
                details={"optimal": optimal_steps, "actual": actual_steps},
            ))

        # Tool selection F1
        if expected_tools:
            actual_tool_names = [tc.name for tc in trace.tool_calls]
            expected_set = set(expected_tools)
            actual_set = set(actual_tool_names)
            tp = len(expected_set & actual_set)
            fp = len(actual_set - expected_set)
            fn = len(expected_set - actual_set)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            metrics.append(MetricResult(
                name="tool_selection_f1",
                value=round(f1, 3),
                unit="f1",
                higher_is_better=True,
            ))

        # Sequence accuracy: longest common subsequence / expected length
        if expected_tools:
            actual_names = [tc.name for tc in trace.tool_calls]
            lcs_len = self._lcs_length(expected_tools, actual_names)
            seq_acc = lcs_len / len(expected_tools) if expected_tools else 0
            metrics.append(MetricResult(
                name="sequence_accuracy",
                value=round(seq_acc, 3),
                unit="ratio",
                higher_is_better=True,
            ))

        # Token efficiency
        if trace.total_tokens > 0:
            metrics.append(MetricResult(
                name="total_tokens",
                value=trace.total_tokens,
                unit="tokens",
                higher_is_better=False,
            ))

        # Error rate in tool calls
        if trace.tool_results:
            error_count = sum(1 for r in trace.tool_results if not r.success)
            error_rate = error_count / len(trace.tool_results)
            metrics.append(MetricResult(
                name="tool_error_rate",
                value=round(error_rate, 3),
                unit="ratio",
                higher_is_better=False,
            ))

        return TaskEvaluation(
            task_id=trace.task_id,
            category=self.category,
            metrics=metrics,
            passed=passed,
        )

    def _check_task_success(self, trace: TaskTrace, expected: dict[str, Any]) -> bool:
        if not trace.success:
            return False
        if not expected:
            return True

        for key, value in expected.items():
            if key == "files_created":
                created = {tc.arguments.get("path") for tc in trace.tool_calls
                           if tc.name in ("write_file", "write_many_files")}
                if not set(value).issubset(created):
                    return False
            elif key == "files_contain":
                for path, substring in value.items():
                    write_calls = [tc for tc in trace.tool_calls
                                   if tc.name == "write_file" and tc.arguments.get("path") == path]
                    if not write_calls:
                        return False
                    content = write_calls[-1].arguments.get("content", "")
                    if substring not in content:
                        return False
            elif key == "final_contains":
                if not all(v in trace.final_answer for v in value):
                    return False
            elif key == "terminal_ran":
                commands = [tc.arguments.get("command", "") for tc in trace.tool_calls
                            if tc.name == "terminal"]
                if not all(any(v in cmd for cmd in commands) for v in value):
                    return False
        return True

    @staticmethod
    def _lcs_length(seq1: list[str], seq2: list[str]) -> int:
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]
