"""Statistical aggregation for multi-run benchmark evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from benchmark.evaluators import CategoryReport, MetricResult, TaskEvaluation


# t-distribution critical values for 95% CI (two-tailed), df=1..30
_T_TABLE_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    25: 2.060, 30: 2.042,
}


def _t_critical(df: int, confidence: float = 0.95) -> float:
    """Lookup t-distribution critical value for given degrees of freedom."""
    if confidence != 0.95:
        return 1.96
    if df in _T_TABLE_95:
        return _T_TABLE_95[df]
    for key in sorted(_T_TABLE_95.keys()):
        if key >= df:
            return _T_TABLE_95[key]
    return 1.96


@dataclass
class AggregatedMetric:
    """Statistical summary of a single metric across multiple runs."""

    name: str
    mean: float
    stddev: float
    ci_low: float
    ci_high: float
    n: int
    unit: str = ""
    higher_is_better: bool = True
    raw_values: list[float] = field(default_factory=list)

    def format_short(self) -> str:
        if self.stddev < 0.001:
            return f"{self.mean:.3f}"
        return f"{self.mean:.3f}±{self.stddev:.3f}"

    def format_latex(self) -> str:
        if self.stddev < 0.001:
            return f"{self.mean:.2f}"
        return f"{self.mean:.2f}$\\pm${self.stddev:.2f}"


@dataclass
class AggregatedTaskEvaluation:
    """Aggregated evaluation for a single task across runs."""

    task_id: str
    category: str
    metrics: dict[str, AggregatedMetric] = field(default_factory=dict)
    pass_rate: float = 0.0
    n: int = 0


@dataclass
class AggregatedCategoryReport:
    """Aggregated report for one category across multiple runs."""

    category: str
    agent_name: str
    profile: str
    n_runs: int = 0
    aggregated_metrics: dict[str, AggregatedMetric] = field(default_factory=dict)
    per_task: list[AggregatedTaskEvaluation] = field(default_factory=list)
    pass_rate: AggregatedMetric | None = None
    per_run_reports: list[CategoryReport] = field(default_factory=list)


def aggregate_values(values: list[float], confidence: float = 0.95) -> tuple[float, float, float, float]:
    """Compute mean, stddev, ci_low, ci_high for a list of values."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0, mean, mean
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    stddev = math.sqrt(variance)
    t_crit = _t_critical(n - 1, confidence)
    margin = t_crit * stddev / math.sqrt(n)
    return mean, stddev, mean - margin, mean + margin


def aggregate_metric_values(
    name: str,
    values: list[float],
    *,
    unit: str = "",
    higher_is_better: bool = True,
    confidence: float = 0.95,
) -> AggregatedMetric:
    """Create an AggregatedMetric from raw values."""
    mean, stddev, ci_low, ci_high = aggregate_values(values, confidence)
    return AggregatedMetric(
        name=name,
        mean=mean,
        stddev=stddev,
        ci_low=ci_low,
        ci_high=ci_high,
        n=len(values),
        unit=unit,
        higher_is_better=higher_is_better,
        raw_values=list(values),
    )


def aggregate_category_reports(
    reports: list[CategoryReport],
    profile: str = "",
) -> AggregatedCategoryReport:
    """Aggregate multiple runs of the same category into statistical summaries."""
    if not reports:
        return AggregatedCategoryReport(category="", agent_name="", profile=profile)

    category = reports[0].category
    agent_name = reports[0].agent_name
    n_runs = len(reports)

    # Aggregate summary metrics
    all_metric_names: set[str] = set()
    for report in reports:
        all_metric_names.update(report.summary_metrics.keys())

    aggregated_metrics: dict[str, AggregatedMetric] = {}
    for metric_name in sorted(all_metric_names):
        values = [
            report.summary_metrics[metric_name]
            for report in reports
            if metric_name in report.summary_metrics
        ]
        if values:
            aggregated_metrics[metric_name] = aggregate_metric_values(
                metric_name, values
            )

    # Aggregate pass rate
    pass_rates = [report.pass_rate for report in reports]
    pass_rate_agg = aggregate_metric_values("pass_rate", pass_rates, higher_is_better=True)

    # Aggregate per-task metrics
    task_ids: list[str] = []
    if reports[0].evaluations:
        task_ids = [ev.task_id for ev in reports[0].evaluations]

    per_task: list[AggregatedTaskEvaluation] = []
    for task_id in task_ids:
        task_metrics: dict[str, list[float]] = {}
        task_passed: list[float] = []

        for report in reports:
            for ev in report.evaluations:
                if ev.task_id != task_id:
                    continue
                task_passed.append(1.0 if ev.passed else 0.0)
                for m in ev.metrics:
                    task_metrics.setdefault(m.name, []).append(m.value)

        agg_task = AggregatedTaskEvaluation(
            task_id=task_id,
            category=category,
            n=n_runs,
            pass_rate=sum(task_passed) / len(task_passed) if task_passed else 0.0,
        )
        for m_name, m_values in task_metrics.items():
            agg_task.metrics[m_name] = aggregate_metric_values(m_name, m_values)
        per_task.append(agg_task)

    return AggregatedCategoryReport(
        category=category,
        agent_name=agent_name,
        profile=profile,
        n_runs=n_runs,
        aggregated_metrics=aggregated_metrics,
        per_task=per_task,
        pass_rate=pass_rate_agg,
        per_run_reports=reports,
    )
