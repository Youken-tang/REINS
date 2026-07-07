"""Evaluator base and shared utilities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from benchmark.adapters import TaskTrace


@dataclass
class MetricResult:
    """Single metric measurement."""
    name: str
    value: float
    unit: str = ""
    higher_is_better: bool = True
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskEvaluation:
    """Evaluation result for a single task."""
    task_id: str
    category: str
    metrics: list[MetricResult] = field(default_factory=list)
    passed: bool = False
    notes: str = ""


@dataclass
class CategoryReport:
    """Aggregated report for one benchmark category."""
    category: str
    agent_name: str
    evaluations: list[TaskEvaluation] = field(default_factory=list)
    summary_metrics: dict[str, float] = field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        if not self.evaluations:
            return 0.0
        return sum(1 for e in self.evaluations if e.passed) / len(self.evaluations)

    def compute_summary(self) -> None:
        if not self.evaluations:
            return
        all_metrics: dict[str, list[float]] = {}
        for ev in self.evaluations:
            for m in ev.metrics:
                all_metrics.setdefault(m.name, []).append(m.value)
        for name, values in all_metrics.items():
            self.summary_metrics[name] = sum(values) / len(values)
        self.summary_metrics["pass_rate"] = self.pass_rate


class Evaluator(ABC):
    """Base evaluator interface."""

    @property
    @abstractmethod
    def category(self) -> str:
        ...

    @abstractmethod
    def evaluate_task(self, trace: TaskTrace, task_spec: dict[str, Any]) -> TaskEvaluation:
        ...

    def evaluate_batch(self, traces: list[TaskTrace], task_specs: list[dict[str, Any]]) -> CategoryReport:
        report = CategoryReport(
            category=self.category,
            agent_name=traces[0].agent_name if traces else "unknown",
        )
        for trace, spec in zip(traces, task_specs):
            evaluation = self.evaluate_task(trace, spec)
            report.evaluations.append(evaluation)
        report.compute_summary()
        return report
