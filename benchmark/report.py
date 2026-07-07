"""Benchmark report generation and display."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmark.evaluators import CategoryReport


def print_report(
    results: dict[str, CategoryReport],
    primary: str,
    compare: str | None = None,
) -> None:
    """Print a formatted comparison report to stdout."""
    print(f"\n{'='*70}")
    print("BENCHMARK RESULTS")
    print(f"{'='*70}")

    categories = set()
    for key in results:
        parts = key.rsplit("_", 1)
        if len(parts) == 2:
            categories.add(parts[1])
        else:
            cat = key.split("_", 1)[1] if "_" in key else key
            categories.add(cat)

    # Extract category from keys more robustly
    agent_categories: dict[str, dict[str, CategoryReport]] = {}
    for key, report in results.items():
        agent = report.agent_name
        cat = report.category
        agent_categories.setdefault(agent, {})[cat] = report

    all_categories = sorted(set(
        cat for agent_cats in agent_categories.values() for cat in agent_cats
    ))

    for cat in all_categories:
        print(f"\n--- {cat.upper()} ---")
        print(f"{'Metric':<30} ", end="")

        agents = sorted(agent_categories.keys())
        for agent in agents:
            print(f"{agent:<20} ", end="")
        print()
        print("-" * (32 + 21 * len(agents)))

        # Collect all metric names from this category
        metric_names: set[str] = set()
        for agent in agents:
            report = agent_categories.get(agent, {}).get(cat)
            if report:
                metric_names.update(report.summary_metrics.keys())

        for metric in sorted(metric_names):
            print(f"{metric:<30} ", end="")
            for agent in agents:
                report = agent_categories.get(agent, {}).get(cat)
                if report and metric in report.summary_metrics:
                    val = report.summary_metrics[metric]
                    print(f"{val:<20.3f} ", end="")
                else:
                    print(f"{'N/A':<20} ", end="")
            print()

    # Overall summary
    print(f"\n{'='*70}")
    print("OVERALL SUMMARY")
    print(f"{'='*70}")
    print(f"{'Agent':<20} {'Pass Rate':<12} {'Avg Efficiency':<16} {'Categories'}")
    print("-" * 60)

    for agent in sorted(agent_categories.keys()):
        cats = agent_categories[agent]
        total_pass = sum(r.pass_rate for r in cats.values())
        avg_pass = total_pass / len(cats) if cats else 0

        efficiencies = []
        for r in cats.values():
            if "step_efficiency" in r.summary_metrics:
                efficiencies.append(r.summary_metrics["step_efficiency"])
            elif "parallelism_utilization" in r.summary_metrics:
                efficiencies.append(r.summary_metrics["parallelism_utilization"])
        avg_eff = sum(efficiencies) / len(efficiencies) if efficiencies else 0

        print(f"{agent:<20} {avg_pass:<12.1%} {avg_eff:<16.3f} {len(cats)}")


def save_report(results: dict[str, CategoryReport], output_path: str) -> None:
    """Serialize results to JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    serialized: dict[str, Any] = {}
    for key, report in results.items():
        serialized[key] = {
            "category": report.category,
            "agent_name": report.agent_name,
            "pass_rate": report.pass_rate,
            "summary_metrics": report.summary_metrics,
            "evaluations": [
                {
                    "task_id": ev.task_id,
                    "passed": ev.passed,
                    "metrics": [
                        {
                            "name": m.name,
                            "value": m.value,
                            "unit": m.unit,
                            "higher_is_better": m.higher_is_better,
                        }
                        for m in ev.metrics
                    ],
                }
                for ev in report.evaluations
            ],
        }

    with open(path, "w") as f:
        json.dump(serialized, f, indent=2, ensure_ascii=False)
