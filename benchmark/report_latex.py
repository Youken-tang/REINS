"""LaTeX and CSV report generation."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

from benchmark.evaluators import CategoryReport
from benchmark.stats import (
    AggregatedCategoryReport,
    AggregatedMetric,
    aggregate_category_reports,
)


def _normalize(
    report: CategoryReport | AggregatedCategoryReport | None,
) -> AggregatedCategoryReport | None:
    """Coerce a single-run CategoryReport into an AggregatedCategoryReport view."""
    if report is None:
        return None
    if isinstance(report, AggregatedCategoryReport):
        return report
    return aggregate_category_reports([report])


def _normalize_results(
    results: dict[str, Any],
) -> dict[str, AggregatedCategoryReport]:
    out: dict[str, AggregatedCategoryReport] = {}
    for key, val in results.items():
        norm = _normalize(val)
        if norm is not None:
            out[key] = norm
    return out


def generate_speedup_table(
    results: dict[str, Any],
    profiles: list[str],
    categories: list[str],
) -> str:
    """Generate LaTeX tabular for baseline speedup comparison.

    Rows = categories, Columns = profiles.
    Cells show mean +/- stddev speedup.
    """
    norm = _normalize_results(results)
    n_profiles = len(profiles)
    col_spec = "l" + "r" * n_profiles
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Speedup comparison across baselines (mean $\pm$ std, $n=5$)}",
        r"\label{tab:speedup}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
    ]

    header = "Category & " + " & ".join(
        _latex_escape(p) for p in profiles
    ) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for cat in categories:
        row_parts = [_latex_escape(cat)]
        for profile in profiles:
            key = f"{profile}_{cat}"
            report = norm.get(key)
            if report and "speedup" in report.aggregated_metrics:
                m = report.aggregated_metrics["speedup"]
                cell = f"${m.mean:.2f} \\pm {m.stddev:.2f}$"
            elif report and "speedup_ratio" in report.aggregated_metrics:
                m = report.aggregated_metrics["speedup_ratio"]
                cell = f"${m.mean:.2f} \\pm {m.stddev:.2f}$"
            else:
                cell = "---"
            row_parts.append(cell)
        lines.append(" & ".join(row_parts) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def generate_ablation_table(
    results: dict[str, Any],
    ablations: list[str],
    categories: list[str],
) -> str:
    """Generate LaTeX tabular for ablation study.

    Shows relative degradation from full system for each ablation.
    """
    norm = _normalize_results(results)
    n_ablations = len(ablations)
    col_spec = "l" + "r" * n_ablations
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablation study: relative speedup degradation (\%)}",
        r"\label{tab:ablation}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
    ]

    header = "Category & " + " & ".join(
        _latex_escape(a.replace("no_", "$-$")) for a in ablations
    ) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for cat in categories:
        full_key = f"high_agent_{cat}"
        full_report = norm.get(full_key)
        full_speedup = 1.0
        if full_report:
            if "speedup" in full_report.aggregated_metrics:
                full_speedup = full_report.aggregated_metrics["speedup"].mean
            elif "speedup_ratio" in full_report.aggregated_metrics:
                full_speedup = full_report.aggregated_metrics["speedup_ratio"].mean

        row_parts = [_latex_escape(cat)]
        for ablation in ablations:
            key = f"{ablation}_{cat}"
            report = norm.get(key)
            if report and full_speedup > 0:
                abl_speedup = 1.0
                if "speedup" in report.aggregated_metrics:
                    abl_speedup = report.aggregated_metrics["speedup"].mean
                elif "speedup_ratio" in report.aggregated_metrics:
                    abl_speedup = report.aggregated_metrics["speedup_ratio"].mean
                degradation = (1.0 - abl_speedup / full_speedup) * 100
                cell = f"${degradation:+.1f}\\%$"
            else:
                cell = "---"
            row_parts.append(cell)
        lines.append(" & ".join(row_parts) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def generate_metrics_table(
    results: dict[str, Any],
    profiles: list[str],
    metric_names: list[str],
    category: str,
) -> str:
    """Generate LaTeX tabular for detailed metrics comparison."""
    norm = _normalize_results(results)
    n_metrics = len(metric_names)
    col_spec = "l" + "r" * n_metrics
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Detailed metrics for \texttt{{{category}}} category}}",
        rf"\label{{tab:metrics_{category}}}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
    ]

    header = "Profile & " + " & ".join(
        _latex_escape(m) for m in metric_names
    ) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for profile in profiles:
        key = f"{profile}_{category}"
        report = norm.get(key)
        row_parts = [_latex_escape(profile)]
        for metric in metric_names:
            if report and metric in report.aggregated_metrics:
                m = report.aggregated_metrics[metric]
                cell = f"${m.mean:.3f}$"
            else:
                cell = "---"
            row_parts.append(cell)
        lines.append(" & ".join(row_parts) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def generate_csv(
    results: dict[str, Any],
) -> str:
    """Generate flat CSV with all metrics for plotting."""
    norm = _normalize_results(results)
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "profile", "category", "metric", "mean", "stddev",
        "ci_low", "ci_high", "n",
    ])

    for key, report in sorted(norm.items()):
        parts = key.rsplit("_", 1)
        if len(parts) == 2:
            profile, category = parts
        else:
            profile = report.agent_name
            category = report.category

        for metric_name, metric in report.aggregated_metrics.items():
            writer.writerow([
                profile,
                category,
                metric_name,
                f"{metric.mean:.6f}",
                f"{metric.stddev:.6f}",
                f"{metric.ci_low:.6f}",
                f"{metric.ci_high:.6f}",
                metric.n,
            ])

    return output.getvalue()


def save_latex_report(
    results: dict[str, Any],
    output_dir: str | Path,
    profiles: list[str] | None = None,
    categories: list[str] | None = None,
) -> dict[str, Path]:
    """Save all LaTeX tables and CSV to output directory."""
    from benchmark.profiles import BASELINES, ABLATIONS

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if profiles is None:
        profiles = list(BASELINES.keys())
    if categories is None:
        categories = sorted(set(
            key.rsplit("_", 1)[1] for key in results if "_" in key
        ))

    ablation_names = list(ABLATIONS.keys())
    files: dict[str, Path] = {}

    speedup_path = out / "table_speedup.tex"
    speedup_path.write_text(
        generate_speedup_table(results, profiles, categories),
        encoding="utf-8",
    )
    files["speedup"] = speedup_path

    ablation_path = out / "table_ablation.tex"
    ablation_path.write_text(
        generate_ablation_table(results, ablation_names, categories),
        encoding="utf-8",
    )
    files["ablation"] = ablation_path

    csv_path = out / "results.csv"
    csv_path.write_text(generate_csv(results), encoding="utf-8")
    files["csv"] = csv_path

    return files


def _latex_escape(text: str) -> str:
    """Escape special LaTeX characters."""
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text
