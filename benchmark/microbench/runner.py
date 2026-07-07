""" microbench runner: drive synthetic DAGs through ``CausalRuntime``
under a configuration matrix and dump per-run metrics.json.

 specifies:

- DAG shapes: chain, fanout, diamond, conflict_pair, unknown_burst
- max_workers ∈ {1, 2, 4, 8, 16, 32}
- delivery_debounce ∈ {0.0, 0.025, 0.100}
- ``base_ms ∈ {5, 50, 500}`` per node
- ``runs`` repetitions per cell, report median + p95

The runner writes one trace.jsonl per (config, run) so
``benchmark/scripts/parse_trace.py`` can compute basic + F1–F8 metrics
offline. A summary metrics.json containing wall_seconds, parallelism
efficiency, ready→running queue depth, and resource.conflict counts is
also written next to the trace.

Entry point per the spec:

    benchmark/runner.py microbench --shape chain --n 8 --workers 8 --runs 5

Each invocation produces:

    results/T1a_<shape>_<n>_<workers>_<debounce>/
        run-0/trace.jsonl
        run-0/metrics.json
        ...
        summary.json   (median across runs)
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

# Make `import high_agent` and `from benchmark...` work when invoked as a
# script from the repo root.
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from high_agent.runtime.scheduler import CausalRuntime  # noqa: E402

from benchmark.microbench.dag_shapes import build_dag  # noqa: E402
from benchmark.scripts.parse_trace import (  # noqa: E402
    compute_F1_F8,
    compute_basic_metrics,
    load_trace,
)


@dataclass
class RunConfig:
    shape: str
    n: int
    base_ms: int
    max_workers: int
    delivery_debounce: float
    critical_path_refill: bool = False

    def label(self) -> str:
        return (
            f"T1a_{self.shape}_n{self.n}_b{self.base_ms}"
            f"_w{self.max_workers}_d{int(self.delivery_debounce * 1000)}"
            f"{'_cp' if self.critical_path_refill else ''}"
        )


@dataclass
class RunMetrics:
    config: dict[str, Any]
    wall_seconds: float
    task_seconds: float
    parallelism_efficiency: float
    worker_idle_ratio: float
    completed_tasks: int
    conflict_count: int
    ready_to_run_p50: float
    ready_to_run_p95: float
    trace_path: str
    extras: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ───────────────────────────── core runner ─────────────────────────────────


def _refill_callback() -> Any:
    def _cb(*_args, **_kwargs) -> None:
        return None

    return _cb


def run_single(
    config: RunConfig,
    *,
    out_dir: Path,
    workspace: Path,
    timeout_seconds: float = 60.0,
) -> RunMetrics:
    """Run a single (config, DAG) cell and return parsed metrics.

    Trace is written to ``out_dir/trace.jsonl``; metrics.json is written
    next to it for downstream figure scripts.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "trace.jsonl"

    runtime = CausalRuntime(
        max_workers=config.max_workers,
        workspace_root=str(workspace),
        delivery_debounce=config.delivery_debounce,
        trace_path=trace_path,
        on_refill_needed=_refill_callback() if config.critical_path_refill else None,
    )
    try:
        dag = build_dag(config.shape, config.n, config.base_ms)
        runtime.submit(dag.tasks)

        deadline = time.monotonic() + timeout_seconds
        seen: set[str] = set()
        while seen != set(dag.task_ids) and time.monotonic() < deadline:
            remaining = max(0.05, deadline - time.monotonic())
            batch = runtime.wait_next_delivery(timeout=remaining)
            if batch is None:
                continue
            for ev in batch.events:
                seen.add(ev.task_id)
        if seen != set(dag.task_ids):
            missing = sorted(set(dag.task_ids) - seen)
            raise TimeoutError(
                f"microbench {config.label()} timed out after {timeout_seconds}s, "
                f"missing {len(missing)}/{len(dag.task_ids)} task(s): {missing[:4]}"
            )
    finally:
        runtime.shutdown()
        runtime.trace.close()

    events = load_trace(trace_path)
    basic = compute_basic_metrics(events, max_workers=config.max_workers)
    findings = compute_F1_F8(events)

    metrics = RunMetrics(
        config=asdict(config),
        wall_seconds=basic.wall_seconds,
        task_seconds=basic.task_seconds,
        parallelism_efficiency=basic.parallelism_efficiency,
        worker_idle_ratio=basic.worker_idle_ratio,
        completed_tasks=basic.task_count,
        conflict_count=int(findings["F2"]["conflict_count"]),
        ready_to_run_p50=float(findings["F4"]["ready_to_run_p50"]),
        ready_to_run_p95=float(findings["F4"]["ready_to_run_p95"]),
        trace_path=str(trace_path),
        extras={
            "F2_conflict_density": findings["F2"]["conflict_density"],
            "F6_llm_rtt_share": findings["F6"]["llm_rtt_share"],
            "max_workers_observed": basic.max_workers_observed,
        },
    )
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics.as_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metrics


def run_matrix(
    configs: Iterable[RunConfig],
    *,
    runs: int,
    out_root: Path,
    workspace: Path,
    timeout_seconds: float = 60.0,
) -> dict[str, dict[str, Any]]:
    """Run each config ``runs`` times; return ``{label: summary}``."""
    summaries: dict[str, dict[str, Any]] = {}
    for cfg in configs:
        cell_dir = out_root / cfg.label()
        cell_dir.mkdir(parents=True, exist_ok=True)
        per_run: list[RunMetrics] = []
        for r in range(runs):
            run_dir = cell_dir / f"run-{r}"
            metrics = run_single(
                cfg,
                out_dir=run_dir,
                workspace=workspace,
                timeout_seconds=timeout_seconds,
            )
            per_run.append(metrics)
        summary = _summarize(cfg, per_run)
        (cell_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        summaries[cfg.label()] = summary
    return summaries


def _summarize(cfg: RunConfig, runs: list[RunMetrics]) -> dict[str, Any]:
    if not runs:
        return {"config": asdict(cfg), "runs": 0}

    def _med(values: list[float]) -> float:
        return float(statistics.median(values)) if values else 0.0

    def _p95(values: list[float]) -> float:
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        idx = int(round((len(sorted_vals) - 1) * 0.95))
        return float(sorted_vals[idx])

    walls = [r.wall_seconds for r in runs]
    tasks = [r.task_seconds for r in runs]
    eff = [r.parallelism_efficiency for r in runs]
    p50 = [r.ready_to_run_p50 for r in runs]
    p95 = [r.ready_to_run_p95 for r in runs]
    conflicts = [r.conflict_count for r in runs]
    return {
        "config": asdict(cfg),
        "runs": len(runs),
        "wall_seconds_median": round(_med(walls), 6),
        "wall_seconds_p95": round(_p95(walls), 6),
        "task_seconds_median": round(_med(tasks), 6),
        "parallelism_efficiency_median": round(_med(eff), 4),
        "ready_to_run_p50_median": round(_med(p50), 6),
        "ready_to_run_p95_median": round(_med(p95), 6),
        "conflict_count_median": int(_med([float(c) for c in conflicts])),
        "completed_tasks": runs[0].completed_tasks,
    }


# ───────────────────────────── matrix expand ───────────────────────────────


_DEFAULT_N = {
    "chain": [8, 16, 32, 64],
    "fanout": [4, 8, 16, 32],
    "diamond": [4, 8, 16],
    "conflict_pair": [2],
    "unknown_burst": [4, 8, 16],
}


def expand_matrix(
    *,
    shapes: list[str],
    sizes: list[int] | None,
    base_ms_list: list[int],
    workers_list: list[int],
    debounce_list: list[float],
    critical_path: bool,
) -> list[RunConfig]:
    out: list[RunConfig] = []
    for shape in shapes:
        n_values = sizes if sizes else _DEFAULT_N.get(shape, [4])
        for n in n_values:
            for base_ms in base_ms_list:
                for workers in workers_list:
                    for debounce in debounce_list:
                        out.append(
                            RunConfig(
                                shape=shape,
                                n=n,
                                base_ms=base_ms,
                                max_workers=workers,
                                delivery_debounce=debounce,
                                critical_path_refill=critical_path,
                            )
                        )
    return out


# ───────────────────────────── CLI ─────────────────────────────────────────


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="microbench",
        description="REINS synthetic-DAG microbench runner",
    )
    p.add_argument(
        "--shape",
        default="chain",
        help="comma-separated list of shapes "
        "(chain,fanout,diamond,conflict_pair,unknown_burst)",
    )
    p.add_argument(
        "--n",
        default=None,
        help="comma-separated list of DAG sizes; default uses the set",
    )
    p.add_argument(
        "--base-ms",
        default="50",
        help="comma-separated list of per-node sleep ms (default 50)",
    )
    p.add_argument(
        "--workers",
        default="4",
        help="comma-separated list of max_workers (default 4)",
    )
    p.add_argument(
        "--debounce",
        default="0",
        help="comma-separated list of delivery_debounce seconds (default 0)",
    )
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--critical-path-refill", action="store_true")
    p.add_argument("--out", type=Path, default=Path("results/"))
    p.add_argument(
        "--timeout-seconds",
        type=float,
        default=60.0,
        help="per-run hard timeout",
    )
    return p


def _split_csv(s: str | None, conv) -> list:
    if s is None:
        return []
    return [conv(x) for x in s.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    shapes = _split_csv(args.shape, str)
    sizes = _split_csv(args.n, int) if args.n else None
    base_ms_list = _split_csv(args.base_ms, int) or [50]
    workers_list = _split_csv(args.workers, int) or [4]
    debounce_list = _split_csv(args.debounce, float) or [0.0]

    configs = expand_matrix(
        shapes=shapes,
        sizes=sizes,
        base_ms_list=base_ms_list,
        workers_list=workers_list,
        debounce_list=debounce_list,
        critical_path=args.critical_path_refill,
    )
    if not configs:
        print("no configs to run", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    workspace = args.out / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    summaries = run_matrix(
        configs,
        runs=args.runs,
        out_root=args.out,
        workspace=workspace,
        timeout_seconds=args.timeout_seconds,
    )
    (args.out / "matrix_summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"wrote {len(summaries)} cell(s) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
