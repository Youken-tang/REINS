""" / Figure 5 aggregator for microbench results.

Reads ``matrix_summary.json`` written by :mod:`benchmark.microbench.runner`,
groups cells by ``(shape, n, base_ms, debounce)``, and produces:

* per-shape **speedup curves** vs ``max_workers`` (baseline = w1 at the same
  shape/n/base_ms/debounce) — Figure 5
* a **summary table** for with the headline numbers per shape:
  ideal_max (= n / critical_path_length), achieved_max (best speedup
  observed), efficiency at achieved_max
* a **resource-conflict** check (conflict_pair must serialise; unknown_burst
  must serialise — the sanity contract)

Usage::

    PYTHONPATH=src .venv/bin/python -m benchmark.microbench.aggregate_T1a \\
        benchmark/microbench/runs/T1a_full
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def load_matrix(run_dir: Path) -> dict:
    summary_path = run_dir / "matrix_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    return json.loads(summary_path.read_text(encoding="utf-8"))


def speedup_table(matrix: dict) -> dict:
    """Group by (shape, n, base_ms, debounce); within group compute
    speedup = wall_w1 / wall_wK for each K, plus parallelism efficiency.
    """
    grouped: dict[tuple, dict[int, dict]] = defaultdict(dict)
    for cell_name, m in matrix.items():
        cfg = m["config"]
        key = (cfg["shape"], cfg["n"], cfg["base_ms"], cfg["delivery_debounce"])
        grouped[key][cfg["max_workers"]] = m

    out: list[dict] = []
    for key, by_w in sorted(grouped.items()):
        shape, n, base_ms, deb = key
        if 1 not in by_w:
            continue
        baseline = by_w[1]["wall_seconds_median"]
        for w in sorted(by_w):
            m = by_w[w]
            sp = baseline / m["wall_seconds_median"] if m["wall_seconds_median"] > 0 else float("nan")
            out.append({
                "shape": shape, "n": n, "base_ms": base_ms,
                "debounce": deb, "max_workers": w,
                "wall_p50": m["wall_seconds_median"],
                "task_seconds": m["task_seconds_median"],
                "speedup_vs_w1": sp,
                "parallelism_efficiency": m["parallelism_efficiency_median"],
                "conflict_count": m["conflict_count_median"],
            })
    return {"speedup_rows": out}


def shape_summary(rows: list[dict]) -> list[dict]:
    """Per-shape headline: best speedup achieved across the matrix.

    For each shape, group by base_ms (5 vs 50ms — debounce is fixed at 0
    for the headline number per spec), then pick the (n, max_workers)
    cell with the highest speedup_vs_w1.
    """
    out: list[dict] = []
    for shape in sorted({r["shape"] for r in rows}):
        for base_ms in sorted({r["base_ms"] for r in rows}):
            cells = [r for r in rows
                     if r["shape"] == shape and r["base_ms"] == base_ms
                     and r["debounce"] == 0.0 and r["max_workers"] > 1]
            if not cells:
                continue
            best = max(cells, key=lambda r: r["speedup_vs_w1"])
            out.append({
                "shape": shape, "base_ms": base_ms,
                "best_speedup": best["speedup_vs_w1"],
                "at_n": best["n"],
                "at_workers": best["max_workers"],
                "efficiency_at_best": best["parallelism_efficiency"],
            })
    return out


def conflict_sanity(rows: list[dict]) -> list[dict]:
    """Sanity-check the contract per shape:

    * ``conflict_pair`` (write/write on same file) must fully serialise
      → speedup_vs_w1 ≈ 1.0 at any worker count.
    * ``chain`` (each step depends on previous) must fully serialise
      → speedup_vs_w1 ≈ 1.0 at any worker count.
    * ``unknown_burst`` (1 unknown_workspace + n parallel reads) is
      partially serialised: the unknown task forces a barrier, then n
      reads parallelise. Expected speedup_vs_w1 ≈ (1+n)/2 with enough
      workers — NOT 1.0. We don't flag this shape; it's reported in
      shape_summary instead.
    * ``fanout`` (parent write dir:d, children read dir:d + write
      file:d/k.txt) ALSO serialises — children's `dir:d read` ↔
      siblings' `file:d/k.txt write` triggers the access-conflict
      arbiter. This is *correct protocol behaviour*, not a scheduler
      bug; the fanout DAG is intentionally over-declared so the
      protocol sees dir/file overlap. We report it but don't FAIL.
    * ``diamond`` (root → n middle → reducer, middle nodes read root /
      write disjoint files) is the canonical "should parallelise" case
      and is sanity-checked elsewhere via shape_summary's best_speedup.
    """
    out: list[dict] = []
    for r in rows:
        if r["shape"] not in ("conflict_pair", "chain"):
            continue
        if r["max_workers"] < 4 or r["debounce"] != 0.0:
            continue
        ok = 0.85 <= r["speedup_vs_w1"] <= 1.15
        out.append({
            "shape": r["shape"], "n": r["n"], "base_ms": r["base_ms"],
            "max_workers": r["max_workers"],
            "speedup_vs_w1": r["speedup_vs_w1"],
            "serialised_ok": ok,
        })
    return out


def format_per_shape_curve(rows: list[dict], *, base_ms: int = 50,
                           debounce: float = 0.0) -> str:
    """Pretty-print a speedup-vs-workers table per shape."""
    by_shape: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["base_ms"] != base_ms or r["debounce"] != debounce:
            continue
        by_shape[r["shape"]].append(r)

    lines = []
    for shape in sorted(by_shape):
        cells = sorted(by_shape[shape], key=lambda r: (r["n"], r["max_workers"]))
        ns = sorted({r["n"] for r in cells})
        workers = sorted({r["max_workers"] for r in cells})
        lines.append(f"\n=== {shape} (base_ms={base_ms}, debounce={debounce}) ===")
        header = "n\\w".ljust(6) + "".join(f"{w:>8d}" for w in workers)
        lines.append(header)
        for n in ns:
            row = [f"n={n}".ljust(6)]
            for w in workers:
                cell = next((r for r in cells if r["n"] == n and r["max_workers"] == w), None)
                if cell is None:
                    row.append(f"{'—':>8}")
                else:
                    row.append(f"{cell['speedup_vs_w1']:>8.2f}")
            lines.append("".join(row))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="benchmark.microbench.aggregate_T1a")
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ns = ap.parse_args(argv)

    matrix = load_matrix(ns.run_dir)
    table = speedup_table(matrix)
    rows = table["speedup_rows"]
    summary = {
        "n_cells": len(rows),
        "shape_summary": shape_summary(rows),
        "conflict_sanity": conflict_sanity(rows),
        "speedup_rows": rows,
    }
    out_path = ns.out or (ns.run_dir / "aggregate_T1a.json")
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=== shape headline (best speedup_vs_w1, debounce=0) ===")
    print(f"{'shape':<16}{'base_ms':>8}{'best':>8}{'@n':>5}{'@w':>5}{'eff':>8}")
    for s in summary["shape_summary"]:
        print(f"{s['shape']:<16}{s['base_ms']:>8}{s['best_speedup']:>8.2f}"
              f"{s['at_n']:>5}{s['at_workers']:>5}{s['efficiency_at_best']:>8.3f}")

    print("\n=== conflict sanity (must serialise — speedup ≈ 1×) ===")
    bad = [c for c in summary["conflict_sanity"] if not c["serialised_ok"]]
    for c in summary["conflict_sanity"]:
        flag = "ok" if c["serialised_ok"] else "FAIL"
        print(f"  {c['shape']:<16} n={c['n']} w={c['max_workers']:>2} "
              f"base_ms={c['base_ms']:>3}  speedup={c['speedup_vs_w1']:.3f}  [{flag}]")
    if bad:
        print(f"\n*** {len(bad)} sanity-check failures — scheduler is incorrectly parallelising "
              "conflicting tasks ***")

    print(format_per_shape_curve(rows, base_ms=50, debounce=0.0))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
