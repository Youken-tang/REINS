"""runner — REINS-Bench CLI dispatcher.

 Three top-level modes, glued through one
``argparse`` parser so all three speak the same prompt-corpus / output
conventions::

    runner run --system reins        --prompt w_scaffold_001 --runs 3
    runner run --system custom --adapter mypath/adapter.py --prompt w_scaffold_001
    runner replay --trace results/T1d_xxx/trace.jsonl --scheduler reins
    runner score --root results/ --system reins [--baseline-root ...]

The three modes share only a corpus loader (``schema.load_corpus``)
and the cell-record builder (``scoring.cell_record_from_artifacts``).
``run`` and ``replay`` produce per-cell directories on disk; ``score``
folds an existing tree into a ``report.json``.

The dispatcher is intentionally thin: each mode delegates to a helper
in this module so the CLI surface and the programmatic surface stay
aligned. Adapters loaded via ``--adapter mypath/adapter.py`` need to
expose either a ``SchedulerAdapter``-shaped object as
``ADAPTER`` / ``adapter`` / a callable ``build()`` returning one.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from .schema import (  # noqa: E402
    TaskSpec,
    discover_task_specs,
    filter_corpus,
    load_corpus,
    load_task_spec,
)
from .scoring import (  # noqa: E402
    CellRecord,
    build_report,
    cell_record_from_artifacts,
    discover_cells,
    load_records_from_root,
)


# ───────────────────────────── Shared types ─────────────────────────────────


@dataclass
class RunOptions:
    """Knobs threaded through all three modes (run/replay/score)."""

    runs_per_prompt: int = 3
    max_workers: int = 8
    delivery_debounce: float = 0.05
    max_iterations: int = 40
    max_planner_requests: int = 4
    delivery_timeout: float = 30.0
    strict_nogil: bool = True
    run_tests: bool = False
    timeout_per_command: float = 60.0


# ───────────────────────────── corpus loading ────────────────────────────────


def _default_corpus_root() -> Path:
    return _REPO / "benchmark" / "reins_bench" / "prompts"


def _load_filtered_corpus(
    corpus_root: Path,
    *,
    group: str | None = None,
    ids: Sequence[str] | None = None,
    languages: Sequence[str] | None = None,
) -> list[TaskSpec]:
    if corpus_root.is_file():
        specs = [load_task_spec(corpus_root)]
    else:
        specs = load_corpus(corpus_root)
    return filter_corpus(
        specs,
        group=group,
        ids=ids,
        languages=languages,
    )


# ───────────────────────────── `run` mode ───────────────────────────────────


@dataclass
class CorpusEntry:
    """Legacy corpus-entry shape consumed by older driver code."""

    id: str
    category: str
    language: str
    title: str
    prompt: str
    expected_min_tools: int = 5
    notes: str = ""


def _entry_from_spec(spec: TaskSpec) -> Any:
    """Adapt a TaskSpec into the legacy CorpusEntry shape (id / category /
    language / title / prompt) still consumed by some driver code.
    """
    return CorpusEntry(
        id=spec.id,
        category=spec.group,
        language=(spec.languages[0] if spec.languages else "python"),
        title=spec.title,
        prompt=spec.prompt,
        expected_min_tools=5,
        notes=str(spec.source_path or ""),
    )


def _load_custom_adapter(adapter_path: Path) -> Any:
    """Import a third-party adapter module and pluck its SchedulerAdapter.

    The module may expose any of:

    - ``ADAPTER`` (preferred): an instance ready to use
    - ``adapter``: same, lowercase
    - ``build()``: callable returning an adapter

    Anything else is rejected.
    """
    if not adapter_path.exists():
        raise FileNotFoundError(f"adapter file {adapter_path} does not exist")
    spec = importlib.util.spec_from_file_location(
        f"reins_bench_custom_{adapter_path.stem}", adapter_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load adapter from {adapter_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for attr in ("ADAPTER", "adapter"):
        candidate = getattr(module, attr, None)
        if candidate is not None:
            return candidate
    builder = getattr(module, "build", None)
    if callable(builder):
        return builder()
    raise AttributeError(
        f"{adapter_path} must define ADAPTER / adapter / build() to be usable"
    )


def _resolve_system_adapter(system: str, adapter_path: Path | None) -> Any:
    """Map a ``--system`` flag to a concrete SchedulerAdapter instance.

    The built-in ``reins`` / ``sequential`` / ``ray`` adapter modules are
    not bundled with this open-source release; use ``--system custom
    --adapter path/to/adapter.py`` to plug in your own SchedulerAdapter.
    """
    if system == "custom":
        if not adapter_path:
            raise ValueError("--system custom requires --adapter path/to/file.py")
        return _load_custom_adapter(adapter_path)
    if system in {"reins", "sequential", "ray"}:
        raise NotImplementedError(
            f"--system {system!r} is not bundled in this release; "
            "use --system custom --adapter path/to/adapter.py"
        )
    raise ValueError(f"unknown --system {system!r}")


def execute_run_mode(
    *,
    specs: Sequence[TaskSpec],
    adapter: Any,
    model_client: Any,
    out_root: Path,
    options: RunOptions,
    on_progress: Callable[..., None] | None = None,
) -> list[Any]:
    """Iterate (spec × run) calling ``adapter.execute`` for each cell."""
    out_root.mkdir(parents=True, exist_ok=True)
    cells: list[Any] = []
    for spec in specs:
        entry = _entry_from_spec(spec)
        for run in range(options.runs_per_prompt):
            cell = adapter.execute(
                entry=entry,
                model_client=model_client,
                run=run,
                out_root=out_root,
                max_workers=options.max_workers,
                delivery_debounce=options.delivery_debounce,
                max_iterations=options.max_iterations,
                max_planner_requests=options.max_planner_requests,
                delivery_timeout=options.delivery_timeout,
                strict_nogil=options.strict_nogil,
            )
            cells.append(cell)
            if on_progress:
                on_progress(spec, run, cell)
    return cells


# ───────────────────────────── `replay` mode ─────────────────────────────────


def execute_replay_mode(
    *,
    trace_path: Path,
    scheduler: str,
    out_root: Path,
) -> Path:
    """Replay one trace through an alternate scheduler.

    The replay loop reuses ``benchmark.scripts.parse_trace.extract_tool_calls``
    which already knows how to extract a tool sequence from a trace. The
    bench-side replay is intentionally a thin wrapper: it lets a caller
    re-evaluate the *same* tool sequence under a different scheduler
    configuration (resource_aware off, debounce 0, etc.) without burning
    fresh model tokens. The scheduler runners themselves live in
    ``benchmark/scripts/parse_trace.py``; when alternate
    schedulers (``reins-no-resource``, etc.) are wired there, they
    register here through ``_REPLAY_RUNNERS``.
    """
    from benchmark.scripts.parse_trace import (
        extract_tool_calls,
        load_trace,
        replay_tool_calls,
    )

    if not trace_path.exists():
        raise FileNotFoundError(trace_path)
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"replay_{scheduler}_{trace_path.stem}.jsonl"

    runner = _resolve_replay_runner(scheduler)
    events = load_trace(trace_path)
    calls = extract_tool_calls(events)
    replayed = replay_tool_calls(events, runner=runner)
    out_path.write_text(
        "\n".join(json.dumps(_replay_event(call, ret), ensure_ascii=False)
                  for call, ret in zip(calls, replayed)) + "\n",
        encoding="utf-8",
    )
    return out_path


def _replay_event(call: Any, ret: Any) -> dict[str, Any]:
    """Serialise one replayed call (call-extracted shape + runner return)."""
    return {
        "task_id": getattr(call, "task_id", ""),
        "kind": getattr(call, "kind", ""),
        "parent_id": getattr(call, "parent_id", None),
        "deps": list(getattr(call, "deps", []) or []),
        "resource_access": getattr(call, "resource_access", None),
        "submitted_at": getattr(call, "submitted_at", 0.0),
        "runner_result": ret,
    }


def _replay_passthrough(call: Any) -> dict[str, Any]:
    """Default «do-nothing» replay runner — returns the call's id only.

    Useful as a baseline; alternate runners (resource-aware off, etc.)
    can replace this once lands them.
    """
    return {"replayed": getattr(call, "task_id", "")}


_REPLAY_RUNNERS: dict[str, Callable[[Any], Any]] = {
    "reins": _replay_passthrough,
    "reins-default": _replay_passthrough,
}


def _resolve_replay_runner(scheduler: str) -> Callable[..., Any]:
    """Return a replay-time scheduler runner callable.

    Currently only the «reins» pass-through runner is registered;
    alternate schedulers (``reins-no-resource`` etc.) will register
    here as lands them.
    """
    runner = _REPLAY_RUNNERS.get(scheduler)
    if runner is None:
        raise ValueError(
            f"replay scheduler {scheduler!r} not registered; "
            f"known: {sorted(_REPLAY_RUNNERS)}. See to add one."
        )
    return runner


# ───────────────────────────── `score` mode ─────────────────────────────────


def execute_score_mode(
    *,
    root: Path,
    specs: Sequence[TaskSpec],
    system: str,
    baseline_root: Path | None = None,
    baseline_system: str = "sequential",
    run_tests: bool = False,
    run_dir_glob: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Walk a results tree and produce the report.json shape."""
    records: list[CellRecord] = load_records_from_root(
        root,
        specs,
        system=system,
        run_dir_glob=run_dir_glob,
        run_tests=run_tests,
    )
    baseline_per_prompt: dict[str, dict[str, Any]] | None = None
    if baseline_root is not None:
        baseline_records = load_records_from_root(
            baseline_root,
            specs,
            system=baseline_system,
            run_tests=run_tests,
        )
        baseline_report = build_report(
            system=baseline_system,
            records=baseline_records,
        )
        baseline_per_prompt = baseline_report["prompts"]
    return build_report(
        system=system,
        records=records,
        baseline_per_prompt=baseline_per_prompt,
    )


# ───────────────────────────── argparse glue ────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="reins_bench.runner",
        description="REINS-Bench CLI: run / replay / score against the public "
        "evaluation set per",
    )
    sub = p.add_subparsers(dest="mode", required=True)

    # ── run
    run_p = sub.add_parser("run", help="execute one or more prompts under a system")
    run_p.add_argument(
        "--system",
        choices=("reins", "ray", "sequential", "custom"),
        default="reins",
    )
    run_p.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help="path to a custom SchedulerAdapter file (required when --system custom)",
    )
    run_p.add_argument(
        "--corpus",
        type=Path,
        default=_default_corpus_root(),
        help="corpus root directory or single .yaml file",
    )
    run_p.add_argument("--prompt", action="append", default=[], help="filter by id (repeatable)")
    run_p.add_argument("--group", choices=("w_fix", "w_scaffold"), default=None)
    run_p.add_argument("--language", action="append", default=[])
    run_p.add_argument("--out", type=Path, default=_REPO / "benchmark/results")
    run_p.add_argument("--runs", type=int, default=3)
    run_p.add_argument("--max-workers", type=int, default=8)
    run_p.add_argument("--delivery-debounce", type=float, default=0.05)
    run_p.add_argument("--max-iterations", type=int, default=40)
    run_p.add_argument("--max-planner-requests", type=int, default=4)
    run_p.add_argument("--delivery-timeout", type=float, default=30.0)
    run_p.add_argument(
        "--strict-nogil",
        action="store_true",
        help="(default off) require Py_GIL_DISABLED=1 at runtime startup",
    )
    run_p.add_argument(
        "--model",
        default=None,
        help="override model (defaults to high_agent config.yaml)",
    )

    # ── replay
    rep_p = sub.add_parser(
        "replay", help="replay a trace through an alternate scheduler"
    )
    rep_p.add_argument("--trace", type=Path, required=True)
    rep_p.add_argument("--scheduler", default="reins")
    rep_p.add_argument("--out", type=Path, default=_REPO / "benchmark/results/replays")

    # ── score
    sc_p = sub.add_parser("score", help="fold a results tree into report.json")
    sc_p.add_argument(
        "--root",
        type=Path,
        required=True,
        help="results directory containing T1d_*/T2a_*/T2b_*/ run dirs",
    )
    sc_p.add_argument(
        "--corpus",
        type=Path,
        default=_default_corpus_root(),
        help="corpus root directory or single .yaml file",
    )
    sc_p.add_argument("--system", default="reins")
    sc_p.add_argument(
        "--baseline-root",
        type=Path,
        default=None,
        help="optional sibling tree whose passing prompts seed scheduler_score",
    )
    sc_p.add_argument("--baseline-system", default="sequential")
    sc_p.add_argument(
        "--run-tests",
        action="store_true",
        help="actually run must_pass_tests commands (default: skip, gate as PASS)",
    )
    sc_p.add_argument(
        "--run-dir-glob",
        action="append",
        default=[],
        help="fnmatch pattern on run-dir names; repeatable. Default: every cell.",
    )
    sc_p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write report.json to this path (default: stdout)",
    )

    return p


def _build_real_model_client(model: str | None) -> Any:
    from high_agent.config import load_config, load_secrets
    from high_agent.llm.client import ModelClient
    from high_agent.llm.providers import resolve_model_config

    config = load_config()
    secrets = load_secrets()
    overrides = {"model": model} if model else None
    resolved = resolve_model_config(config, secrets, cli_overrides=overrides)
    return ModelClient(settings=resolved.settings)


def _emit_report(report: dict[str, Any], out_path: Path | None) -> None:
    text = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False)
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    else:
        sys.stdout.write(text + "\n")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.mode == "run":
        return _main_run(args)
    if args.mode == "replay":
        return _main_replay(args)
    if args.mode == "score":
        return _main_score(args)
    print(f"unknown mode {args.mode!r}", file=sys.stderr)
    return 2


def _main_run(args: argparse.Namespace) -> int:
    specs = _load_filtered_corpus(
        args.corpus,
        group=args.group,
        ids=args.prompt or None,
        languages=args.language or None,
    )
    if not specs:
        print(
            f"no task specs matched under {args.corpus} "
            f"(group={args.group}, prompts={args.prompt}, langs={args.language})",
            file=sys.stderr,
        )
        return 1
    adapter = _resolve_system_adapter(args.system, args.adapter)
    client = _build_real_model_client(args.model)
    options = RunOptions(
        runs_per_prompt=args.runs,
        max_workers=args.max_workers,
        delivery_debounce=args.delivery_debounce,
        max_iterations=args.max_iterations,
        max_planner_requests=args.max_planner_requests,
        delivery_timeout=args.delivery_timeout,
        strict_nogil=args.strict_nogil,
    )

    def _print(spec: TaskSpec, run: int, cell: Any) -> None:
        wall = getattr(getattr(cell, "artifacts", None), "wall_seconds", None)
        wall_str = f" ({wall:.2f}s)" if isinstance(wall, (int, float)) else ""
        print(f"ok  [{args.system}] {spec.id} run{run}{wall_str}")

    cells = execute_run_mode(
        specs=specs,
        adapter=adapter,
        model_client=client,
        out_root=args.out,
        options=options,
        on_progress=_print,
    )
    print(f"{len(cells)} cell(s) written to {args.out}")
    return 0


def _main_replay(args: argparse.Namespace) -> int:
    out_path = execute_replay_mode(
        trace_path=args.trace,
        scheduler=args.scheduler,
        out_root=args.out,
    )
    print(f"replay → {out_path}")
    return 0


def _main_score(args: argparse.Namespace) -> int:
    if args.corpus.is_file():
        specs = [load_task_spec(args.corpus)]
    else:
        specs = load_corpus(args.corpus)
    if not specs:
        print(f"no task specs under {args.corpus}", file=sys.stderr)
        return 1
    if not discover_cells(args.root):
        print(f"no run cells under {args.root}", file=sys.stderr)
        return 1
    report = execute_score_mode(
        root=args.root,
        specs=specs,
        system=args.system,
        baseline_root=args.baseline_root,
        baseline_system=args.baseline_system,
        run_tests=args.run_tests,
        run_dir_glob=args.run_dir_glob or None,
    )
    _emit_report(report, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
