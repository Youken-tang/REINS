"""Command line interface for high-agent."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

from high_agent import ensure_nogil
from high_agent.agent import (
    MainAgent,
    create_agent_loop_handler,
    create_agent_loop_step_handler,
    create_worker_handler,
)
from high_agent.approval import ApprovalManager
from high_agent.cli.session import InteractiveSession
from high_agent.cli.setup import print_noninteractive_setup_guidance, run_setup
from high_agent.config import get_config_paths, load_config, load_secrets
from high_agent.llm import ModelClient, resolve_model_config
from high_agent.llm.providers import missing_config_reason
from high_agent.runtime import AgentTaskSpec, CausalRuntime, ComponentWrite, TaskResult
from high_agent.tools import ToolRegistry, ToolResultStore, create_core_registry


def runtime_demo() -> int:
    ensure_nogil(strict=True)
    with tempfile.TemporaryDirectory() as tmp:
        trace = Path(tmp) / "trace.jsonl"
        runtime = CausalRuntime(max_workers=3, workspace_root=tmp, trace_path=trace)
        runtime.start()

        def make_dir(ctx):
            target = Path(tmp) / "demo"
            target.mkdir()
            return TaskResult.completed(
                "created demo directory",
                writes=[ComponentWrite(f"dir:{target}", True)],
            )

        def write_file(ctx):
            target = Path(tmp) / "demo" / f"{ctx.task.input['name']}.txt"
            target.write_text(ctx.task.input["content"], encoding="utf-8")
            return TaskResult.completed(
                f"wrote {target.name}",
                writes=[ComponentWrite(f"file:{target}", ctx.task.input["content"])],
            )

        root = AgentTaskSpec(kind="tool", goal="create root", writes={f"dir:{Path(tmp) / 'demo'}"}, handler=make_dir)
        a = AgentTaskSpec(kind="tool", goal="write a", input={"name": "a", "content": "A"}, writes={f"file:{Path(tmp) / 'demo' / 'a.txt'}"}, handler=write_file)
        b = AgentTaskSpec(kind="tool", goal="write b", input={"name": "b", "content": "B"}, writes={f"file:{Path(tmp) / 'demo' / 'b.txt'}"}, handler=write_file)
        runtime.submit([root, a, b])
        while runtime.pending_count():
            batch = runtime.wait_next_delivery(timeout=1)
            if batch:
                print(batch.digest)
        runtime.shutdown()
        print(f"trace: {trace}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="high-agent")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("runtime-demo")

    setup_parser = sub.add_parser("setup")
    setup_parser.add_argument("--non-interactive", action="store_true")
    sub.add_parser("model").add_argument("--non-interactive", action="store_true")

    run_parser = sub.add_parser("run")
    _add_model_args(run_parser)
    run_parser.add_argument("--workspace", default=os.getcwd())
    run_parser.add_argument("--yes", action="store_true", help="allow terminal and outside-workspace file operations")
    run_parser.add_argument("--max-iterations", type=int, default=None)
    run_parser.add_argument("--trace", action="store_true", help="write runtime trace under HIGH_AGENT_HOME/traces")
    run_parser.add_argument("prompt", nargs="*")

    chat_parser = sub.add_parser("chat")
    _add_model_args(chat_parser)
    chat_parser.add_argument("--workspace", default=os.getcwd())
    chat_parser.add_argument("--yes", action="store_true")
    chat_parser.add_argument("--max-iterations", type=int, default=None)
    chat_parser.add_argument("--trace", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "runtime-demo":
        return runtime_demo()
    if args.command in {"setup", "model"}:
        return run_setup(non_interactive=bool(args.non_interactive))
    if args.command == "run":
        prompt = " ".join(args.prompt).strip() or sys.stdin.read().strip()
        if not prompt:
            print("high-agent run requires a prompt", file=sys.stderr)
            return 2
        return run_prompt(args, prompt)
    if args.command == "chat":
        return run_chat(args)
    return 2


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider")
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--api-mode", choices=["chat_completions", "anthropic_messages", "codex_responses"])
    parser.add_argument("--api-key")
    parser.add_argument("--max-workers", type=int)
    parser.add_argument("--model-timeout", type=float, help="model HTTP timeout seconds; default 600")
    parser.add_argument(
        "--planner-stale-seconds",
        type=float,
        help=(
            "controller cap on a single in-flight planner request before it is "
            "abandoned and a fresh planner is dispatched; defaults to --model-timeout"
        ),
    )
    parser.add_argument(
        "--planner-stuck-threshold",
        type=int,
        help=(
            "controller breaker: stop re-dispatching planners against the same "
            "runtime snapshot after this many timeouts (default 3)"
        ),
    )
    parser.add_argument(
        "--agent-loop-max-iterations",
        type=int,
        help=(
            "max LLM iterations per delegate_task(mode='sub_agent') call "
            "before the sub-agent surfaces agent_loop_max_iterations failure "
            "(default 16)"
        ),
    )
    parser.add_argument(
        "--agent-loop-timeout",
        type=float,
        help=(
            "wall-clock seconds a single delegate_task sub-agent may run "
            "before timing out (default 240)"
        ),
    )
    parser.add_argument("--delivery-debounce", type=float, help="runtime delivery debounce seconds")


def run_prompt(args: argparse.Namespace, prompt: str) -> int:
    ensure_nogil(strict=True)
    paths = get_config_paths()
    config = load_config(paths)
    agent_cfg = config.get("agent") if isinstance(config.get("agent"), dict) else {}
    max_iterations = _max_iterations(args, agent_cfg)
    try:
        agent = _build_agent(args, prompt)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    try:
        answer = agent.run(prompt, max_iterations=max_iterations)
        if answer:
            print(answer)
        return 0
    except Exception as exc:
        print(f"运行失败：{exc}", file=sys.stderr)
        return 1
    finally:
        agent.runtime.shutdown()


def run_chat(args: argparse.Namespace) -> int:
    ensure_nogil(strict=True)
    paths = get_config_paths()
    config = load_config(paths)
    agent_cfg = config.get("agent") if isinstance(config.get("agent"), dict) else {}
    max_iterations = _max_iterations(args, agent_cfg)
    session: InteractiveSession

    def _factory(prompt: str) -> MainAgent:
        args.workspace = session.workspace
        args.yes = session.allow_yes
        args.trace_path = session.next_trace_path() if session.trace_enabled else None
        return _build_agent(
            args,
            prompt,
            tools=session.build_tool_registry(),
            approval_manager=session.approval_manager,
            result_store=session.result_store,
        )

    session = InteractiveSession(
        agent_factory=_factory,
        workspace=os.path.abspath(args.workspace),
        allow_yes=bool(args.yes),
        max_iterations=max_iterations,
        trace_enabled=bool(args.trace),
    )
    return session.run()


def _build_agent(
    args: argparse.Namespace,
    objective: str,
    *,
    tools: ToolRegistry | None = None,
    approval_manager: ApprovalManager | None = None,
    result_store: ToolResultStore | None = None,
) -> MainAgent:
    paths = get_config_paths()
    config = load_config(paths)
    secrets = load_secrets(paths)
    resolved = resolve_model_config(
        config,
        secrets,
        cli_overrides={
            "provider": args.provider,
            "model": args.model,
            "base_url": args.base_url,
            "api_mode": args.api_mode,
            "api_key": args.api_key,
        },
    )
    missing = missing_config_reason(resolved)
    if missing:
        print_noninteractive_setup_guidance(file=sys.stderr)
        raise RuntimeError(f"模型配置不可用：{missing}")

    runtime_cfg = config.get("runtime") if isinstance(config.get("runtime"), dict) else {}
    agent_cfg = config.get("agent") if isinstance(config.get("agent"), dict) else {}
    max_workers = int(args.max_workers or os.getenv("HIGH_AGENT_MAX_WORKERS") or runtime_cfg.get("max_workers") or 8)
    max_planner_requests = _max_planner_requests(runtime_cfg)
    max_iterations = _max_iterations(args, agent_cfg)
    delivery_debounce = _delivery_debounce_seconds(args, runtime_cfg)
    critical_path_fanout = int(runtime_cfg.get("critical_path_fanout") or 2)
    critical_path_signal_budget = int(runtime_cfg.get("critical_path_signal_budget") or 16)
    trace_path = getattr(args, "trace_path", None)
    if getattr(args, "trace", False) and not trace_path:
        trace_dir = paths.home / "traces" / time.strftime("%Y%m%d")
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = trace_dir / f"run-{int(time.time() * 1000)}.jsonl"
    runtime = CausalRuntime(
        max_workers=max_workers,
        workspace_root=args.workspace,
        trace_path=trace_path,
        delivery_debounce=delivery_debounce,
        critical_path_fanout=critical_path_fanout,
        critical_path_signal_budget=critical_path_signal_budget,
    )
    model_timeout = _model_timeout_seconds(args, config)
    planner_stale_seconds = _planner_stale_seconds(args, config, runtime_cfg, model_timeout)
    planner_stuck_threshold = _planner_stuck_threshold(args, runtime_cfg)
    model_client = ModelClient(resolved.settings, timeout=model_timeout)
    approval_manager = approval_manager or ApprovalManager(
        policy="auto" if bool(args.yes) else "ask",
        interactive=sys.stdin.isatty(),
    )
    result_store = result_store or ToolResultStore(paths.home / "tool-results")
    registry = tools or create_core_registry(
        allow_terminal=bool(args.yes),
        allow_outside_workspace=bool(args.yes),
        approval_manager=approval_manager,
        result_store=result_store,
    )
    # Wire production task executors. The kind="agent_loop" entry handler and
    # kind="agent_loop_step" iteration handler are required for delegate_task
    # mode="sub_agent" (the schema-facing alias retained for prompt
    # compatibility, v11-C8) to actually run; without them the scheduler falls
    # through to TaskResult.completed(task.goal) and the AgentLoop never
    # executes. v11-C10 retired the legacy kind="sub_agent" /
    # kind="sub_agent_step" registrations now that no lowering path produces
    # them
    #
    # v11-D1: ``create_worker_handler`` now takes the tool_registry so worker
    # turns can lower their tool_calls into runtime children. Without this
    # wiring, ``delegate_task(mode='worker')`` silently dropped any
    # tool_calls the model produced — the trace symptom was empty
    # ``schemas/`` / ``services/`` / ``templates/`` directories under a
    # delegated scaffold.
    #
    # v11-D6: agent_loop budget is configurable. Defaults bumped 8→16
    # iterations and 120s→240s after the 2026-05 e2e run truncated ~138
    # of 187 sub-agent failures at the old ceiling. Override via CLI
    # ``--agent-loop-max-iterations`` / ``--agent-loop-timeout``, env
    # ``HIGH_AGENT_AGENT_LOOP_MAX_ITERATIONS`` /
    # ``HIGH_AGENT_AGENT_LOOP_TIMEOUT_SECONDS``, or
    # ``agent.agent_loop_max_iterations`` /
    # ``agent.agent_loop_timeout_seconds`` in config.yaml.
    agent_loop_max_iterations = _agent_loop_max_iterations(args, agent_cfg)
    agent_loop_timeout = _agent_loop_timeout(args, agent_cfg)
    runtime.executors.setdefault("worker", create_worker_handler(model_client, registry))
    runtime.executors.setdefault(
        "agent_loop",
        create_agent_loop_handler(
            model_client,
            registry,
            max_iterations=agent_loop_max_iterations,
            timeout=agent_loop_timeout,
        ),
    )
    runtime.executors.setdefault(
        "agent_loop_step", create_agent_loop_step_handler(model_client, registry)
    )
    return MainAgent(
        objective,
        runtime,
        model_client=model_client,
        tools=registry,
        max_planner_requests=max_planner_requests,
        planner_stale_seconds=planner_stale_seconds,
        planner_stuck_threshold=planner_stuck_threshold,
        tool_use_enforcement=agent_cfg.get("tool_use_enforcement", "auto"),
    )


def _model_timeout_seconds(args: argparse.Namespace, config: dict) -> float:
    model_cfg = config.get("model") if isinstance(config.get("model"), dict) else {}
    runtime_cfg = config.get("runtime") if isinstance(config.get("runtime"), dict) else {}
    raw = (
        getattr(args, "model_timeout", None)
        or os.getenv("HIGH_AGENT_MODEL_TIMEOUT_SECONDS")
        or os.getenv("HIGH_AGENT_MODEL_TIMEOUT")
        or model_cfg.get("timeout_seconds")
        or runtime_cfg.get("model_timeout_seconds")
        or 600
    )
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = 600.0
    return max(1.0, value)


def _planner_stale_seconds(
    args: argparse.Namespace,
    config: dict,
    runtime_cfg: dict,
    model_timeout: float,
) -> float:
    """Resolve the controller's planner_stale_seconds cap.

    v11-D2: previously hardcoded 120s in ``AgentRunController._cancel_stale_planners``.
    The default now mirrors ``model_timeout`` so the controller does not give
    up on a planner future before the underlying httpx client would. Override
    via CLI ``--planner-stale-seconds``, env ``HIGH_AGENT_PLANNER_STALE_SECONDS``,
    or ``runtime.planner_stale_seconds`` in config.yaml.
    """
    agent_cfg = config.get("agent") if isinstance(config.get("agent"), dict) else {}
    raw = (
        getattr(args, "planner_stale_seconds", None)
        or os.getenv("HIGH_AGENT_PLANNER_STALE_SECONDS")
        or runtime_cfg.get("planner_stale_seconds")
        or agent_cfg.get("planner_stale_seconds")
        or model_timeout
    )
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = float(model_timeout)
    return max(1.0, value)


def _planner_stuck_threshold(
    args: argparse.Namespace,
    runtime_cfg: dict,
) -> int:
    """Resolve the controller's planner_stuck_threshold breaker.

    v11-D3: number of planner timeouts on the same snapshot_seq before the
    controller stops re-dispatching against it. Default 3. Override via
    CLI ``--planner-stuck-threshold``, env ``HIGH_AGENT_PLANNER_STUCK_THRESHOLD``,
    or ``runtime.planner_stuck_threshold``.
    """
    raw = (
        getattr(args, "planner_stuck_threshold", None)
        or os.getenv("HIGH_AGENT_PLANNER_STUCK_THRESHOLD")
        or runtime_cfg.get("planner_stuck_threshold")
        or 3
    )
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = 3
    return max(1, value)


def _delivery_debounce_seconds(args: argparse.Namespace, runtime_cfg: dict) -> float:
    raw = (
        getattr(args, "delivery_debounce", None)
        if getattr(args, "delivery_debounce", None) is not None
        else os.getenv("HIGH_AGENT_DELIVERY_DEBOUNCE_SECONDS")
        or os.getenv("HIGH_AGENT_DELIVERY_DEBOUNCE")
        or runtime_cfg.get("delivery_debounce_seconds")
        or runtime_cfg.get("delivery_debounce")
        or 0.05
    )
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = 0.05
    return max(0.0, value)


def _max_planner_requests(runtime_cfg: dict) -> int:
    raw = (
        os.getenv("HIGH_AGENT_MAX_PLANNER_REQUESTS")
        or runtime_cfg.get("max_planner_requests")
        or 4
    )
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = 4
    return max(1, value)


def _max_iterations(args: argparse.Namespace, agent_cfg: dict) -> int:
    raw = (
        getattr(args, "max_iterations", None)
        or os.getenv("HIGH_AGENT_MAX_ITERATIONS")
        or agent_cfg.get("max_iterations")
        or 200
    )
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = 200
    return max(1, value)


def _agent_loop_max_iterations(args: argparse.Namespace, agent_cfg: dict) -> int:
    """Resolve agent_loop sub-agent iteration ceiling.

    v11-D6: previously hardcoded 8 in ``AgentLoopState`` / ``create_agent_loop_handler``.
    Default raised to 16 (8 was insufficient for multi-file scaffolds —
    see in [agent/loop.py](src/high_agent/agent/loop.py)). Override
    via CLI ``--agent-loop-max-iterations``, env
    ``HIGH_AGENT_AGENT_LOOP_MAX_ITERATIONS``, or
    ``agent.agent_loop_max_iterations`` in config.yaml.
    """
    raw = (
        getattr(args, "agent_loop_max_iterations", None)
        or os.getenv("HIGH_AGENT_AGENT_LOOP_MAX_ITERATIONS")
        or agent_cfg.get("agent_loop_max_iterations")
        or 16
    )
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = 16
    return max(1, value)


def _agent_loop_timeout(args: argparse.Namespace, agent_cfg: dict) -> float:
    """Resolve agent_loop sub-agent wall-clock timeout in seconds.

    v11-D6: previously hardcoded 120s. Default raised to 240s. Override
    via CLI ``--agent-loop-timeout``, env
    ``HIGH_AGENT_AGENT_LOOP_TIMEOUT_SECONDS``, or
    ``agent.agent_loop_timeout_seconds`` in config.yaml.
    """
    raw = (
        getattr(args, "agent_loop_timeout", None)
        or os.getenv("HIGH_AGENT_AGENT_LOOP_TIMEOUT_SECONDS")
        or agent_cfg.get("agent_loop_timeout_seconds")
        or 240.0
    )
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = 240.0
    return max(1.0, value)


if __name__ == "__main__":
    raise SystemExit(main())
