"""TUI layout, panel rendering, and console helpers."""

from __future__ import annotations

import re
import shutil
import sys
import time
from typing import Any, TYPE_CHECKING

from high_agent.time_utils import format_duration_compact

if TYPE_CHECKING:
    from high_agent.agent import MainAgent

PANEL_MAX_LINE_CHARS = 1_200
PANEL_MAX_TEXT_CHARS = 8_000
TASK_PANEL_WIDTH = 70
TASK_PANEL_MIN_COLUMNS = 120
TASK_PANEL_MAX_ROWS = 18
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
LEAKED_ANSI_RE = re.compile(r"\?\[[0-9;]*m")

_RICH_TOKENS = (
    "[bold]", "[/bold]", "[red]", "[/red]", "[dim]", "[/dim]",
    "[green]", "[/green]", "[yellow]", "[/yellow]",
)


def panel_safe_text(text: str) -> str:
    if len(text) > PANEL_MAX_TEXT_CHARS:
        omitted = len(text) - PANEL_MAX_TEXT_CHARS
        text = f"{text[:PANEL_MAX_TEXT_CHARS]}\n... <omitted {omitted} chars>"
    rows: list[str] = []
    for line in text.splitlines() or [""]:
        if len(line) <= PANEL_MAX_LINE_CHARS:
            rows.append(line)
            continue
        omitted = len(line) - PANEL_MAX_LINE_CHARS
        rows.append(f"{line[:PANEL_MAX_LINE_CHARS]} ... <omitted {omitted} chars>")
    return "\n".join(rows)


def task_node(records: list[Any], index: int, *, now: float, width: int) -> str:
    if index >= len(records):
        return ""
    record = records[index]
    task_id = short_task_id(str(getattr(record, "task_id", "?")))
    state = str(getattr(record, "state", "?"))
    if state == "running":
        label = f"{task_id} {format_duration_compact(float(record.run_seconds(now)))}"
    elif state in {"failed", "blocked"}:
        label = f"{task_id} !"
    elif state == "completed":
        label = f"{task_id} {format_duration_compact(float(record.run_seconds(now)))}"
    else:
        label = task_id
    return clip_cell(f"[{label}]", width)


def waiting_edge(record: Any) -> str:
    node = clip_cell(f"[{short_task_id(str(getattr(record, 'task_id', '?')))}]", 12)
    waiting_on = ",".join(str(item) for item in getattr(record, "waiting_on", [])[:2]) or "dependency"
    return f"{node} <- {clip_cell(waiting_on, TASK_PANEL_WIDTH - 18)}"


def failure_node(record: Any, *, now: float) -> str:
    node = task_node([record], 0, now=now, width=12)
    summary = str(getattr(record, "summary", "") or getattr(record, "goal", ""))
    return f"{node} {clip_cell(summary, TASK_PANEL_WIDTH - 15)}"


def short_task_id(task_id: str) -> str:
    if len(task_id) <= 10:
        return task_id
    return task_id[:8]


def clip_cell(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    return text[: max(0, width - 3)] + "..."


def clip_panel_line(line: str) -> str:
    width = _dynamic_panel_width() - 1
    if len(line) <= width:
        return line
    return line[: max(0, width - 3)] + "..."


def _dynamic_panel_width() -> int:
    """Get actual panel width based on terminal size (half of terminal)."""
    cols = terminal_columns()
    half = cols // 2
    return max(TASK_PANEL_WIDTH, half)


def render_task_panel_from_agent(agent: "MainAgent") -> str:
    try:
        ledger = agent.runtime.ledger
        records_func = getattr(ledger, "records_snapshot", None)
        records = list(records_func().values()) if callable(records_func) else []
        counts = ledger.counts()
        timing = ledger.timing() if callable(getattr(ledger, "timing", None)) else None
    except Exception:
        return "parallel status\nunavailable"

    now = time.monotonic()
    total = len(records)
    done_count = counts.get("completed", 0)
    fail_count = counts.get("failed", 0) + counts.get("blocked", 0)
    run_count = counts.get("running", 0)
    wait_count = counts.get("waiting", 0)
    ready_count = counts.get("ready", 0)

    lines: list[str] = []

    if total > 0:
        finished_count = done_count + fail_count
        pct = int(finished_count * 100 / total) if total else 0
        bar_width = _dynamic_panel_width() - 16
        filled = int(bar_width * pct / 100)
        bar = "█" * filled + "░" * (bar_width - filled)
        lines.append(f"{bar} {pct}% ({done_count}/{total})")
    else:
        lines.append("idle")

    if timing is not None:
        planner_info = _get_planner_info(agent)
        parallel_display = f"∥={run_count}" if run_count else (f"planner={planner_info}" if planner_info else "∥=0")
        lines.append(
            f"wall={format_duration_compact(timing.wall_seconds)} "
            f"task={format_duration_compact(timing.task_seconds)} "
            f"{parallel_display}"
        )
    elif run_count:
        lines.append(f"∥={run_count}")

    running = sorted(
        [r for r in records if r.state == "running"],
        key=lambda item: item.started_at or item.created_at,
    )
    waiting = sorted(
        [r for r in records if r.state == "waiting"],
        key=lambda item: item.updated_at,
    )
    failed_blocked = sorted(
        [r for r in records if r.state in {"failed", "blocked"}],
        key=lambda item: item.updated_at,
        reverse=True,
    )
    completed = sorted(
        [r for r in records if r.state == "completed"],
        key=lambda item: item.finished_at or item.updated_at,
        reverse=True,
    )

    if running:
        lines.append("")
        lines.append(f"▶ RUNNING ({len(running)})")
        for r in running[:5]:
            tool = _record_tool_label(r)
            dur = format_duration_compact(r.run_seconds(now))
            lines.append(clip_panel_line(f"  {tool}  {dur}"))

    if waiting:
        lines.append("")
        lines.append(f"⏳ WAITING ({len(waiting)})")
        for r in waiting[:4]:
            tool = _record_tool_label(r)
            deps = ",".join(str(w) for w in getattr(r, "waiting_on", [])[:2]) or "dep"
            lines.append(clip_panel_line(f"  {tool} ← {deps}"))

    if completed and not running:
        lines.append("")
        lines.append("✓ LAST DONE")
        for r in completed[:3]:
            tool = _record_tool_label(r)
            dur = format_duration_compact(r.run_seconds(now))
            lines.append(clip_panel_line(f"  {tool}  {dur}"))

    if failed_blocked:
        lines.append("")
        lines.append(f"✗ FAILED ({len(failed_blocked)})")
        for r in failed_blocked[:3]:
            tool = _record_tool_label(r)
            summary = str(getattr(r, "summary", "") or "")[:40]
            lines.append(clip_panel_line(f"  {tool}: {summary}"))

    if not records:
        lines.append("")
        lines.append("idle")

    if not running and not waiting:
        planner_info = _get_planner_info(agent)
        if planner_info:
            lines.append("")
            lines.append(f"⟳ waiting for model ({planner_info} in-flight)")

    return "\n".join(lines[:TASK_PANEL_MAX_ROWS])


def _get_planner_info(agent: "MainAgent") -> str:
    """Get number of in-flight planner requests from controller if available."""
    try:
        controller = getattr(agent, "_current_controller", None)
        if controller is None:
            return ""
        planner_seq = getattr(controller, "_planner_seq", 0)
        usage = getattr(controller, "usage", None)
        if usage and hasattr(usage, "model_calls"):
            in_flight = planner_seq - usage.model_calls
            if in_flight > 0:
                return str(in_flight)
        return ""
    except Exception:
        return ""


def _record_tool_label(record: Any) -> str:
    """Extract a human-readable tool label from a TaskRecord."""
    goal = str(getattr(record, "goal", "") or "")
    task_id = str(getattr(record, "task_id", "") or "")
    width = _dynamic_panel_width() - 10
    if goal:
        return clip_cell(goal, width)
    if "tool-call" in task_id or "worker" in task_id:
        return clip_cell(task_id[:12], width)
    return clip_cell(task_id, width)


def task_panel_visible(task_panel_enabled: bool) -> bool:
    if not task_panel_enabled:
        return False
    if not sys.stdin.isatty():
        return False
    return terminal_columns() >= TASK_PANEL_MIN_COLUMNS


def terminal_columns() -> int:
    try:
        from prompt_toolkit.application.current import get_app
        app = get_app()
        return int(app.output.get_size().columns)
    except Exception:
        pass
    return shutil.get_terminal_size((120, 24)).columns


def strip_rich(text: str) -> str:
    text = ANSI_RE.sub("", text)
    text = LEAKED_ANSI_RE.sub("", text)
    for token in _RICH_TOKENS:
        text = text.replace(token, "")
    return text


class PlainConsole:
    def print(self, *values: object, **kwargs: object) -> None:
        text = " ".join(str(value) for value in values)
        print(strip_rich(text))


class TuiConsole:
    def __init__(self, append_output: Any) -> None:
        self._append_output = append_output

    def print(self, *values: object, **kwargs: object) -> None:
        text = " ".join(str(value) for value in values)
        self._append_output(strip_rich(text))
