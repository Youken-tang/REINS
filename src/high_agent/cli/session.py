"""Interactive CLI session."""

from __future__ import annotations

import json
import os
import re
import shutil
import shlex
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any

from high_agent.agent import MainAgent
from high_agent.approval import ApprovalDecision, ApprovalManager, ApprovalRequest
from high_agent.cli.commands import CommandRegistry
from high_agent.config import get_config_paths, load_config
from high_agent.memory import ContextCompressor, MemoryStore, SessionStore
from high_agent.plugins import PluginManager
from high_agent.cli.tui import (
    ANSI_RE,
    LEAKED_ANSI_RE,
    PANEL_MAX_LINE_CHARS,
    PANEL_MAX_TEXT_CHARS,
    TASK_PANEL_MAX_ROWS,
    TASK_PANEL_MIN_COLUMNS,
    TASK_PANEL_WIDTH,
    PlainConsole as _PlainConsole,
    TuiConsole as _TuiConsole,
    clip_cell as _clip_cell,
    clip_panel_line as _clip_panel_line,
    failure_node as _failure_node,
    panel_safe_text as _panel_safe_text,
    render_task_panel_from_agent,
    short_task_id as _short_task_id,
    strip_rich,
    task_node as _task_node,
    task_panel_visible as _task_panel_visible_check,
    terminal_columns as _terminal_columns_fn,
    waiting_edge as _waiting_edge,
)
from high_agent.runtime.types import DeliveryBatch
from high_agent.time_utils import format_duration_compact
from high_agent.tools import ToolRegistry, ToolResultStore, ToolsetRegistry

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.application import Application
    from prompt_toolkit.application.current import get_app
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.filters import Condition
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import ConditionalContainer, HSplit, Layout, VSplit, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.layout.dimension import Dimension
    from prompt_toolkit.layout.menus import CompletionsMenu
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.patch_stdout import patch_stdout
    from prompt_toolkit.widgets import TextArea
except ModuleNotFoundError:  # pragma: no cover - fallback in minimal envs
    Application = None
    Completer = object
    Completion = None
    ConditionalContainer = None
    Condition = None
    FileHistory = None
    FormattedTextControl = None
    KeyBindings = None
    CompletionsMenu = None
    HSplit = None
    Layout = None
    VSplit = None
    Dimension = None
    PromptSession = None
    patch_stdout = None
    TextArea = None
    Window = None
    get_app = None

try:
    from rich.console import Console
except ModuleNotFoundError:  # pragma: no cover - fallback in minimal envs
    Console = None


AgentFactory = Callable[[str], MainAgent]
COMMAND_COMPLETION_MAX_CHARS = 80
PANEL_MAX_LINE_CHARS = 1_200
PANEL_MAX_TEXT_CHARS = 8_000
PASTE_COLLAPSE_CHARS = 8_000
PASTE_COLLAPSE_LINES = 5
TASK_PANEL_WIDTH = 70
TASK_PANEL_MIN_COLUMNS = 120
TASK_PANEL_MAX_ROWS = 18
PASTE_REF_RE = re.compile(r"\[Pasted text #\d+: \d+ lines -> ([^\]]+)\]")
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
LEAKED_ANSI_RE = re.compile(r"\?\[[0-9;]*m")


@dataclass
class PendingApproval:
    request: ApprovalRequest
    event: Event = field(default_factory=Event)
    decision: ApprovalDecision | None = None


@dataclass
class InteractiveSession:
    agent_factory: AgentFactory
    workspace: str
    allow_yes: bool = False
    max_iterations: int = 200
    console: Any = None
    conversation_history: list[dict[str, str]] = field(default_factory=list)
    last_digest: str = ""
    approval_manager: ApprovalManager | None = None
    toolsets: ToolsetRegistry | None = None
    session_store: SessionStore | None = None
    compressor: ContextCompressor | None = None
    memory_store: MemoryStore | None = None
    plugin_manager: PluginManager | None = None
    result_store: ToolResultStore | None = None
    trace_enabled: bool = False
    task_panel_enabled: bool = True
    command_registry: CommandRegistry = field(init=False)
    session_id: str = field(init=False)
    session_usage: dict[str, int] = field(init=False, default_factory=lambda: {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "model_calls": 0,
        "context_estimate": 0,
    })
    task_usage: dict[str, float] = field(init=False, default_factory=lambda: {
        "task_seconds": 0.0,
        "wall_seconds": 0.0,
        "completed_tasks": 0.0,
        "running_tasks": 0.0,
        "last_task_seconds": 0.0,
        "last_wall_seconds": 0.0,
    })
    turn_usage: dict[str, float] = field(init=False, default_factory=lambda: {
        "last_turn_seconds": 0.0,
        "total_turn_seconds": 0.0,
        "turn_count": 0.0,
    })
    last_tasks: str = ""
    last_trace_path: Path | None = None
    _prompt: Any = field(init=False, default=None)
    _application: Any = field(init=False, default=None)
    _input_area: Any = field(init=False, default=None)
    _status_area: Any = field(init=False, default=None)
    _hint_area: Any = field(init=False, default=None)
    _task_panel_area: Any = field(init=False, default=None)
    _task_panel_text: str = field(init=False, default="")
    _transcript_path: Path = field(init=False)
    _tui_busy: bool = field(init=False, default=False)
    _tui_lock: Lock = field(init=False, default_factory=Lock)
    _observability_lock: Lock = field(init=False, default_factory=Lock)
    _approval_lock: Lock = field(init=False, default_factory=Lock)
    _turn_usage_lock: Lock = field(init=False, default_factory=Lock)
    _pending_approval: PendingApproval | None = field(init=False, default=None)
    _paste_counter: int = field(init=False, default=0)
    _suppress_paste_collapse: bool = field(init=False, default=False)
    _usage_snapshots: dict[int, dict[str, int]] = field(init=False, default_factory=dict)
    _task_time_snapshots: dict[int, dict[str, float]] = field(init=False, default_factory=dict)
    _current_turn_started_at: float | None = field(init=False, default=None)
    _current_agent: MainAgent | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.console = self.console or (Console() if Console is not None else _PlainConsole())
        paths = get_config_paths()
        paths.home.mkdir(parents=True, exist_ok=True)
        self.result_store = self.result_store or ToolResultStore(paths.home / "tool-results")
        tui_approval_callback = self._request_tui_approval if sys.stdin.isatty() else None
        self.approval_manager = self.approval_manager or ApprovalManager(
            policy="auto" if self.allow_yes else "ask",
            callback=tui_approval_callback,
            interactive=False if tui_approval_callback else sys.stdin.isatty(),
        )
        self.toolsets = self.toolsets or ToolsetRegistry()
        self._sync_toolset_policy()
        self.session_store = self.session_store or SessionStore(paths.home / "sessions.db")
        self.compressor = self.compressor or ContextCompressor()
        self.memory_store = self.memory_store or MemoryStore(paths.home / "memory.db")
        self.plugin_manager = self.plugin_manager or PluginManager(paths.home / "plugins")
        self.plugin_manager.load()
        self.command_registry = CommandRegistry()
        self._register_builtin_commands()
        self.plugin_manager.register_commands(self.command_registry, self.console)

        session_dir = paths.home / "sessions"
        session_dir.mkdir(parents=True, exist_ok=True)
        title = time.strftime("chat-%Y%m%d-%H%M%S")
        self.session_id = self.session_store.create(title)
        self._transcript_path = session_dir / f"{title}.jsonl"

        history_path = paths.home / "history.txt"
        if PromptSession is not None and FileHistory is not None and sys.stdin.isatty():
            self._prompt = PromptSession(
                history=FileHistory(str(history_path)),
                completer=self._command_completer(),
                complete_while_typing=False,
            )
        if sys.stdin.isatty():
            self._application = self._build_application()

    def run(self) -> int:
        if self._application is not None and self._input_area is not None:
            return self._run_tui_loop()
        self.console.print("[bold]high-agent chat[/bold]  输入 /help 查看命令，Ctrl-D 或 /exit 退出。")
        while True:
            try:
                raw = self._read("> ")
            except EOFError:
                self.console.print("")
                return 0
            text = raw.strip()
            if not text:
                continue
            if text.startswith("/"):
                code = self._handle_command(text)
                if code is not None:
                    return code
                continue
            if text.startswith("!") and self.allow_yes:
                self._run_shell(text[1:].strip())
                continue
            self._run_user_turn(text)

    def build_tool_registry(self) -> ToolRegistry:
        self._sync_toolset_policy()
        registry = self.toolsets.registry()
        self.plugin_manager.register_tools(registry)
        return registry

    def _read(self, prompt: str) -> str:
        if self._application is not None and self._input_area is not None:
            self._refresh_panels()
            self._input_area.buffer.text = ""
            try:
                value = self._application.run()
            except EOFError:
                raise
            except Exception as exc:
                self.console.print(f"[yellow]TUI 输入失败，切换到简易 prompt：{exc}[/yellow]")
                self._application = None
            else:
                return str(value or "")
        if self._prompt is not None:
            return self._prompt.prompt(prompt, bottom_toolbar=self._toolbar)
        return input(prompt)

    def _run_tui_loop(self) -> int:
        old_console = self.console
        self.console = _PlainConsole()
        self.console.print("high-agent chat  输入 /help 查看命令，Ctrl-D 或 /exit 退出。")
        try:
            if patch_stdout is not None:
                with patch_stdout():
                    result = self._application.run()
            else:
                result = self._application.run()
        finally:
            self._cancel_pending_approval()
            self.console = old_console
        return int(result or 0)

    def _toolbar(self) -> str:
        toolsets = ",".join(sorted(self.toolsets.enabled)) if self.toolsets else ""
        session = getattr(self, "session_id", "")[:12] or "new"
        return (
            f"model={self._model_label()} | ctx~{self.session_usage['context_estimate']} tok | "
            f"tok={self.session_usage['input_tokens']}/{self.session_usage['output_tokens']}/"
            f"{self.session_usage['total_tokens']} | calls={self.session_usage['model_calls']} | "
            f"turn={format_duration_compact(self._display_turn_seconds())} | "
            f"task={format_duration_compact(self.task_usage['task_seconds'])} | "
            f"workspace={self.workspace} | yes={self.allow_yes} | "
            f"toolsets={toolsets} | session={session} | trace={'on' if self.trace_enabled else 'off'}"
        )

    def _command_completer(self) -> Any:
        return SlashCommandCompleter(self.command_registry)

    def _build_tui_key_bindings(self) -> Any:
        if KeyBindings is None:
            return None
        bindings = KeyBindings()

        @bindings.add("c-d", is_global=True)
        def _exit(event: Any) -> None:
            event.app.exit(result=0)

        @bindings.add("c-j")
        def _newline(event: Any) -> None:
            event.current_buffer.insert_text("\n")

        @bindings.add("/", eager=True)
        def _slash(event: Any) -> None:
            event.current_buffer.insert_text("/")
            event.current_buffer.start_completion(select_first=False)

        @bindings.add("enter", eager=True, is_global=True)
        def _submit(event: Any) -> None:
            buffer = self._input_area.buffer if self._input_area is not None else event.current_buffer
            self._submit_tui_buffer(buffer, event.app)

        return bindings

    def _accept_tui_buffer(self, buffer: Any) -> bool:
        self._submit_tui_buffer(buffer, get_app())
        return True

    def _submit_tui_buffer(self, buffer: Any, app: Any) -> None:
        text = buffer.text
        buffer.text = ""
        self._submit_tui_text(text, app)

    def _submit_tui_text(self, raw: str, app: Any) -> None:
        visible_text = raw.strip()
        if not visible_text:
            self._refresh_panels()
            app.invalidate()
            return
        try:
            command = shlex.split(visible_text)[0].lower() if visible_text.startswith("/") else ""
        except ValueError as exc:
            self._append_output(f"命令解析失败：{exc}")
            app.invalidate()
            return
        if command in {"/exit", "/quit", "/q"}:
            app.exit(result=0)
            return
        if self._answer_pending_approval(visible_text):
            app.invalidate()
            return
        text = self._expand_paste_references(visible_text)
        with self._tui_lock:
            if self._tui_busy:
                self._append_output("上一条输入仍在处理，请稍后。")
                app.invalidate()
                return
            self._tui_busy = True
        self._start_tui_turn_ticker(app)
        if text == visible_text:
            self._append_output(f"> {visible_text}")
        else:
            self._append_output(f"> {visible_text}\n[paste expanded for agent input]")

        def _worker() -> None:
            try:
                code = self._process_text(text)
                if code is not None:
                    app.exit(result=code)
            finally:
                with self._tui_lock:
                    self._tui_busy = False
                self._refresh_panels()
                app.invalidate()

        Thread(target=_worker, name="high-agent-tui-turn", daemon=True).start()

    def _start_tui_turn_ticker(self, app: Any) -> None:
        def _tick() -> None:
            while True:
                with self._tui_lock:
                    busy = self._tui_busy
                if not busy:
                    return
                agent = self._current_agent
                if agent is not None:
                    self._capture_agent_observability(agent)
                self._refresh_panels()
                try:
                    app.invalidate()
                except Exception:
                    pass
                time.sleep(0.5)

        Thread(target=_tick, name="high-agent-tui-timer", daemon=True).start()

    def _process_text(self, text: str) -> int | None:
        if text.startswith("/"):
            return self._handle_command(text)
        if text.startswith("!") and self.allow_yes:
            self._run_shell(text[1:].strip())
            return None
        self._run_user_turn(text)
        return None

    def _request_tui_approval(self, request: ApprovalRequest) -> ApprovalDecision:
        pending = PendingApproval(request)
        with self._approval_lock:
            if self._pending_approval is not None:
                return ApprovalDecision(False, "once", "another approval request is already pending")
            self._pending_approval = pending
        self._append_output(f"Approval required: {request.action} {request.resource}")
        if request.reason:
            self._append_output(f"Reason: {request.reason}")
        self._append_output("Allow? [y/N/s=允许本会话/a=永久允许]")
        pending.event.wait()
        return pending.decision or ApprovalDecision(False, "once", "approval cancelled")

    def _answer_pending_approval(self, text: str) -> bool:
        with self._approval_lock:
            pending = self._pending_approval
        if pending is None:
            return False

        answer = text.strip().lower()
        if answer in {"a", "always"}:
            decision = ApprovalDecision(True, "always", "user approved always")
            message = "审批已永久允许。"
        elif answer in {"s", "session"}:
            decision = ApprovalDecision(True, "session", "user approved session")
            message = "审批已在本会话允许。"
        elif answer in {"y", "yes"}:
            decision = ApprovalDecision(True, "once", "user approved once")
            message = "审批已允许一次。"
        elif answer in {"", "n", "no"}:
            decision = ApprovalDecision(False, "once", "user denied")
            message = "审批已拒绝。"
        else:
            self._append_output("请回复 y、s、a 或 n。")
            return True

        pending.decision = decision
        with self._approval_lock:
            if self._pending_approval is pending:
                self._pending_approval = None
        pending.event.set()
        self._append_output(message)
        return True

    def _cancel_pending_approval(self) -> None:
        with self._approval_lock:
            pending = self._pending_approval
            self._pending_approval = None
        if pending is not None and not pending.event.is_set():
            pending.decision = ApprovalDecision(False, "once", "TUI exited")
            pending.event.set()

    def _build_application(self) -> Any:
        if (
            Application is None
            or ConditionalContainer is None
            or Condition is None
            or Dimension is None
            or FormattedTextControl is None
            or HSplit is None
            or Layout is None
            or VSplit is None
            or CompletionsMenu is None
            or TextArea is None
            or Window is None
            or KeyBindings is None
            or get_app is None
        ):
            return None
        bindings = self._build_tui_key_bindings()
        if bindings is None:
            return None

        self._hint_area = Window(
            content=FormattedTextControl(self._tui_hint_fragments),
            height=Condition(lambda: self._tui_busy or self._pending_approval is not None),
            wrap_lines=False,
        )
        self._input_area = TextArea(
            multiline=True,
            height=Dimension(min=1, max=8, preferred=1),
            prompt="> ",
            completer=self._command_completer(),
            complete_while_typing=False,
            accept_handler=self._accept_tui_buffer,
            wrap_lines=False,
        )
        self._install_paste_collapse(self._input_area)
        self._status_area = Window(
            content=FormattedTextControl(lambda: [("", self._toolbar())]),
            height=1,
            wrap_lines=False,
        )
        self._task_panel_area = Window(
            content=FormattedTextControl(lambda: [("", self._render_task_panel())]),
            width=Dimension(weight=1),
            wrap_lines=False,
        )
        left = HSplit(
            [
                self._hint_area,
                Window(char="-", height=1),
                self._input_area,
                self._status_area,
                CompletionsMenu(max_height=12, scroll_offset=1),
            ],
            width=Dimension(weight=1),
        )
        panel_filter = Condition(self._task_panel_visible)
        root = VSplit(
            [
                left,
                ConditionalContainer(Window(char="|", width=1, dont_extend_width=True), filter=panel_filter),
                ConditionalContainer(self._task_panel_area, filter=panel_filter),
            ],
            width=Dimension(weight=1),
        )
        return Application(
            layout=Layout(root, focused_element=self._input_area),
            full_screen=False,
            key_bindings=bindings,
            min_redraw_interval=0.05,
            max_render_postpone_time=0.05,
            mouse_support=False,
        )

    def _run_user_turn(self, prompt: str) -> None:
        turn_start = time.monotonic()
        self._current_turn_started_at = turn_start
        self._refresh_panels()
        memory_digest = self.memory_store.render_digest(limit=5) if self.memory_store else ""
        effective_history = list(self.conversation_history)
        if memory_digest:
            effective_history.append({"role": "user", "content": memory_digest})
        agent = self.agent_factory(prompt)
        self._current_agent = agent
        self._refresh_task_panel_from_agent(agent)
        answer = ""
        error: Exception | None = None

        def _delivery_callback(batch: DeliveryBatch) -> None:
            self._on_delivery(batch)
            self._capture_agent_observability(agent)

        try:
            answer = agent.run(
                prompt,
                max_iterations=self.max_iterations,
                conversation_history=effective_history,
                on_delivery=_delivery_callback,
            )
        except Exception as exc:
            error = exc
        finally:
            agent.runtime.shutdown()
            self._capture_agent_observability(agent)
            if self._current_agent is agent:
                self._current_agent = None
            self._usage_snapshots.pop(id(agent), None)
            self._task_time_snapshots.pop(id(agent), None)
            duration = max(0.0, time.monotonic() - turn_start)
            self._record_turn_completion(duration)
            if self._current_turn_started_at == turn_start:
                self._current_turn_started_at = None
            self._refresh_panels()

        if error is not None:
            self.console.print(f"[red]运行失败：{error}[/red]")
            self._append_transcript({"role": "error", "content": str(error)})
            return

        self.conversation_history.append({"role": "user", "content": prompt})
        if answer:
            self.conversation_history.append({"role": "assistant", "content": answer})
            self.console.print(answer)
        self._append_transcript({"role": "user", "content": prompt})
        self._append_transcript({"role": "assistant", "content": answer})
        self.plugin_manager.run_hooks("after_turn", {"session_id": self.session_id, "prompt": prompt})

    def _on_delivery(self, batch: DeliveryBatch) -> None:
        self.last_digest = batch.digest
        for event in batch.events:
            duration = event.metadata.get("duration_seconds")
            suffix = f" ({format_duration_compact(float(duration))})" if duration is not None else ""
            self.console.print(f"[dim]✓ {event.summary}{suffix}[/dim]")
        self._append_transcript({"role": "runtime", "content": batch.digest})

    def _handle_command(self, text: str) -> int | None:
        parts = shlex.split(text)
        if not parts:
            return None
        command = parts[0].lower()
        args = parts[1:]
        entry = self.command_registry.get(command)
        if entry is None:
            self.console.print(f"未知命令：{command}。输入 /help 查看可用命令。")
            return None
        return self.command_registry.dispatch(command, args)

    def _register_builtin_commands(self) -> None:
        commands = self.command_registry
        commands.register("/exit", lambda args: 0, "退出", aliases=("/quit", "/q"))
        commands.register("/help", self._cmd_help, "显示命令")
        commands.register("/clear", self._cmd_clear, "清空当前会话上下文")
        commands.register("/history", self._cmd_history, "显示最近 n 条会话消息")
        commands.register("/sessions", self._cmd_sessions, "列出已保存会话")
        commands.register("/resume", self._cmd_resume, "恢复一个会话：/resume <session_id-prefix>")
        commands.register("/compress", self._cmd_compress, "压缩上下文", aliases=("/compact",))
        commands.register("/memory", self._cmd_memory, "管理长期记忆")
        commands.register("/plugins", self._cmd_plugins, "显示插件状态")
        commands.register("/model", self._cmd_model, "显示模型配置")
        commands.register("/usage", self._cmd_usage, "显示上下文估算和模型 token usage")
        commands.register("/tasks", self._cmd_tasks, "显示最近 runtime ledger snapshot")
        commands.register("/logs", self._cmd_logs, "显示 transcript、trace 和 tool result 路径")
        commands.register("/trace", self._cmd_trace, "启用或关闭 runtime trace：/trace on|off")
        commands.register("/panel", self._cmd_panel, "右侧 runtime 并行任务面板：/panel [on|off]")
        commands.register("/status", self._cmd_status, "显示 runtime digest、workspace、权限状态")
        commands.register("/tools", self._cmd_tools, "工具集：/tools [list|enable|disable] [name]")
        commands.register("/workspace", self._cmd_workspace, "查看或切换 workspace")
        commands.register("/yes", self._cmd_yes, "开启危险操作自动允许")
        commands.register("/no", self._cmd_no, "关闭危险操作自动允许")
        commands.register("/run", self._cmd_run, "以当前上下文运行一条 prompt")

    def _cmd_help(self, args: list[str]) -> None:
        for entry in self.command_registry.entries():
            aliases = f" ({', '.join(entry.aliases)})" if entry.aliases else ""
            self.console.print(f"{entry.name}{aliases:<18} {entry.help}")
        self.console.print("!<command>            /yes 开启后，直接在本地 shell 执行命令")
        return None

    def _cmd_clear(self, args: list[str]) -> None:
        self.conversation_history.clear()
        self.console.print("已清空当前会话上下文。")
        return None

    def _cmd_history(self, args: list[str]) -> None:
        self._print_history(limit=int(args[0]) if args and args[0].isdigit() else 10)
        return None

    def _cmd_sessions(self, args: list[str]) -> None:
        records = self.session_store.list(limit=int(args[0]) if args and args[0].isdigit() else 20)
        if not records:
            self.console.print("暂无已保存会话。")
            return None
        for record in records:
            self.console.print(f"{record.session_id}  {record.message_count:>3}  {record.title}")
        return None

    def _cmd_resume(self, args: list[str]) -> None:
        if not args:
            self.console.print("用法：/resume <session_id-prefix>")
            return None
        target = self._resolve_session_id(args[0])
        if target is None:
            self.console.print(f"未找到会话：{args[0]}")
            return None
        messages = self.session_store.resume(target)
        self.session_id = target
        self.conversation_history = [
            {"role": str(item.get("role")), "content": str(item.get("content") or "")}
            for item in messages
            if item.get("role") in {"user", "assistant"}
        ]
        self.console.print(f"已恢复 {target}，上下文消息 {len(self.conversation_history)} 条。")
        return None

    def _cmd_compress(self, args: list[str]) -> None:
        budget = int(args[0]) if args and args[0].isdigit() else 12_000
        result = self.compressor.maybe_compress(self.conversation_history, budget)
        self.conversation_history = [
            {"role": str(item.get("role")), "content": str(item.get("content") or "")}
            for item in result.messages
            if item.get("role") in {"user", "assistant", "system"}
        ]
        if result.compressed:
            self._append_transcript({"role": "system", "content": result.summary})
            self.console.print(f"已压缩：{result.original_count} -> {result.retained_count} 条消息。")
        else:
            self.console.print("当前上下文未超过压缩阈值。")
        return None

    def _cmd_memory(self, args: list[str]) -> None:
        if not args:
            digest = self.memory_store.render_digest(limit=12)
            self.console.print(digest or "暂无长期记忆。")
            return None
        action = args[0]
        if action == "add" and len(args) >= 3:
            fact_id = self.memory_store.write_fact(args[1], " ".join(args[2:]), source="chat")
            self.console.print(f"已保存记忆：{fact_id}")
            return None
        if action == "search":
            query = " ".join(args[1:])
            self.console.print(self.memory_store.render_digest(query, limit=12) or "没有匹配记忆。")
            return None
        self.console.print("用法：/memory | /memory add <key> <value> | /memory search <query>")
        return None

    def _cmd_plugins(self, args: list[str]) -> None:
        if args and args[0] == "reload":
            self.plugin_manager.load()
            self.plugin_manager.register_commands(self.command_registry, self.console)
        if not self.plugin_manager.manifests:
            self.console.print("未加载插件。")
            return None
        for manifest in self.plugin_manager.manifests:
            self.console.print(
                f"{manifest.name} enabled={manifest.enabled} tools={len(manifest.tools)} "
                f"commands={len(manifest.commands)} hooks={len(manifest.hooks)}"
            )
        return None

    def _cmd_model(self, args: list[str]) -> None:
        model = load_config().get("model", {})
        if not isinstance(model, dict) or not model:
            self.console.print("未配置模型。运行 high-agent setup。")
            return None
        self.console.print(
            f"provider={model.get('provider')} model={model.get('model')} "
            f"api_mode={model.get('api_mode')} base_url={model.get('base_url')}"
        )
        return None

    def _cmd_usage(self, args: list[str]) -> None:
        self.console.print(
            "usage: "
            f"context_estimate={self.session_usage['context_estimate']} "
            f"input={self.session_usage['input_tokens']} "
            f"output={self.session_usage['output_tokens']} "
            f"total={self.session_usage['total_tokens']} "
            f"model_calls={self.session_usage['model_calls']}"
        )
        return None

    def _cmd_tasks(self, args: list[str]) -> None:
        self.console.print(self.last_tasks or "tasks: no runtime has completed in this session yet")
        return None

    def _cmd_logs(self, args: list[str]) -> None:
        self.console.print(f"transcript: {self._transcript_path}")
        self.console.print(f"sessions_db: {self.session_store.path}")
        self.console.print(f"tool_results: {self.result_store.root}")
        self.console.print(f"trace: {self.last_trace_path or 'off'}")
        return None

    def _cmd_trace(self, args: list[str]) -> None:
        if not args:
            self.console.print(f"trace: {'on' if self.trace_enabled else 'off'}")
            return None
        value = args[0].lower()
        if value in {"on", "true", "yes", "1"}:
            self.trace_enabled = True
            self.console.print("runtime trace 已开启。")
        elif value in {"off", "false", "no", "0"}:
            self.trace_enabled = False
            self.console.print("runtime trace 已关闭。")
        else:
            self.console.print("用法：/trace on|off")
        self._refresh_panels()
        return None

    def _cmd_panel(self, args: list[str]) -> None:
        if args:
            value = args[0].lower()
            if value in {"on", "true", "yes", "1"}:
                self.task_panel_enabled = True
                self.console.print("runtime 并行任务面板已开启。")
            elif value in {"off", "false", "no", "0"}:
                self.task_panel_enabled = False
                self.console.print("runtime 并行任务面板已关闭。")
            else:
                self.console.print("用法：/panel [on|off]")
                return None
        state = "on" if self.task_panel_enabled else "off"
        self.console.print(f"panel={state} visible={self._task_panel_visible()} min_width={TASK_PANEL_MIN_COLUMNS}")
        self.console.print(self._render_task_panel())
        self._refresh_panels()
        return None

    def _cmd_status(self, args: list[str]) -> None:
        self._print_status()
        return None

    def _cmd_tools(self, args: list[str]) -> None:
        if not args or args[0] == "list":
            self.console.print(f"enabled toolsets: {', '.join(sorted(self.toolsets.enabled))}")
            self.console.print(f"tools: {', '.join(self.build_tool_registry().names())}")
            return None
        action = args[0]
        if len(args) < 2:
            self.console.print("用法：/tools list | /tools enable <toolset> | /tools disable <toolset>")
            return None
        name = args[1]
        try:
            if action == "enable":
                self.toolsets.enable(name)
                self.console.print(f"已启用 toolset：{name}")
            elif action == "disable":
                self.toolsets.disable(name)
                self.console.print(f"已禁用 toolset：{name}")
            else:
                self.console.print("用法：/tools list | /tools enable <toolset> | /tools disable <toolset>")
        except KeyError as exc:
            self.console.print(str(exc))
        return None

    def _cmd_workspace(self, args: list[str]) -> None:
        if args:
            path = Path(args[0]).expanduser()
            if not path.is_absolute():
                path = Path(self.workspace) / path
            self.workspace = os.path.abspath(path)
            self.console.print(f"workspace = {self.workspace}")
        else:
            self.console.print(f"workspace = {self.workspace}")
        return None

    def _cmd_yes(self, args: list[str]) -> None:
        self.allow_yes = True
        self.approval_manager.policy = "auto"
        self._sync_toolset_policy()
        self.console.print("已开启 --yes 行为：terminal、执行代码和 workspace 外路径允许执行。")
        return None

    def _cmd_no(self, args: list[str]) -> None:
        self.allow_yes = False
        self.approval_manager.policy = "ask"
        self._sync_toolset_policy()
        self.console.print("已关闭 --yes 行为。")
        return None

    def _cmd_run(self, args: list[str]) -> None:
        if args:
            self._run_user_turn(" ".join(args))
        else:
            self.console.print("用法：/run <prompt>")
        return None

    def _print_history(self, *, limit: int) -> None:
        recent = self.conversation_history[-limit:]
        if not recent:
            self.console.print("暂无会话历史。")
            return
        for item in recent:
            role = item.get("role", "?")
            content = item.get("content", "")
            self.console.print(f"[{role}] {content[:1000]}")

    def _print_status(self) -> None:
        self.console.print(f"session: {self.session_id}")
        self.console.print(f"workspace: {self.workspace}")
        self.console.print(f"yes: {self.allow_yes}")
        self.console.print(f"history messages: {len(self.conversation_history)}")
        self.console.print(f"toolsets: {', '.join(sorted(self.toolsets.enabled))}")
        self.console.print(f"plugins: {len(self.plugin_manager.manifests)}")
        self.console.print(f"usage: {self.session_usage}")
        self.console.print(
            "turn_time: "
            f"last={format_duration_compact(self.turn_usage['last_turn_seconds'])} "
            f"total={format_duration_compact(self.turn_usage['total_turn_seconds'])} "
            f"count={int(self.turn_usage['turn_count'])}"
        )
        self.console.print(
            "task_time: "
            f"task={format_duration_compact(self.task_usage['task_seconds'])} "
            f"wall={format_duration_compact(self.task_usage['wall_seconds'])} "
            f"completed={int(self.task_usage['completed_tasks'])} "
            f"running={int(self.task_usage['running_tasks'])}"
        )
        self.console.print(f"trace: {self.last_trace_path or 'off'}")
        self.console.print(f"last digest: {self.last_digest or 'none'}")
        self.console.print(f"transcript: {self._transcript_path}")

    def _run_shell(self, command: str) -> None:
        if not command:
            return
        proc = subprocess.run(command, cwd=self.workspace, shell=True, text=True, capture_output=True, check=False)
        if proc.stdout:
            self.console.print(proc.stdout.rstrip())
        if proc.stderr:
            self.console.print(f"[red]{proc.stderr.rstrip()}[/red]")
        self.console.print(f"[dim]exit_code={proc.returncode}[/dim]")

    def _append_transcript(self, item: dict[str, str]) -> None:
        payload = {"ts": time.time(), **item}
        with self._transcript_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.session_store.append(self.session_id, payload)

    def _append_output(self, text: str) -> None:
        text = _panel_safe_text(text)
        if self._application is None:
            self.console.print(text)
            return
        self.console.print(text)
        self._refresh_panels()

    def _refresh_panels(self) -> None:
        if self._application is not None:
            try:
                self._application.invalidate()
            except Exception:
                pass

    def _render_task_panel(self) -> str:
        return self._task_panel_text or (
            "parallel graph\n"
            "idle\n"
            "真实 AgentTask 状态会在任务运行时显示。"
        )

    def _refresh_task_panel_from_agent(self, agent: MainAgent) -> None:
        self._task_panel_text = render_task_panel_from_agent(agent)

    def _task_panel_visible(self) -> bool:
        return _task_panel_visible_check(self.task_panel_enabled)

    def _terminal_columns(self) -> int:
        return _terminal_columns_fn()

    def _tui_hint_fragments(self) -> list[tuple[str, str]]:
        if self._pending_approval is not None:
            return [("", "审批等待中：输入 y / s / a / n 后回车")]
        if self._tui_busy:
            return [("", "agent 正在执行；当前输入区仍可编辑，回车会提示稍后再试")]
        return []

    def _install_paste_collapse(self, input_area: Any) -> None:
        def _on_text_changed(buffer: Any) -> None:
            if self._suppress_paste_collapse:
                return
            text = str(buffer.text or "")
            if not text or text.startswith("/") or text.startswith("[Pasted text #"):
                return
            if len(text) < PASTE_COLLAPSE_CHARS and text.count("\n") + 1 < PASTE_COLLAPSE_LINES:
                return
            self._paste_counter += 1
            paste_dir = get_config_paths().home / "pastes"
            paste_dir.mkdir(parents=True, exist_ok=True)
            paste_file = paste_dir / f"paste_{self._paste_counter}_{int(time.time() * 1000)}.txt"
            paste_file.write_text(text, encoding="utf-8")
            marker = f"[Pasted text #{self._paste_counter}: {text.count(chr(10)) + 1} lines -> {paste_file}]"
            self._suppress_paste_collapse = True
            try:
                buffer.text = marker
                buffer.cursor_position = len(marker)
            finally:
                self._suppress_paste_collapse = False

        input_area.buffer.on_text_changed += _on_text_changed

    def _expand_paste_references(self, text: str) -> str:
        def _replace(match: re.Match[str]) -> str:
            path = Path(match.group(1)).expanduser()
            try:
                return path.read_text(encoding="utf-8")
            except OSError:
                return match.group(0)

        return PASTE_REF_RE.sub(_replace, text)

    def _resolve_session_id(self, prefix: str) -> str | None:
        for record in self.session_store.list(limit=200):
            if record.session_id == prefix or record.session_id.startswith(prefix):
                return record.session_id
        return None

    def _sync_toolset_policy(self) -> None:
        self.toolsets.allow_terminal = self.allow_yes
        self.toolsets.allow_outside_workspace = self.allow_yes
        self.toolsets.approval_manager = self.approval_manager
        self.toolsets.result_store = self.result_store

    def next_trace_path(self) -> Path:
        root = get_config_paths().home / "traces" / self.session_id
        root.mkdir(parents=True, exist_ok=True)
        self.last_trace_path = root / f"run-{int(time.time() * 1000)}.jsonl"
        return self.last_trace_path

    def _capture_agent_observability(self, agent: MainAgent) -> None:
        with self._observability_lock:
            self._capture_agent_observability_locked(agent)

    def _capture_agent_observability_locked(self, agent: MainAgent) -> None:
        usage = getattr(agent, "last_usage", None)
        if usage is not None:
            data = usage.as_dict()
            agent_key = id(agent)
            previous = self._usage_snapshots.get(
                agent_key,
                {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "model_calls": 0},
            )
            self.session_usage["context_estimate"] = data["context_estimate"]
            for key in ("input_tokens", "output_tokens", "total_tokens", "model_calls"):
                self.session_usage[key] += max(0, int(data[key]) - int(previous.get(key, 0)))
            self._usage_snapshots[agent_key] = {
                "input_tokens": int(data["input_tokens"]),
                "output_tokens": int(data["output_tokens"]),
                "total_tokens": int(data["total_tokens"]),
                "model_calls": int(data["model_calls"]),
            }
        try:
            counts = agent.runtime.ledger.counts()
            digest = agent.runtime.status_digest().text
            rows = [f"{key}={value}" for key, value in sorted(counts.items())]
            timing_summary = ""
            timing_func = getattr(agent.runtime.ledger, "timing", None)
            if callable(timing_func):
                timing = timing_func()
                timing_data = timing.as_dict()
                task_key = id(agent)
                previous_timing = self._task_time_snapshots.get(
                    task_key,
                    {"task_seconds": 0.0, "wall_seconds": 0.0, "completed_tasks": 0.0},
                )
                for key in ("task_seconds", "wall_seconds", "completed_tasks"):
                    self.task_usage[key] += max(0.0, float(timing_data[key]) - float(previous_timing.get(key, 0.0)))
                self.task_usage["running_tasks"] = float(timing_data["running_tasks"])
                self.task_usage["last_task_seconds"] = float(timing_data["task_seconds"])
                self.task_usage["last_wall_seconds"] = float(timing_data["wall_seconds"])
                self._task_time_snapshots[task_key] = {
                    "task_seconds": float(timing_data["task_seconds"]),
                    "wall_seconds": float(timing_data["wall_seconds"]),
                    "completed_tasks": float(timing_data["completed_tasks"]),
                }
                timing_summary = f"\n{timing.summary()}"
            self.last_tasks = f"tasks: {', '.join(rows) or 'idle'}{timing_summary}\n{digest}"
            self._refresh_task_panel_from_agent(agent)
        except Exception:
            self.last_tasks = "tasks: unavailable"
            self._task_panel_text = "parallel graph\ntasks unavailable"
        trace_path = getattr(getattr(agent.runtime, "trace", None), "path", None)
        if trace_path:
            self.last_trace_path = trace_path
        self._refresh_panels()

    def _record_turn_completion(self, duration: float) -> None:
        with self._turn_usage_lock:
            self.turn_usage["last_turn_seconds"] = duration
            self.turn_usage["total_turn_seconds"] += duration
            self.turn_usage["turn_count"] += 1

    def _display_turn_seconds(self) -> float:
        if self._current_turn_started_at is not None:
            return max(0.0, time.monotonic() - self._current_turn_started_at)
        return float(self.turn_usage["last_turn_seconds"])

    def _model_label(self) -> str:
        model = load_config().get("model", {})
        if isinstance(model, dict):
            provider = model.get("provider") or "?"
            name = model.get("model") or "?"
            return f"{provider}/{name}"
        return "unconfigured"


class InteractiveApp(InteractiveSession):
    """Prompt-toolkit based app facade used by the public v0.4 interface."""


class SlashCommandCompleter(Completer):
    """Complete slash commands using CommandRegistry, including aliases."""

    def __init__(self, registry: CommandRegistry) -> None:
        self.registry = registry

    def get_completions(self, document: Any, complete_event: Any) -> Any:
        if Completion is None:
            return
        token = _slash_command_token(document.current_line_before_cursor)
        if token is None:
            return
        for name, help_text in self._commands():
            if name.lower().startswith(token.lower()):
                yield Completion(name, start_position=-len(token), display=name, display_meta=help_text)

    def _commands(self) -> list[tuple[str, str]]:
        rows: list[tuple[str, str]] = []
        for entry in self.registry.entries():
            rows.append((entry.name, entry.help))
            for alias in entry.aliases:
                rows.append((alias, f"alias for {entry.name}"))
        return sorted(rows)


def _slash_command_token(before_cursor: str) -> str | None:
    if len(before_cursor) > COMMAND_COMPLETION_MAX_CHARS:
        return None
    stripped = before_cursor.lstrip()
    if not stripped.startswith("/") or " " in stripped:
        return None
    return stripped
