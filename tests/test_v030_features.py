from __future__ import annotations

import http.server
import json
import os
import socketserver
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from high_agent.agent import MainAgent
from high_agent.approval import ApprovalDecision, ApprovalManager
from high_agent.cli.commands import CommandRegistry
from high_agent.cli.session import InteractiveSession
from high_agent.llm.types import NormalizedResponse, ToolCall
from high_agent.memory import ContextCompressor, MemoryStore, SessionStore
from high_agent.plugins import PluginManager
from high_agent.runtime import CausalRuntime, TaskResult
from high_agent.runtime.resource_access import ResourceAccess
from high_agent.tools import ToolRegistry, ToolResultStore, ToolsetRegistry, create_core_registry


class ApprovalAndToolTests(unittest.TestCase):
    def test_approval_policies_and_session_cache(self) -> None:
        auto = ApprovalManager(policy="auto", interactive=False)
        self.assertTrue(auto.request(_approval_request()).approved)

        deny = ApprovalManager(policy="deny", interactive=False)
        self.assertFalse(deny.request(_approval_request()).approved)

        calls: list[str] = []
        cached = ApprovalManager(
            policy="ask",
            interactive=False,
            callback=lambda request: (calls.append(request.key), ApprovalDecision(True, "session"))[1],
        )
        self.assertTrue(cached.request(_approval_request()).approved)
        self.assertTrue(cached.request(_approval_request()).approved)
        self.assertEqual(calls, ["terminal:echo hi"])

    def test_tool_result_store_keeps_large_output_out_of_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ToolResultStore(Path(tmp) / "tool-results")
            registry = ToolRegistry(result_store=store)
            registry.register(
                name="big",
                schema={"description": "big output", "parameters": {"type": "object", "properties": {}}},
                handler=lambda args: "x" * 200,
                resource_access=lambda args, root: ResourceAccess.empty(),
                max_result_size_chars=20,
            )
            runtime = CausalRuntime(workspace_root=tmp, delivery_debounce=0.0)
            self.addCleanup(runtime.shutdown)
            runtime.submit([registry.task_from_call("big", {}, workspace_root=tmp, task_id="big")])
            self.assertTrue(runtime.wait_all(timeout=1))
            result = runtime.collect({"big"})["big"]
            payload = json.loads(result.summary)
            self.assertTrue(payload["truncated"])
            self.assertEqual(store.get(payload["result_id"]), "x" * 200)

    def test_tool_registry_rejects_duplicate_name_without_override(self) -> None:
        # previously a second register() with the same name silently
        # replaced the first entry. Two plugins / extensions accidentally
        # sharing a name (or a typo collision against a core tool like
        # write_file) would swap the binding with no warning, and the wrong
        # handler ran with the wrong resource_access declared.
        registry = ToolRegistry()
        registry.register(
            name="dup",
            schema={"description": "v1", "parameters": {"type": "object", "properties": {}}},
            handler=lambda args: "v1",
            resource_access=lambda args, root: ResourceAccess.empty(),
        )
        with self.assertRaises(ValueError) as ctx:
            registry.register(
                name="dup",
                schema={"description": "v2", "parameters": {"type": "object", "properties": {}}},
                handler=lambda args: "v2",
                resource_access=lambda args, root: ResourceAccess.empty(),
            )
        self.assertIn("already registered", str(ctx.exception))
        # The original binding survives.
        self.assertEqual(registry.get("dup").handler({}), "v1")

    def test_tool_registry_allows_override_when_explicit(self) -> None:
        # Hot-reload / test fixtures still need to swap; override=True opts in.
        registry = ToolRegistry()
        registry.register(
            name="swap",
            schema={"description": "v1", "parameters": {"type": "object", "properties": {}}},
            handler=lambda args: "v1",
            resource_access=lambda args, root: ResourceAccess.empty(),
        )
        registry.register(
            name="swap",
            schema={"description": "v2", "parameters": {"type": "object", "properties": {}}},
            handler=lambda args: "v2",
            resource_access=lambda args, root: ResourceAccess.empty(),
            override=True,
        )
        self.assertEqual(registry.get("swap").handler({}), "v2")

    def test_toolsets_enable_disable_and_project_tools(self) -> None:
        toolsets = ToolsetRegistry()
        self.assertIn("terminal", toolsets.tool_names())
        toolsets.disable("terminal")
        self.assertNotIn("terminal", toolsets.registry().names())
        toolsets.enable("mcp")
        self.assertIn("mcp_call", toolsets.registry().names())

    def test_patch_todo_python_http_and_mcp_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            approval = ApprovalManager(policy="auto", interactive=False)
            registry = create_core_registry(allow_terminal=True, approval_manager=approval)
            runtime = CausalRuntime(workspace_root=tmp, delivery_debounce=0.0)
            self.addCleanup(runtime.shutdown)

            (Path(tmp) / "a.txt").write_text("hello old\n", encoding="utf-8")
            tasks = [
                registry.task_from_call("patch_file", {"path": "a.txt", "replacements": [{"old": "old", "new": "new"}]}, workspace_root=tmp, task_id="patch"),
                registry.task_from_call("todo_write", {"items": [{"task": "build", "status": "doing"}]}, workspace_root=tmp, task_id="todo-w"),
                registry.task_from_call("todo_read", {}, workspace_root=tmp, task_id="todo-r"),
                registry.task_from_call("run_python", {"code": "print('py-ok')"}, workspace_root=tmp, task_id="py"),
                registry.task_from_call("mcp_call", {"server": "fake", "tool": "ping"}, workspace_root=tmp, task_id="mcp"),
            ]
            runtime.submit(tasks)
            self.assertTrue(runtime.wait_all(timeout=2))
            results = runtime.collect({task.task_id for task in tasks})
            self.assertIn("new", (Path(tmp) / "a.txt").read_text(encoding="utf-8"))
            self.assertIn("py-ok", results["py"].summary)
            self.assertEqual(results["mcp"].status, "completed")

            blocked = create_core_registry()
            task = blocked.task_from_call("run_python", {"code": "print('no')"}, workspace_root=tmp, task_id="blocked-py")
            runtime.submit([task])
            self.assertTrue(runtime.wait_all(timeout=1))
            self.assertEqual(runtime.collect({"blocked-py"})["blocked-py"].status, "blocked")

    def test_http_fetch_uses_browser_lite_tool(self) -> None:
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"hello browser-lite")

            def log_message(self, format: str, *args: object) -> None:
                return

        with socketserver.TCPServer(("127.0.0.1", 0), Handler) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    registry = create_core_registry()
                    runtime = CausalRuntime(workspace_root=tmp, delivery_debounce=0.0)
                    self.addCleanup(runtime.shutdown)
                    url = f"http://127.0.0.1:{server.server_address[1]}/"
                    runtime.submit([registry.task_from_call("http_fetch", {"url": url}, workspace_root=tmp, task_id="fetch")])
                    self.assertTrue(runtime.wait_all(timeout=2))
                    self.assertIn("browser-lite", runtime.collect({"fetch"})["fetch"].summary)
            finally:
                server.shutdown()


class MemoryPluginCommandTests(unittest.TestCase):
    def test_session_store_compressor_and_memory_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session_store = SessionStore(Path(tmp) / "sessions.db")
            session_id = session_store.create("demo")
            session_store.append(session_id, {"role": "user", "content": "hello"})
            session_store.append(session_id, {"role": "assistant", "content": "world"})
            self.assertEqual(len(session_store.resume(session_id)), 2)
            self.assertEqual(session_store.list()[0].message_count, 2)

            compressor = ContextCompressor(keep_recent=2, max_item_chars=40)
            messages = [{"role": "user", "content": f"message {i} " + "x" * 50} for i in range(6)]
            compressed = compressor.maybe_compress(messages, budget=80)
            self.assertTrue(compressed.compressed)
            self.assertLess(len(compressed.messages), len(messages))
            self.assertIn("Context compressed", compressed.summary)

            memory = MemoryStore(Path(tmp) / "memory.db")
            memory.write_fact("project", "uses high-agent", source="test")
            self.assertIn("uses high-agent", memory.render_digest("project"))

    def test_plugin_manager_registers_enabled_manifest_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "plugins"
            enabled = root / "demo"
            disabled = root / "off"
            enabled.mkdir(parents=True)
            disabled.mkdir(parents=True)
            (enabled / "plugin.yaml").write_text(
                """
name: demo
enabled: true
tools:
  - name: demo_tool
    description: Demo tool
    resource_access: none
    response:
      ok: true
commands:
  - name: /demo
    help: Demo command
    response: demo command ran
hooks:
  - name: after_turn
""".strip(),
                encoding="utf-8",
            )
            (disabled / "plugin.yaml").write_text("name: off\nenabled: false\n", encoding="utf-8")

            manager = PluginManager(root)
            loaded = manager.load()
            self.assertEqual([manifest.name for manifest in loaded], ["demo"])

            registry = ToolRegistry()
            tool_names = manager.register_tools(registry)
            self.assertEqual(tool_names, ["demo_tool"])
            self.assertIn("demo_tool", registry.names())

            console = _CollectConsole()
            commands = CommandRegistry()
            command_names = manager.register_commands(commands, console)
            self.assertEqual(command_names, ["/demo"])
            self.assertIsNone(commands.dispatch("/demo", []))
            self.assertIn("demo command ran", console.lines[-1])
            self.assertEqual(manager.run_hooks("after_turn")[0]["plugin"], "demo")

    def test_command_registry_and_interactive_session_commands(self) -> None:
        registry = CommandRegistry()
        called: list[list[str]] = []
        registry.register("/x", lambda args: (called.append(args), None)[1], "x", aliases=("/xx",))
        self.assertIsNone(registry.dispatch("/xx", ["1"]))
        self.assertEqual(called, [["1"]])

        with tempfile.TemporaryDirectory() as tmp:
            old_home = os.environ.get("HIGH_AGENT_HOME")
            os.environ["HIGH_AGENT_HOME"] = str(Path(tmp) / "home")
            try:
                console = _CollectConsole()
                session = InteractiveSession(
                    agent_factory=lambda prompt: _FakeAgent(),
                    workspace=tmp,
                    allow_yes=False,
                    console=console,
                )
                self.assertIsNone(session._handle_command("/memory add project high-agent"))
                self.assertIsNone(session._handle_command("/memory search project"))
                self.assertIsNone(session._handle_command("/tools disable terminal"))
                self.assertNotIn("terminal", session.build_tool_registry().names())
                self.assertIsNone(session._handle_command("/compress 1"))
                self.assertIsNone(session._handle_command("/sessions"))
                self.assertIsNone(session._handle_command("/plugins"))
            finally:
                if old_home is None:
                    os.environ.pop("HIGH_AGENT_HOME", None)
                else:
                    os.environ["HIGH_AGENT_HOME"] = old_home


class ProjectBuildEndToEndTests(unittest.TestCase):
    def test_fake_model_builds_tests_fixes_and_finishes_project(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = CausalRuntime(workspace_root=tmp, max_workers=4, delivery_debounce=0.0)
            self.addCleanup(runtime.shutdown)
            fake = _FakeModel(
                [
                    NormalizedResponse(
                        None,
                        [
                            ToolCall("mkdir", "mkdir", json.dumps({"path": "app"})),
                            ToolCall("code", "write_file", json.dumps({"path": "app/math_utils.py", "content": "def add(a, b):\n    return a - b\n"})),
                            ToolCall("test", "write_file", json.dumps({"path": "test_math.py", "content": "from app.math_utils import add\nassert add(2, 3) == 5\n"})),
                        ],
                        "tool_calls",
                    ),
                    NormalizedResponse(
                        None,
                        [ToolCall("run-1", "run_python", json.dumps({"code": "import sys; sys.path.insert(0, '.'); import test_math"}))],
                        "tool_calls",
                    ),
                    NormalizedResponse(
                        None,
                        [ToolCall("fix", "patch_file", json.dumps({"path": "app/math_utils.py", "replacements": [{"old": "return a - b", "new": "return a + b"}]}))],
                        "tool_calls",
                    ),
                    NormalizedResponse(
                        None,
                        [ToolCall("run-2", "run_python", json.dumps({"code": "import sys; sys.path.insert(0, '.'); import test_math; print('tests ok')"}))],
                        "tool_calls",
                    ),
                    NormalizedResponse("项目已创建并通过测试。", None, "stop"),
                    NormalizedResponse("项目已创建并通过测试。", None, "stop"),
                ]
            )
            registry = create_core_registry(allow_terminal=True, approval_manager=ApprovalManager(policy="auto", interactive=False))
            agent = MainAgent("build project", runtime, model_client=fake, tools=registry, max_planner_requests=1)
            answer = agent.run("构建一个带测试的小项目", max_iterations=10)
            self.assertIn("通过测试", answer)
            self.assertEqual((Path(tmp) / "app" / "math_utils.py").read_text(encoding="utf-8"), "def add(a, b):\n    return a + b\n")
            self.assertGreaterEqual(len(fake.calls), 5)


def _approval_request():
    from high_agent.approval import ApprovalRequest

    return ApprovalRequest("terminal", "echo hi")


class _FakeModel:
    def __init__(self, responses: list[NormalizedResponse]) -> None:
        self.responses = responses
        self.calls: list[list[dict]] = []

    def complete(self, messages: list[dict], tools: list[dict] | None = None, **params) -> NormalizedResponse:
        self.calls.append([dict(message) for message in messages])
        if not self.responses:
            return NormalizedResponse("", None, "stop")
        return self.responses.pop(0)


class _FakeRuntime:
    def shutdown(self) -> None:
        pass


class _FakeAgent:
    runtime = _FakeRuntime()

    def run(self, prompt: str, **kwargs) -> str:
        callback = kwargs.get("on_delivery")
        if callback:
            from high_agent.runtime.types import DeliveryBatch, DeliveryEvent

            event = DeliveryEvent(
                seq=1,
                task_id="task-1",
                kind="tool",
                summary="tool done",
                result=TaskResult.completed("tool done"),
            )
            callback(DeliveryBatch(events=[event], digest="fake delivery", batch_seq=1))
        return "session answer"


class _CollectConsole:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def print(self, *values: object, **kwargs: object) -> None:
        self.lines.append(" ".join(str(value) for value in values))


if __name__ == "__main__":
    unittest.main()
