from __future__ import annotations

import contextlib
import io
import json
import os
import stat
import sys
import tempfile
import threading
import unittest
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from high_agent.agent import MainAgent
from high_agent.cli.session import InteractiveSession
from high_agent.cli.setup import run_setup
from high_agent.config import ConfigPaths, load_config, load_secrets, save_config, save_secrets
from high_agent.cli.main import _build_agent
from high_agent.llm.client import ModelClient, ModelClientError
from high_agent.llm.providers import ModelSettings, resolve_model_config
from high_agent.llm.transport import AnthropicMessagesTransport, OpenAIChatCompletionsTransport, OpenAIResponsesTransport
from high_agent.llm.types import NormalizedResponse, ToolCall
from high_agent.runtime import CausalRuntime, DeliveryBatch, TaskResult
from high_agent.runtime.types import DeliveryEvent
from high_agent.tools import create_core_registry


class ConfigAndSetupTests(unittest.TestCase):
    def test_config_and_secrets_roundtrip_with_secret_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = ConfigPaths(Path(tmp), Path(tmp) / "config.yaml", Path(tmp) / "secrets.yaml")
            save_config({"model": {"provider": "openai"}}, paths)
            save_secrets({"providers": {"openai": {"api_key": "sk-test"}}}, paths)
            self.assertEqual(load_config(paths)["model"]["provider"], "openai")
            self.assertEqual(load_secrets(paths)["providers"]["openai"]["api_key"], "sk-test")
            self.assertEqual(stat.S_IMODE(os.stat(paths.secrets_path).st_mode), 0o600)

    def test_model_resolution_precedence_uses_cli_then_env_then_config(self) -> None:
        config = {
            "model": {
                "provider": "openai",
                "model": "config-model",
                "base_url": "https://api.openai.com/v1",
            }
        }
        secrets = {"providers": {"openai": {"api_key": "secret-key"}}}
        resolved = resolve_model_config(
            config,
            secrets,
            cli_overrides={"model": "cli-model"},
            env={"HIGH_AGENT_MODEL": "env-model", "OPENAI_API_KEY": "env-key"},
        )
        self.assertEqual(resolved.settings.model, "cli-model")
        self.assertEqual(resolved.settings.api_key, "env-key")
        self.assertEqual(resolved.settings.api_mode, "codex_responses")

    def test_setup_noninteractive_prints_guidance(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = run_setup(non_interactive=True)
        self.assertEqual(code, 0)
        self.assertIn("HIGH_AGENT_PROVIDER", out.getvalue())

    def test_setup_interactive_saves_model_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = ConfigPaths(Path(tmp), Path(tmp) / "config.yaml", Path(tmp) / "secrets.yaml")
            answers = iter(["custom", "http://127.0.0.1:9999/v1", "chat_completions", "test-model"])
            with contextlib.redirect_stdout(io.StringIO()):
                code = run_setup(
                    paths=paths,
                    input_func=lambda prompt: next(answers),
                    getpass_func=lambda prompt: "sk-test",
                )
            self.assertEqual(code, 0)
            self.assertEqual(load_config(paths)["model"]["model"], "test-model")
            self.assertEqual(load_secrets(paths)["providers"]["custom"]["api_key"], "sk-test")


class TransportTests(unittest.TestCase):
    def test_chat_completions_transport_normalizes_tool_call(self) -> None:
        settings = ModelSettings("custom", "m", "https://example.test/v1", "chat_completions", "key")
        transport = OpenAIChatCompletionsTransport(settings)
        request = transport.build_http_request(model="m", messages=[{"role": "user", "content": "hi"}], tools=[])
        self.assertEqual(request.url, "https://example.test/v1/chat/completions")
        response = transport.normalize_response(
            {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {"id": "call-1", "function": {"name": "noop", "arguments": "{}"}}
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        )
        self.assertEqual(response.tool_calls[0].name, "noop")

    def test_anthropic_transport_normalizes_tool_use(self) -> None:
        settings = ModelSettings("anthropic", "m", "https://api.anthropic.com", "anthropic_messages", "key")
        transport = AnthropicMessagesTransport(settings)
        request = transport.build_http_request(model="m", messages=[{"role": "user", "content": "hi"}], tools=[])
        self.assertEqual(request.url, "https://api.anthropic.com/v1/messages")
        response = transport.normalize_response(
            {"content": [{"type": "tool_use", "id": "u1", "name": "noop", "input": {"x": 1}}], "usage": {}}
        )
        self.assertEqual(response.tool_calls[0].arguments, '{"x": 1}')

    def test_responses_transport_normalizes_function_call(self) -> None:
        settings = ModelSettings("openai", "m", "https://api.openai.com/v1", "codex_responses", "key")
        transport = OpenAIResponsesTransport(settings)
        request = transport.build_http_request(model="m", messages=[{"role": "user", "content": "hi"}], tools=[])
        self.assertEqual(request.url, "https://api.openai.com/v1/responses")
        response = transport.normalize_response(
            {"output": [{"type": "function_call", "call_id": "c1", "name": "noop", "arguments": "{}"}]}
        )
        self.assertEqual(response.tool_calls[0].id, "c1")

    def test_model_client_accepts_injected_http_client(self) -> None:
        settings = ModelSettings("custom", "m", "https://example.test/v1", "chat_completions", "key")
        client = ModelClient(settings, http_client=_FakeHttpClient({"choices": [{"message": {"content": "ok"}}]}))
        self.assertEqual(client.complete([{"role": "user", "content": "hi"}]).content, "ok")

    def test_model_client_uses_streaming_chat_completions(self) -> None:
        settings = ModelSettings("custom", "m", "https://example.test/v1", "chat_completions", "key")
        http = _FakeStreamHttpClient(
            [
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call-1","function":{"name":"noop","arguments":"{\\"x\\""}}]}}]}\n',
                "\n",
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":":1}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":7,"completion_tokens":2,"total_tokens":9}}\n',
                "\n",
                "data: [DONE]\n",
                "\n",
            ]
        )
        response = ModelClient(settings, http_client=http).complete([{"role": "user", "content": "hi"}])
        self.assertTrue(http.stream_requests)
        self.assertTrue(http.stream_requests[0]["json"]["stream"])
        self.assertEqual(response.tool_calls[0].id, "call-1")
        self.assertEqual(response.tool_calls[0].name, "noop")
        self.assertEqual(response.tool_calls[0].arguments, '{"x":1}')
        self.assertEqual(response.usage.total_tokens, 9)

    def test_anthropic_transport_normalizes_streaming_text_and_tool_use(self) -> None:
        settings = ModelSettings("anthropic", "m", "https://api.anthropic.com", "anthropic_messages", "key")
        response = AnthropicMessagesTransport(settings).normalize_stream_events(
            [
                {"event": "message_start", "data": {"type": "message_start", "message": {"usage": {"input_tokens": 5}}}},
                {"event": "content_block_start", "data": {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}},
                {"event": "content_block_delta", "data": {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}}},
                {"event": "content_block_start", "data": {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "u1", "name": "noop", "input": {}}}},
                {"event": "content_block_delta", "data": {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{\"x\":1}"}}},
                {"event": "message_delta", "data": {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 3}}},
            ]
        )
        self.assertEqual(response.content, "hi")
        self.assertEqual(response.tool_calls[0].id, "u1")
        self.assertEqual(response.tool_calls[0].arguments, '{"x":1}')
        self.assertEqual(response.usage.total_tokens, 8)

    def test_responses_transport_normalizes_streaming_text_and_function_call(self) -> None:
        settings = ModelSettings("openai", "m", "https://api.openai.com/v1", "codex_responses", "key")
        response = OpenAIResponsesTransport(settings).normalize_stream_events(
            [
                {"event": "response.output_text.delta", "data": {"type": "response.output_text.delta", "delta": "hi"}},
                {"event": "response.output_item.added", "data": {"type": "response.output_item.added", "output_index": 1, "item": {"type": "function_call", "id": "fc1", "call_id": "c1", "name": "noop", "arguments": ""}}},
                {"event": "response.function_call_arguments.delta", "data": {"type": "response.function_call_arguments.delta", "output_index": 1, "delta": "{\"x\""}},
                {"event": "response.function_call_arguments.delta", "data": {"type": "response.function_call_arguments.delta", "output_index": 1, "delta": ":1}"}},
                {"event": "response.completed", "data": {"type": "response.completed", "response": {"status": "completed", "usage": {"input_tokens": 4, "output_tokens": 2, "total_tokens": 6}}}},
            ]
        )
        self.assertEqual(response.content, "hi")
        self.assertEqual(response.tool_calls[0].id, "c1")
        self.assertEqual(response.tool_calls[0].arguments, '{"x":1}')
        self.assertEqual(response.usage.total_tokens, 6)

    def test_model_client_wraps_read_timeout_with_configuration_hint(self) -> None:
        import httpx

        settings = ModelSettings("custom", "m", "https://example.test/v1", "chat_completions", "key")
        client = ModelClient(settings, http_client=_TimeoutHttpClient(httpx.ReadTimeout("read operation timed out")), timeout=12)
        with self.assertRaises(ModelClientError) as caught:
            client.complete([{"role": "user", "content": "hi"}])
        self.assertIn("timed out after 12s", str(caught.exception))
        self.assertIn("HIGH_AGENT_MODEL_TIMEOUT_SECONDS", str(caught.exception))

    def test_build_agent_applies_model_timeout_from_config_env_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old_home = os.environ.get("HIGH_AGENT_HOME")
            old_timeout = os.environ.get("HIGH_AGENT_MODEL_TIMEOUT_SECONDS")
            os.environ["HIGH_AGENT_HOME"] = str(Path(tmp) / "home")
            try:
                paths = ConfigPaths(Path(os.environ["HIGH_AGENT_HOME"]), Path(os.environ["HIGH_AGENT_HOME"]) / "config.yaml", Path(os.environ["HIGH_AGENT_HOME"]) / "secrets.yaml")
                save_config(
                    {
                        "model": {
                            "provider": "custom",
                            "model": "m",
                            "base_url": "https://example.test/v1",
                            "api_mode": "chat_completions",
                            "timeout_seconds": 321,
                        }
                    },
                    paths,
                )
                save_secrets({"providers": {"custom": {"api_key": "key"}}}, paths)
                args = Namespace(
                    provider=None,
                    model=None,
                    base_url=None,
                    api_mode=None,
                    api_key=None,
                    max_workers=None,
                    workspace=tmp,
                    yes=False,
                    trace=False,
                    trace_path=None,
                    model_timeout=None,
                    delivery_debounce=None,
                )
                agent = _build_agent(args, "inspect")
                self.addCleanup(agent.runtime.shutdown)
                self.assertEqual(agent.model_client.timeout, 321)

                os.environ["HIGH_AGENT_MODEL_TIMEOUT_SECONDS"] = "432"
                agent_env = _build_agent(args, "inspect")
                self.addCleanup(agent_env.runtime.shutdown)
                self.assertEqual(agent_env.model_client.timeout, 432)

                args.model_timeout = 543
                agent_cli = _build_agent(args, "inspect")
                self.addCleanup(agent_cli.runtime.shutdown)
                self.assertEqual(agent_cli.model_client.timeout, 543)
            finally:
                if old_home is None:
                    os.environ.pop("HIGH_AGENT_HOME", None)
                else:
                    os.environ["HIGH_AGENT_HOME"] = old_home
                if old_timeout is None:
                    os.environ.pop("HIGH_AGENT_MODEL_TIMEOUT_SECONDS", None)
                else:
                    os.environ["HIGH_AGENT_MODEL_TIMEOUT_SECONDS"] = old_timeout

    def test_build_agent_applies_delivery_debounce_from_config_env_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old_home = os.environ.get("HIGH_AGENT_HOME")
            old_debounce = os.environ.get("HIGH_AGENT_DELIVERY_DEBOUNCE_SECONDS")
            os.environ["HIGH_AGENT_HOME"] = str(Path(tmp) / "home")
            try:
                paths = ConfigPaths(Path(os.environ["HIGH_AGENT_HOME"]), Path(os.environ["HIGH_AGENT_HOME"]) / "config.yaml", Path(os.environ["HIGH_AGENT_HOME"]) / "secrets.yaml")
                save_config(
                    {
                        "model": {
                            "provider": "custom",
                            "model": "m",
                            "base_url": "https://example.test/v1",
                            "api_mode": "chat_completions",
                        },
                        "runtime": {"delivery_debounce_seconds": 0.12},
                    },
                    paths,
                )
                save_secrets({"providers": {"custom": {"api_key": "key"}}}, paths)
                args = Namespace(
                    provider=None,
                    model=None,
                    base_url=None,
                    api_mode=None,
                    api_key=None,
                    max_workers=None,
                    workspace=tmp,
                    yes=False,
                    trace=False,
                    trace_path=None,
                    model_timeout=None,
                    delivery_debounce=None,
                )
                agent = _build_agent(args, "inspect")
                self.addCleanup(agent.runtime.shutdown)
                self.assertEqual(agent.runtime.delivery_debounce, 0.12)

                os.environ["HIGH_AGENT_DELIVERY_DEBOUNCE_SECONDS"] = "0.03"
                agent_env = _build_agent(args, "inspect")
                self.addCleanup(agent_env.runtime.shutdown)
                self.assertEqual(agent_env.runtime.delivery_debounce, 0.03)

                args.delivery_debounce = 0.0
                agent_cli = _build_agent(args, "inspect")
                self.addCleanup(agent_cli.runtime.shutdown)
                self.assertEqual(agent_cli.runtime.delivery_debounce, 0.0)
            finally:
                if old_home is None:
                    os.environ.pop("HIGH_AGENT_HOME", None)
                else:
                    os.environ["HIGH_AGENT_HOME"] = old_home
                if old_debounce is None:
                    os.environ.pop("HIGH_AGENT_DELIVERY_DEBOUNCE_SECONDS", None)
                else:
                    os.environ["HIGH_AGENT_DELIVERY_DEBOUNCE_SECONDS"] = old_debounce

    def test_build_agent_applies_max_planner_requests_from_config_and_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old_home = os.environ.get("HIGH_AGENT_HOME")
            old_max = os.environ.get("HIGH_AGENT_MAX_PLANNER_REQUESTS")
            os.environ["HIGH_AGENT_HOME"] = str(Path(tmp) / "home")
            try:
                paths = ConfigPaths(Path(os.environ["HIGH_AGENT_HOME"]), Path(os.environ["HIGH_AGENT_HOME"]) / "config.yaml", Path(os.environ["HIGH_AGENT_HOME"]) / "secrets.yaml")
                save_config(
                    {
                        "model": {
                            "provider": "custom",
                            "model": "m",
                            "base_url": "https://example.test/v1",
                            "api_mode": "chat_completions",
                        },
                        "runtime": {"max_planner_requests": 3},
                    },
                    paths,
                )
                save_secrets({"providers": {"custom": {"api_key": "key"}}}, paths)
                args = Namespace(
                    provider=None,
                    model=None,
                    base_url=None,
                    api_mode=None,
                    api_key=None,
                    max_workers=None,
                    workspace=tmp,
                    yes=False,
                    trace=False,
                    trace_path=None,
                    model_timeout=None,
                    delivery_debounce=None,
                )
                agent = _build_agent(args, "inspect")
                self.addCleanup(agent.runtime.shutdown)
                self.assertEqual(agent.max_planner_requests, 3)

                os.environ["HIGH_AGENT_MAX_PLANNER_REQUESTS"] = "4"
                agent_env = _build_agent(args, "inspect")
                self.addCleanup(agent_env.runtime.shutdown)
                self.assertEqual(agent_env.max_planner_requests, 4)
            finally:
                if old_home is None:
                    os.environ.pop("HIGH_AGENT_HOME", None)
                else:
                    os.environ["HIGH_AGENT_HOME"] = old_home
                if old_max is None:
                    os.environ.pop("HIGH_AGENT_MAX_PLANNER_REQUESTS", None)
                else:
                    os.environ["HIGH_AGENT_MAX_PLANNER_REQUESTS"] = old_max

    def test_build_agent_defaults_to_four_planner_requests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old_home = os.environ.get("HIGH_AGENT_HOME")
            old_max = os.environ.get("HIGH_AGENT_MAX_PLANNER_REQUESTS")
            os.environ["HIGH_AGENT_HOME"] = str(Path(tmp) / "home")
            try:
                paths = ConfigPaths(Path(os.environ["HIGH_AGENT_HOME"]), Path(os.environ["HIGH_AGENT_HOME"]) / "config.yaml", Path(os.environ["HIGH_AGENT_HOME"]) / "secrets.yaml")
                save_config(
                    {
                        "model": {
                            "provider": "custom",
                            "model": "m",
                            "base_url": "https://example.test/v1",
                            "api_mode": "chat_completions",
                        }
                    },
                    paths,
                )
                save_secrets({"providers": {"custom": {"api_key": "key"}}}, paths)
                args = Namespace(
                    provider=None,
                    model=None,
                    base_url=None,
                    api_mode=None,
                    api_key=None,
                    max_workers=None,
                    workspace=tmp,
                    yes=False,
                    trace=False,
                    trace_path=None,
                    model_timeout=None,
                    delivery_debounce=None,
                )
                os.environ.pop("HIGH_AGENT_MAX_PLANNER_REQUESTS", None)
                agent = _build_agent(args, "inspect")
                self.addCleanup(agent.runtime.shutdown)
                self.assertEqual(agent.max_planner_requests, 4)
            finally:
                if old_home is None:
                    os.environ.pop("HIGH_AGENT_HOME", None)
                else:
                    os.environ["HIGH_AGENT_HOME"] = old_home
                if old_max is None:
                    os.environ.pop("HIGH_AGENT_MAX_PLANNER_REQUESTS", None)
                else:
                    os.environ["HIGH_AGENT_MAX_PLANNER_REQUESTS"] = old_max


class AgentLoopTests(unittest.TestCase):
    def test_main_agent_runs_tool_call_and_returns_final(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = CausalRuntime(workspace_root=tmp, delivery_debounce=0.0)
            self.addCleanup(runtime.shutdown)
            fake = _FakeModel(
                [
                    NormalizedResponse(None, [ToolCall("call-1", "write_file", json.dumps({"path": "a.txt", "content": "A"}))], "tool_calls"),
                    NormalizedResponse("done", None, "stop"),
                ]
            )
            agent = MainAgent("write", runtime, model_client=fake, tools=create_core_registry())
            self.assertEqual(agent.run("write"), "done")
            self.assertEqual((Path(tmp) / "a.txt").read_text(encoding="utf-8"), "A")
            self.assertTrue(any("tool_call_id=call-1" in str(message) for message in agent.last_messages))

    def test_delegate_task_lowers_to_worker_tasks_and_completes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = CausalRuntime(workspace_root=tmp, delivery_debounce=0.0)
            self.addCleanup(runtime.shutdown)
            fake = _FakeModel(
                [
                    NormalizedResponse(None, [ToolCall("call-1", "delegate_task", json.dumps({"goal": "inspect", "tasks": [{"goal": "sub1"}, {"goal": "sub2"}]}))], "tool_calls"),
                    NormalizedResponse("final done", None, "stop"),
                ]
            )
            agent = MainAgent("delegate", runtime, model_client=fake, tools=create_core_registry(), max_planner_requests=1)
            self.assertEqual(agent.run("delegate"), "final done")
            self.assertEqual(len(fake.calls), 2)
            results = runtime.collect(set(runtime._tasks))
            worker_results = [r for tid, r in results.items() if "worker" in tid]
            self.assertEqual(len(worker_results), 2)
            self.assertTrue(all(r.status == "completed" for r in worker_results))

    def test_main_agent_uses_conversation_history_and_delivery_callback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = CausalRuntime(workspace_root=tmp, delivery_debounce=0.0)
            self.addCleanup(runtime.shutdown)
            fake = _FakeModel(
                [
                    NormalizedResponse(None, [ToolCall("call-1", "noop", "{}")], "tool_calls"),
                    NormalizedResponse("done", None, "stop"),
                ]
            )
            deliveries: list[DeliveryBatch] = []
            agent = MainAgent("objective", runtime, model_client=fake, tools=create_core_registry())
            answer = agent.run(
                "next",
                conversation_history=[{"role": "user", "content": "previous"}, {"role": "assistant", "content": "prior answer"}],
                on_delivery=deliveries.append,
            )
            self.assertEqual(answer, "done")
            self.assertTrue(deliveries)
            self.assertTrue(any(msg.get("content") == "previous" for msg in fake.calls[0]))

    def test_main_agent_estimates_usage_when_provider_omits_usage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = CausalRuntime(workspace_root=tmp, delivery_debounce=0.0)
            self.addCleanup(runtime.shutdown)
            fake = _FakeModel([NormalizedResponse("done", None, "stop")])
            agent = MainAgent("estimate usage", runtime, model_client=fake, tools=create_core_registry(), max_planner_requests=1)
            self.assertEqual(agent.run("estimate usage"), "done")
            self.assertEqual(agent.last_usage.model_calls, 1)
            self.assertGreater(agent.last_usage.context_estimate, 0)
            self.assertGreater(agent.last_usage.input_tokens, 0)
            self.assertGreater(agent.last_usage.output_tokens, 0)

    def test_main_agent_exposes_context_estimate_before_model_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = CausalRuntime(workspace_root=tmp, delivery_debounce=0.0)
            self.addCleanup(runtime.shutdown)
            agent = MainAgent("estimate before failure", runtime, model_client=_FailingModel(), tools=create_core_registry())
            answer = agent.run("estimate before failure")
            self.assertGreater(agent.last_usage.context_estimate, 0)

class InteractiveSessionTests(unittest.TestCase):
    def test_session_commands_and_turn_history(self) -> None:
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
                self.assertIsNone(session._handle_command("/tools"))
                self.assertIsNone(session._handle_command("/yes"))
                self.assertTrue(session.allow_yes)
                self.assertIsNone(session._handle_command("/workspace ."))
                session._run_user_turn("hello")
                self.assertEqual(session.conversation_history[-1]["content"], "session answer")
                self.assertIn("fake delivery", session.last_digest)
                self.assertEqual(session.session_usage["model_calls"], 1)
                self.assertEqual(session.session_usage["input_tokens"], 5)
                self.assertEqual(session.session_usage["output_tokens"], 3)
                self.assertIn("tok=5/3/8", session._toolbar())
                self.assertIn("calls=1", session._toolbar())
                self.assertEqual(int(session.turn_usage["turn_count"]), 1)
                self.assertGreaterEqual(session.turn_usage["last_turn_seconds"], 0.0)
                self.assertIn("turn=", session._toolbar())
                self.assertAlmostEqual(session.task_usage["task_seconds"], 0.15)
                self.assertIn("task=150ms", session._toolbar())
                self.assertIn("time:", session.last_tasks)
                self.assertIsNone(session._handle_command("/status"))
                self.assertTrue(any("turn_time:" in line for line in console.lines))
                self.assertEqual(session._usage_snapshots, {})
                self.assertEqual(session._task_time_snapshots, {})
                self.assertTrue(session._transcript_path.exists())
            finally:
                if old_home is None:
                    os.environ.pop("HIGH_AGENT_HOME", None)
                else:
                    os.environ["HIGH_AGENT_HOME"] = old_home

    def test_turn_usage_count_is_thread_safe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old_home = os.environ.get("HIGH_AGENT_HOME")
            os.environ["HIGH_AGENT_HOME"] = str(Path(tmp) / "home")
            try:
                session = InteractiveSession(
                    agent_factory=lambda prompt: _FakeAgent(),
                    workspace=tmp,
                    allow_yes=False,
                    console=_CollectConsole(),
                )
                n_threads = 32
                updates_per_thread = 200
                barrier = threading.Barrier(n_threads)

                def worker() -> None:
                    barrier.wait()
                    for _ in range(updates_per_thread):
                        session._record_turn_completion(0.0)

                threads = [threading.Thread(target=worker) for _ in range(n_threads)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()

                self.assertEqual(int(session.turn_usage["turn_count"]), n_threads * updates_per_thread)
                self.assertEqual(session.turn_usage["total_turn_seconds"], 0.0)
            finally:
                if old_home is None:
                    os.environ.pop("HIGH_AGENT_HOME", None)
                else:
                    os.environ["HIGH_AGENT_HOME"] = old_home


class _FakeResponse:
    status_code = 200
    text = ""

    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def json(self) -> dict:
        return self.payload


class _FakeHttpClient:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.requests: list[dict] = []

    def post(self, url: str, headers: dict, json: dict) -> _FakeResponse:
        self.requests.append({"url": url, "headers": headers, "json": json})
        return _FakeResponse(self.payload)


class _TimeoutHttpClient:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc
        self.requests = 0

    def post(self, url: str, headers: dict, json: dict) -> _FakeResponse:
        self.requests += 1
        raise self.exc


class _FakeStreamHttpClient:
    def __init__(self, lines: list[str], *, status_code: int = 200, content_type: str = "text/event-stream") -> None:
        self.lines = lines
        self.status_code = status_code
        self.content_type = content_type
        self.stream_requests: list[dict] = []

    def stream(self, method: str, url: str, headers: dict, json: dict) -> "_FakeStreamResponse":
        self.stream_requests.append({"method": method, "url": url, "headers": headers, "json": json})
        return _FakeStreamResponse(self.lines, status_code=self.status_code, content_type=self.content_type)


class _FakeStreamResponse:
    def __init__(self, lines: list[str], *, status_code: int, content_type: str) -> None:
        self.lines = lines
        self.status_code = status_code
        self.headers = {"content-type": content_type}
        self.text = ""

    def __enter__(self) -> "_FakeStreamResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def iter_lines(self):
        return iter(self.lines)

    def read(self) -> bytes:
        return "\n".join(self.lines).encode("utf-8")


class _FakeModel:
    def __init__(self, responses: list[NormalizedResponse]) -> None:
        self.responses = responses
        self.calls: list[list[dict]] = []
        self.lock = threading.Lock()

    def complete(self, messages: list[dict], tools: list[dict] | None = None, **params) -> NormalizedResponse:
        with self.lock:
            self.calls.append([dict(message) for message in messages])
            if not self.responses:
                return NormalizedResponse("", None, "stop")
            return self.responses.pop(0)


class _FailingModel:
    def complete(self, messages: list[dict], tools: list[dict] | None = None, **params) -> NormalizedResponse:
        raise RuntimeError("model failed")


class _FakeRuntime:
    def __init__(self) -> None:
        self.ledger = _FakeLedger()
        self.trace = _FakeTrace()

    def shutdown(self) -> None:
        pass

    def status_digest(self):
        class Digest:
            text = "runtime idle"

        return Digest()


class _FakeTrace:
    path = None

    def emit(self, event, **payload):
        pass

    def emit_typed(self, event, **payload):
        pass


class _FakeLedger:
    def counts(self):
        return {"completed": 1}

    def timing(self):
        from high_agent.runtime.ledger import TaskTimingStats

        return TaskTimingStats(wall_seconds=0.2, task_seconds=0.15, completed_task_seconds=0.15, completed_tasks=1)


class _FakeAgent:
    def __init__(self) -> None:
        from high_agent.agent.controller import RunUsage

        self.runtime = _FakeRuntime()
        self.last_usage = RunUsage(input_tokens=5, output_tokens=3, total_tokens=8, model_calls=1, context_estimate=20)

    def run(self, prompt: str, **kwargs) -> str:
        event = DeliveryEvent(
            seq=1,
            task_id="task-1",
            kind="tool",
            summary="tool done",
            result=TaskResult.completed("tool done"),
        )
        kwargs["on_delivery"](DeliveryBatch(events=[event], digest="fake delivery", batch_seq=1))
        return "session answer"


class _CollectConsole:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def print(self, *values: object, **kwargs: object) -> None:
        self.lines.append(" ".join(str(value) for value in values))


if __name__ == "__main__":
    unittest.main()
