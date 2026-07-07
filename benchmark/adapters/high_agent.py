"""high_agent adapter for benchmark with profile support and enhanced metrics."""

from __future__ import annotations

import threading
import time
import types
from pathlib import Path
from typing import Any

from benchmark.adapters import TaskTrace, ToolCall, ToolResult
from benchmark.adapters.base import AgentAdapter, TaskInput
from benchmark.profiles import RuntimeProfile


class HighAgentAdapter(AgentAdapter):
    """Adapter wrapping high_agent's CausalRuntime + AgentRunController."""

    def __init__(self) -> None:
        self._model = ""
        self._base_url = ""
        self._profile: RuntimeProfile | None = None

    @property
    def name(self) -> str:
        return "high_agent"

    def setup(self, model: str, base_url: str, *, profile: RuntimeProfile | None = None, **kwargs: Any) -> None:
        self._model = model
        self._base_url = base_url
        self._profile = profile

    def run_task(self, task: TaskInput) -> TaskTrace:
        from high_agent.runtime.scheduler import CausalRuntime
        from high_agent.agent.controller import AgentRunController, RunUsage
        from high_agent.agent.main import MainAgent
        from high_agent.tools.core import create_core_registry
        from high_agent.tools.result_store import ToolResultStore

        trace = TaskTrace(
            task_id=task.task_id,
            agent_name=self.name,
            start_time=time.time(),
        )

        result_store = ToolResultStore()
        registry = create_core_registry(
            allow_terminal=True,
            allow_outside_workspace=False,
            result_store=result_store,
        )

        # Build runtime with profile overrides
        runtime_kwargs: dict[str, Any] = {
            "max_workers": 8,
            "workspace_root": task.workspace,
            "strict_nogil": False,
        }
        if self._profile:
            runtime_kwargs.update(self._profile.runtime_overrides)

        trace_path = Path(task.workspace) / ".benchmark_trace.jsonl"
        runtime_kwargs["trace_path"] = str(trace_path)

        runtime = CausalRuntime(**runtime_kwargs)

        # Apply adapter hooks
        if self._profile:
            self._apply_hooks(runtime, self._profile)

        client = self._create_client()

        # Disable streaming dispatch if requested
        if self._profile and self._profile.adapter_hooks.get("disable_streaming_dispatch"):
            client.complete_streaming = None

        # Unbounded context hook
        saved_context_limit = None
        if self._profile and self._profile.adapter_hooks.get("unbounded_context"):
            import high_agent.agent.controller as ctrl_mod
            saved_context_limit = ctrl_mod.MAX_PLANNER_CONTEXT_CHARS
            ctrl_mod.MAX_PLANNER_CONTEXT_CHARS = 1_000_000

        # Parallelism sampling thread
        parallelism_samples: list[tuple[float, int]] = []
        stop_sampling = threading.Event()
        sample_thread = threading.Thread(
            target=self._sample_parallelism,
            args=(runtime, parallelism_samples, stop_sampling),
            daemon=True,
        )

        controller = None
        try:
            runtime.start()
            sample_thread.start()

            # Build controller kwargs with profile overrides
            controller_kwargs: dict[str, Any] = {
                "delivery_timeout": min(task.timeout, 30.0),
                "max_iterations": task.max_iterations,
            }
            if self._profile:
                controller_kwargs.update(self._profile.controller_overrides)

            max_planner_requests = controller_kwargs.pop("max_planner_requests", 4)

            agent = MainAgent(
                objective=task.prompt,
                runtime=runtime,
                model_client=client,
                tools=registry,
                max_planner_requests=max_planner_requests,
            )

            cancel_event = threading.Event()
            controller = AgentRunController(
                agent=agent,
                objective=task.prompt,
                messages=[{"role": "user", "content": task.prompt}],
                cancel_event=cancel_event,
                **controller_kwargs,
            )

            result_box: dict[str, Any] = {}

            def _run_controller() -> None:
                try:
                    result_box["final_answer"] = controller.run()
                except Exception as exc:
                    result_box["error"] = exc

            run_thread = threading.Thread(
                target=_run_controller,
                name="high-agent-controller",
                daemon=True,
            )
            run_thread.start()
            run_thread.join(task.timeout)
            if run_thread.is_alive():
                cancel_event.set()
                # Wake any in-flight wait_delivery / dispatch loops so the
                # controller thread can observe cancel_event and unwind.
                try:
                    runtime.shutdown()
                except Exception:
                    pass
                run_thread.join(timeout=5.0)
                trace.error = f"wall_clock_timeout after {task.timeout}s"
            else:
                if "error" in result_box:
                    trace.error = str(result_box["error"])
                else:
                    trace.final_answer = result_box.get("final_answer") or ""
            trace.total_tokens = controller.usage.total_tokens
            trace.model_calls = controller.usage.model_calls

        except Exception as exc:
            trace.error = str(exc)
        finally:
            stop_sampling.set()
            sample_thread.join(timeout=1.0)
            runtime.shutdown()
            trace.end_time = time.time()

            # Restore context limit
            if saved_context_limit is not None:
                import high_agent.agent.controller as ctrl_mod
                ctrl_mod.MAX_PLANNER_CONTEXT_CHARS = saved_context_limit

        # Extract enhanced metrics
        self._extract_metrics(trace, runtime, controller, parallelism_samples, trace_path)

        return trace

    def _apply_hooks(self, runtime: Any, profile: RuntimeProfile) -> None:
        """Apply adapter hooks to the runtime instance."""
        if profile.adapter_hooks.get("disable_conflict_detection"):
            runtime._first_conflict_locked = lambda access: None

        if profile.adapter_hooks.get("fixed_debounce"):
            pass  # delivery_debounce already set via runtime_overrides

    def _create_client(self) -> Any:
        """Create ModelClient from stored model/base_url config."""
        from high_agent.llm.client import ModelClient
        from high_agent.llm.providers import ModelSettings

        import yaml
        from pathlib import Path

        api_key = ""
        secrets_path = Path.home() / ".config" / "high-agent" / "secrets.yaml"
        if secrets_path.exists():
            with open(secrets_path) as f:
                secrets = yaml.safe_load(f) or {}
            providers = secrets.get("providers", {})
            for _, prov_data in providers.items():
                if isinstance(prov_data, dict) and "api_key" in prov_data:
                    api_key = prov_data["api_key"]
                    break

        settings = ModelSettings(
            provider="custom",
            model=self._model,
            base_url=self._base_url,
            api_mode="chat_completions",
            api_key=api_key,
        )
        return ModelClient(settings)

    def _sample_parallelism(
        self,
        runtime: Any,
        samples: list[tuple[float, int]],
        stop: threading.Event,
    ) -> None:
        """Background thread that samples running task count every 100ms."""
        while not stop.is_set():
            try:
                timing = runtime.ledger.timing()
                samples.append((time.time(), timing.running_tasks))
            except Exception:
                pass
            stop.wait(0.1)

    def _extract_metrics(
        self,
        trace: TaskTrace,
        runtime: Any,
        controller: Any,
        parallelism_samples: list[tuple[float, int]],
        trace_path: Path,
    ) -> None:
        """Fill enhanced metrics into the TaskTrace."""
        try:
            timing = runtime.ledger.timing()
            trace.task_seconds = timing.task_seconds
            trace.peak_parallelism = max(
                (count for _, count in parallelism_samples),
                default=timing.running_tasks,
            )
            trace.parallelism_timeline = parallelism_samples
            trace.conflict_count = len(getattr(runtime.ledger, "_conflicts", []))
            trace.batch_count = getattr(runtime, "_batch_seq", 0)
            trace.total_dispatch_count = sum(
                1 for rec in runtime.ledger._records.values()
                if rec.state in ("completed", "failed", "cancelled")
            )

            # Extract from trace file for deeper metrics
            if trace_path.exists():
                from benchmark.metrics import extract_from_trace
                file_metrics = extract_from_trace(trace_path)
                trace.planning_stall_seconds = file_metrics.planning_stall_seconds
                trace.streaming_dispatch_count = file_metrics.streaming_dispatch_count
                if not trace.parallelism_timeline:
                    trace.parallelism_timeline = file_metrics.parallelism_timeline
        except Exception:
            pass

        trace.metadata["profile"] = self._profile.name if self._profile else "high_agent"
        trace.metadata["parallel_batches"] = trace.batch_count
        trace.metadata["max_concurrent"] = runtime.max_workers

    def teardown(self) -> None:
        pass

    def supports_parallel(self) -> bool:
        if self._profile and self._profile.name == "sequential":
            return False
        return True

    def supports_delegation(self) -> bool:
        return True
