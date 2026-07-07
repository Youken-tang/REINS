"""hermes-agent adapter for benchmark.

Runs hermes-agent as a subprocess under a separate Python 3.13 (with GIL) venv,
to keep the comparison framework isolated from the high_agent 3.13t (noGIL)
main runtime. This isolates the free-threaded CPython under test from
baselines that run on stock CPython.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from benchmark.adapters import TaskTrace, ToolCall, ToolResult
from benchmark.adapters.base import AgentAdapter, TaskInput

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_HERMES_DIR = _REPO_ROOT / "hermes-agent"
_HERMES_VENV_PY = _REPO_ROOT / ".venv-hermes" / "bin" / "python"
_RUNNER_SCRIPT = Path(__file__).resolve().parent / "_hermes_runner.py"


def _load_secret_api_key() -> str:
    """Load API key from ~/.config/high-agent/secrets.yaml (best effort)."""
    secrets_path = Path.home() / ".config" / "high-agent" / "secrets.yaml"
    if not secrets_path.exists():
        return ""
    try:
        import yaml
    except ImportError:
        return ""
    try:
        with open(secrets_path) as f:
            secrets = yaml.safe_load(f) or {}
        providers = secrets.get("providers", {}) or {}
        for _, prov in providers.items():
            if isinstance(prov, dict) and prov.get("api_key"):
                return str(prov["api_key"])
    except Exception:
        pass
    return ""


class HermesAgentAdapter(AgentAdapter):
    """Adapter wrapping hermes-agent's AIAgent (sequential runner) via subprocess."""

    def __init__(self) -> None:
        self._model = ""
        self._base_url = ""
        self._api_key = ""
        self._python_bin = ""

    @property
    def name(self) -> str:
        return "hermes_agent"

    def setup(self, model: str, base_url: str, **kwargs: Any) -> None:
        self._model = model
        self._base_url = base_url
        self._api_key = kwargs.get("api_key") or _load_secret_api_key()
        self._python_bin = (
            kwargs.get("python_bin")
            or (str(_HERMES_VENV_PY) if _HERMES_VENV_PY.exists() else sys.executable)
        )

    def run_task(self, task: TaskInput) -> TaskTrace:
        trace = TaskTrace(
            task_id=task.task_id,
            agent_name=self.name,
            start_time=time.time(),
        )

        if not _RUNNER_SCRIPT.exists():
            trace.error = f"hermes runner script missing: {_RUNNER_SCRIPT}"
            trace.end_time = time.time()
            return trace

        if not _HERMES_DIR.exists():
            trace.error = f"hermes-agent source not found: {_HERMES_DIR}"
            trace.end_time = time.time()
            return trace

        payload = {
            "task_id": task.task_id,
            "prompt": task.prompt,
            "workspace": task.workspace,
            "timeout": task.timeout,
            "max_iterations": task.max_iterations,
            "model": self._model,
            "base_url": self._base_url,
            "api_key": self._api_key,
        }

        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        if self._api_key:
            env.setdefault("OPENAI_API_KEY", self._api_key)

        try:
            proc = subprocess.run(
                [self._python_bin, str(_RUNNER_SCRIPT)],
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                timeout=task.timeout + 30.0,
                cwd=task.workspace,
                env=env,
            )
            stdout = proc.stdout or ""
            stderr_tail = (proc.stderr or "").strip()[-500:]

            data: dict[str, Any] | None = None
            for line in reversed(stdout.splitlines()):
                line = line.strip()
                if line.startswith("{") and line.endswith("}"):
                    try:
                        data = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        continue

            if data is None:
                trace.error = (
                    f"no JSON trace from hermes runner (exit {proc.returncode})"
                    + (f": {stderr_tail}" if stderr_tail else "")
                )
            else:
                trace.final_answer = str(data.get("final_answer") or "")
                trace.error = str(data.get("error") or "")
                trace.total_tokens = int(data.get("total_tokens") or 0)
                trace.model_calls = int(data.get("model_calls") or 0)

                for tc in data.get("tool_calls", []) or []:
                    trace.tool_calls.append(ToolCall(
                        name=tc.get("name", "unknown"),
                        arguments=tc.get("arguments", {}) or {},
                        call_id=tc.get("call_id", ""),
                    ))
                for tr in data.get("tool_results", []) or []:
                    trace.tool_results.append(ToolResult(
                        call_id=tr.get("call_id", ""),
                        output=str(tr.get("output", ""))[:500],
                        success=bool(tr.get("success", False)),
                    ))

                if proc.returncode != 0 and not trace.error and stderr_tail:
                    trace.error = f"exit {proc.returncode}: {stderr_tail}"

        except subprocess.TimeoutExpired:
            trace.error = f"hermes timeout after {task.timeout}s"
        except FileNotFoundError as exc:
            trace.error = f"python binary not found: {self._python_bin} ({exc})"
        except Exception as exc:
            trace.error = f"{type(exc).__name__}: {exc}"
        finally:
            trace.end_time = time.time()

        return trace

    def teardown(self) -> None:
        pass

    def supports_parallel(self) -> bool:
        return False

    def supports_delegation(self) -> bool:
        return False
