"""Standalone hermes-agent runner — invoked as subprocess by HermesAgentAdapter.

Reads a JSON task spec from stdin, runs hermes AIAgent with the configured
model, and writes a JSON trace to stdout. Stderr is for logs/diagnostics.

This script is intended to run under a separate venv (Python 3.13 with GIL),
isolated from the high_agent 3.13t (noGIL) main venv.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def _setup_path() -> Path:
    here = Path(__file__).resolve()
    repo_root = here.parent.parent.parent
    hermes_dir = repo_root / "hermes-agent"
    if str(hermes_dir) not in sys.path:
        sys.path.insert(0, str(hermes_dir))
    return hermes_dir


def main() -> int:
    payload = json.loads(sys.stdin.read())
    task_id = payload.get("task_id", "")
    prompt = payload.get("prompt", "")
    workspace = payload.get("workspace", "")
    timeout = float(payload.get("timeout", 120.0))
    max_iterations = int(payload.get("max_iterations", 50))
    model = payload.get("model", "")
    base_url = payload.get("base_url", "")
    api_key = payload.get("api_key", "")

    if workspace:
        os.chdir(workspace)

    hermes_dir = _setup_path()

    result: dict = {
        "task_id": task_id,
        "agent_name": "hermes_agent",
        "start_time": time.time(),
        "tool_calls": [],
        "tool_results": [],
        "final_answer": "",
        "error": "",
        "total_tokens": 0,
        "model_calls": 0,
    }

    try:
        from run_agent import AIAgent  # type: ignore
    except Exception as exc:
        result["error"] = f"hermes import failed: {exc}"
        result["end_time"] = time.time()
        sys.stdout.write(json.dumps(result))
        return 1

    try:
        agent = AIAgent(
            base_url=base_url,
            api_key=api_key,
            provider="openai_compatible",
            api_mode="chat_completions",
            model=model,
            max_iterations=max_iterations,
            quiet_mode=True,
            verbose_logging=False,
            save_trajectories=False,
            skip_context_files=True,
            skip_memory=True,
            load_soul_identity=False,
        )

        response = agent.run_conversation(prompt)
        if isinstance(response, dict):
            result["final_answer"] = str(response.get("final_response", ""))
            result["model_calls"] = int(response.get("api_calls", 0) or 0)
            result["total_tokens"] = int(response.get("total_tokens", 0) or 0)
            messages = response.get("messages") or []
        else:
            result["final_answer"] = str(response)
            messages = list(getattr(agent, "messages", []) or [])

        for msg in messages:
            role = msg.get("role")
            if role == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    fn = tc.get("function", {}) or {}
                    args = fn.get("arguments", "{}")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except (json.JSONDecodeError, TypeError):
                            args = {"raw": args}
                    result["tool_calls"].append({
                        "name": fn.get("name", "unknown"),
                        "arguments": args,
                        "call_id": tc.get("id", ""),
                    })
            elif role == "tool":
                content = str(msg.get("content", ""))
                result["tool_results"].append({
                    "call_id": msg.get("tool_call_id", ""),
                    "output": content[:500],
                    "success": "error" not in content.lower(),
                })

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    result["end_time"] = time.time()
    sys.stdout.write(json.dumps(result, ensure_ascii=False))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
