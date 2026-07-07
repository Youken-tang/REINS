# benchmark/adapters — Reins-into-hermes bridge

Three modules, one test surface, zero modifications to `hermes-agent/`
(hermes-agent is reference code, not part of this project's commits).

## Module map

| File | Role | Hermes counterpart |
| --- | --- | --- |
| [`hermes_reins_runtime.py`](hermes_reins_runtime.py) | Static `ResourceAccess` table for hermes' 14 real-annotation tools + terminal classifier hand-off + ordered `dispatch_tool_calls` | `_PARALLEL_SAFE_TOOLS` / `_PATH_SCOPED_TOOLS` at [`run_agent.py:308-326`](../../hermes-agent/run_agent.py#L308) |
| [`hermes_reins_loop.py`](hermes_reins_loop.py) | Drop-in `execute_tool_calls(assistant_message, messages, *, invoke_tool, runtime, root)` | [`_execute_tool_calls`](../../hermes-agent/run_agent.py#L9098) at run_agent.py:9098 |
| [`runtime_host.py`](runtime_host.py) | `HermesReinsHost` — process-wide `CausalRuntime` lifecycle + env-flag plumbing | (new — hermes has no equivalent today) |
| [`smoke.py`](smoke.py) | 5-prompt offline smoke (deterministic; uses `time.sleep` to simulate IO) | substitutes for W2 «5 prompts on `mini_swe_runner.py`» until hermes-side wired |

## Hermes-side wiring (W3 sweep prep)

Per CLAUDE.md `hermes-agent/` does not accept patches from this project. The minimal in-tree diff to land at `hermes-agent/reins_bridge/` once the bridge moves home:

```python
# hermes-agent/reins_bridge/__init__.py
from benchmark.adapters.runtime_host import HermesReinsHost, get_default_host
from benchmark.adapters.hermes_reins_loop import execute_tool_calls

__all__ = ["HermesReinsHost", "get_default_host", "execute_tool_calls"]
```

Then at the [`_execute_tool_calls`](../../hermes-agent/run_agent.py#L9098) call site, replace the body with one call:

```python
def _execute_tool_calls(self, assistant_message, messages, effective_task_id, api_call_count=0):
    from reins_bridge import get_default_host
    host = get_default_host(workspace_root=self._workspace_root)
    host.dispatch(
        assistant_message,
        messages,
        invoke_tool=lambda name, args: self._invoke_tool(
            name, args, effective_task_id, tool_call_id=None, messages=messages,
        ),
    )
```

Hermes' context engine, system prompt, model client, approval policy, plugin hooks: all unchanged. `_execute_tool_calls_concurrent` and `_execute_tool_calls_sequential` can stay as dead code or be deleted — the bridge handles both paths.

## Env flags

| Variable | Effect | Default |
| --- | --- | --- |
| `HERMES_REINS` | `0`/`false`/`no`/`off` → vanilla sequential path; anything else (or unset) → REINS | on |
| `HERMES_REINS_MAX_WORKERS` | `CausalRuntime` worker pool size | `8` |
| `HERMES_REINS_DEBOUNCE` | `delivery_debounce` seconds | `0.05` |
| `HERMES_REINS_TIMEOUT` | per-batch dispatch timeout (seconds) | `600.0` |
| `HERMES_REINS_TRACE_PATH` | runtime trace JSONL output (consumed by [`benchmark/scripts/parse_trace.py`](../scripts/parse_trace.py)) | unset → no trace |

The sweep harness flips `HERMES_REINS=0` for the `hermes-vanilla` row and `HERMES_REINS=1 HERMES_REINS_TRACE_PATH=results/{run_id}/trace.jsonl` for the `hermes-REINS` row. Same call site, same host, single env-var diff.

## Tool tier coverage (per spike

68 hermes tools total at [`hermes-agent/tools/`](../../hermes-agent/tools/):

* **11 read-only safe** (port of `_PARALLEL_SAFE_TOOLS`): `ha_get_state`, `ha_list_entities`, `ha_list_services`, `read_file`, `search_files`, `session_search`, `skill_view`, `skills_list`, `vision_analyze`, `web_extract`, `web_search`
* **2 path-scoped mutators** (port of `_PATH_SCOPED_TOOLS`): `write_file`, `patch`
* **1 shell** (`terminal`) — hands off to `high_agent.tools.core._process_resource_access` ladder
* **14 agent-internal trivial** (no shared workspace state): `clarify`, `delegate`, `memory_*`, `skill_*`, `todo_*`, `interrupt`, `checkpoint_create`
* **40 unknown fallback** (`browser_*`, `feishu_*`, `mcp_*`, `discord`, `homeassistant`, `image_generation`, `neutts_*`, `send_message`, `vision_*` non-analyze, `voice_*`, `web_*` non-search, `xai_*`, `yuanbao_*`, …) — default to `ResourceAccess.unknown_workspace()` per spike risk plan

The 40 `unknown` tools serialise globally. Per the spike: this preserves the safe baseline. The risk-plan escape hatch was reserved for ≥30 unknown tools; we hit 40, matching the «narrower parallel envelope» branch the spike anticipated.

## Test surface

| File | Tests | Role |
| --- | --- | --- |
| [`tests/test_hermes_reins_adapter.py`](../../tests/test_hermes_reins_adapter.py) | 18 | annotation table semantics, `dispatch_tool_calls` order/error/empty |
| [`tests/test_hermes_reins_loop.py`](../../tests/test_hermes_reins_loop.py) | 14 | drop-in shape, env-flag coercion, sequential vs REINS, edge cases |
| [`tests/test_hermes_reins_runtime_host.py`](../../tests/test_hermes_reins_runtime_host.py) | 13 | host lifecycle, env-flag plumbing, dispatch round-trip, default-host singleton |
| [`tests/test_hermes_reins_integration.py`](../../tests/test_hermes_reins_integration.py) | 2 | end-to-end three-batch smoke + vanilla pass-through, trace-file invariant |

Run from project root:

```bash
PYTHONPATH=src uv run --with pytest --python .venv/bin/python pytest tests/test_hermes_reins*.py -q
```

The deterministic 5-prompt offline smoke (no model needed):

```bash
PYTHONPATH=src .venv/bin/python -m benchmark.adapters.smoke
PYTHONPATH=src .venv/bin/python -m benchmark.adapters.smoke --json
```

Local results (deterministic, sleep-simulated): read-fanout 3.7×, write-fanout 2.9×, readonly-mix 2.9×; same-path-rmw and unknown-fallback both ~1.0× (correctly serialised).

## Status

* W1 — `reins_runtime` extract + adapter skeleton: done (commits [`eff42a9`](../../), [`12ea275`](../../))
* W2 — `loop.py` + `runtime_host.py` + smoke: done (commit [`8970ea8`](../../) plus this commit)
* W3 — 300-run sweep on hermes side: blocked on hermes-agent in-tree wiring (out of scope per CLAUDE.md)
