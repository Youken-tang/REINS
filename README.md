# Reins

Resource-aware causal runtime and reference agent for parallel LLM
tool execution on free-threaded CPython (3.13t / `Py_GIL_DISABLED=1`).

Reins lowers a model's `tool_calls` into `AgentTaskSpec` tasks and
schedules them in parallel by consulting each task's declarative
`ResourceAccess` descriptor (reads / writes / appends / unknown).
Non-conflicting tasks fan out onto a worker pool; conflicting ones
are serialized. A `RuntimeLedger` records task state and produces a
compact `LedgerDigest` that a planner consumes to decide the next
batch — so the planner keeps proposing while earlier tools are still
executing.

The repository ships two layers:

* **`reins`** — the core runtime (scheduler, ledger, context store,
 resource-access model, delivery-driven controller). No CLI, no
 provider, no plugin surface — pure library, stdlib-only.
* **`high_agent`** — a reference agent built on top of `reins`:
 provider-neutral LLM client (Chat Completions / Anthropic Messages /
 Codex Responses), planner / worker / main-agent, tool registry with
 a batteries-included tool set, memory + session store, interactive
 CLI and TUI.

You can embed just `reins` in an existing agent loop, or run the
full `high_agent` CLI as a working demonstration.

## Requirements

- CPython **3.13t** (free-threaded build). The scheduler calls
 `ensure_nogil(strict=True)` at construction; pass `strict_nogil=False`
 from a test to run on a stock interpreter.
- Runtime dependencies: `httpx`, `pyyaml`, `prompt_toolkit`, `rich`,
 `tenacity` (only for the `high_agent` layer — `reins` itself is
 stdlib-only).

## Install

```bash
pip install -e.
```

## Quick start — `reins` (embedded)

```python
from reins import AgentTaskSpec, CausalRuntime, TaskResult
from reins.runtime.resource_access import ResourceAccess


def handler_read(ctx):
 return TaskResult.completed(summary=f"read {ctx.task.input['path']}")


def handler_write(ctx):
 return TaskResult.completed(summary=f"wrote {ctx.task.input['path']}")


runtime = CausalRuntime(max_workers=4)
runtime.start

# Two reads of different files can run in parallel; the write to a.txt is
# serialized after the read of a.txt via resource-access conflict detection.
runtime.submit([
 AgentTaskSpec(
 kind="tool", goal="read a", input={"path": "a.txt"},
 resource_access=ResourceAccess.read("file:a.txt"),
 handler=handler_read,
 ),
 AgentTaskSpec(
 kind="tool", goal="read b", input={"path": "b.txt"},
 resource_access=ResourceAccess.read("file:b.txt"),
 handler=handler_read,
 ),
 AgentTaskSpec(
 kind="tool", goal="write a", input={"path": "a.txt"},
 resource_access=ResourceAccess.write("file:a.txt"),
 handler=handler_write,
 ),
])

runtime.wait_all(timeout=5.0)
while True:
 batch = runtime.wait_next_delivery(timeout=0.1)
 if batch is None:
 break
 for event in batch.events:
 print(event.task_id, event.result.status, event.summary)

runtime.shutdown
```

## Quick start — `high_agent` (full CLI)

```bash
# runtime demo (no model call — verifies scheduler)
PYTHONPATH=src python -m high_agent runtime-demo

# one-shot run
PYTHONPATH=src python -m high_agent run "list files in the current directory"

# interactive REPL
PYTHONPATH=src python -m high_agent chat
```

Configuration precedence: CLI flag > `HIGH_AGENT_*` env var >
`~/.config/high-agent/config.yaml` > default.

## Quick start — benchmarks

The `benchmark/` tree contains the end-to-end evaluation harness, a
scheduler microbenchmark, and per-axis ablation configs.

```bash
# synthetic DAG microbenchmark
PYTHONPATH=src:. python -m benchmark microbench \
 --shape chain,fanout,diamond,conflict_pair,unknown_burst \
 --n 8,16 --workers 1,2,4,8 --runs 3

# end-to-end bug-fix / scaffold corpus
PYTHONPATH=src:. python -m benchmark reins-bench run --help
PYTHONPATH=src:. python -m benchmark reins-bench score --help
```

The bug-fix corpus starts from `benchmark/reins_bench/seeds/w_fix_*/`
(30 tasks). The scaffold corpus is described in the same package's
prompt files. Custom system adapters can be registered via
`benchmark.adapters` (see `high_agent.py` / `hermes_agent.py` /
`opencode.py` for reference implementations).

## Package layout

```text
src/reins/ core runtime (stdlib-only)
├── __init__.py public API re-exports
├── _nogil.py free-threading guard
├── controller.py delivery-driven run controller
├── llm_types.py provider-neutral ToolCall / Response
├── time_utils.py duration formatting for digests
├── tool_calls.py tool_call normalization + sanitizer
└── runtime/
 ├── components.py MVCC component store
 ├── context_store.py partitioned lock-free context store
 ├── ledger.py task ledger + planner digest
 ├── resource_access.py declarative read/write descriptors
 ├── scheduler.py CausalRuntime — the scheduler
 ├── trace.py trace writer
 └── types.py AgentTaskSpec / TaskResult / DeliveryBatch

src/high_agent/ reference agent on top of reins
├── agent/ MainAgent + PlannerAgent + Worker + SubAgent
├── cli/ argparse CLI, interactive REPL, TUI
├── llm/ client, providers, chat/anthropic/responses transports
├── memory/ sqlite session + memory + compression
├── plugins/ plugin manager
├── tools/ tool registry, core tools, delegate, toolsets
├── runtime/ compatibility shims re-exporting reins.runtime
├── approval.py approval policy
├── config.py yaml config loader
└── _nogil.py / time_utils.py re-exports for compatibility

benchmark/ evaluation harness + microbench + ablation configs
├── reins_bench/ end-to-end corpus + runner + scoring
│ ├── schema.py, scoring.py TaskSpec parser, pass gates, conflict audit
│ ├── runner.py run / replay / score CLI
│ ├── prompts/ bug-fix + scaffold prompt templates
│ ├── seeds/w_fix_*/ 30 bug-fix workspaces
│ └── schema/ TaskSpec YAML schemas
├── microbench/ synthetic-DAG microbench
│ ├── dag_shapes.py chain / fanout / diamond / conflict_pair / unknown_burst
│ └── runner.py shape × N × workers × runs sweeper
├── tasks/ early category task specs (42 tasks across 5.jsonl)
│ — parallel / planning / scheduling / tool_use / realistic
├── workspaces/ starting-state fixtures for each task in tasks/
├── evaluators/ per-category evaluators consumed by tasks/
├── configs/ single-axis ablation configs
│ — resource / streaming / critpath / context / suspend
├── adapters/ system adapters (high_agent / hermes / opencode)
└── runner.py + __main__.py `python -m benchmark <experiment>` dispatch

tests/ self-contained test suite
```

## Tests

```bash
pip install -e '.[test]'
PYTHONPATH=src pytest -q
```

The bundled test suite covers the scheduler, ledger, context store,
resource-access conflict rules, controller dedupe / completion gate,
streaming transports, agent loop, tool worker, sub-agent delegation,
and the v0.3 / v0.4 protocol contracts.

## License

Apache License 2.0. See `LICENSE`.
