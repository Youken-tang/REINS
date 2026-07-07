# REINS-Bench

> Public benchmark for **agent schedulers**, with ground-truth
> `ResourceAccess` labels per tool call.

REINS-Bench is a 50-prompt evaluation set (30 W_fix + 20 W_scaffold)
that lets *any* agent runtime — Reins, a Ray wrapper, sequential, a
third-party framework — be scored on the same prompts with the same
gates and against the same baseline.

**Public name:** `REINS-Bench`.
**On-disk Python package:** `reins_bench` (snake_case to keep imports
PEP-8 compatible). Don't @-mention the directory name in citations.

---

## Two metrics, never multiplied

REINS-Bench reports **pass_rate** and **scheduler_score** separately,
by design:

- **`pass_rate`** — fraction of (prompt, run) cells whose final
  workspace satisfies every gate in the prompt's `expected:` block:
  - `min_files` exist (glob or literal)
  - every `must_pass_tests` command exits zero
  - no `forbidden_patterns` regex matches in the workspace
  - the run stayed inside its `budget` (wall_seconds, input/output
    tokens)

- **`scheduler_score`** — geometric mean of
  `wall_seconds / input_tokens / output_tokens` ratios against a named
  baseline system, computed *only* over prompts where **both** the
  candidate and the baseline pass.

The two-axis report exists so a fast-but-corrupt scheduler cannot win.
Collapsing both axes into a single product would let one score mask
the other.

## Conflict accounting

Every cell record also reports four conflict-audit numbers:

- `resource_conflict_count` — number of `resource.conflict` events the
  scheduler emitted.
- `false_positive_conflicts` — events where the contended write set is
  disjoint from the prompt's declared `ground_truth_resource_access`
  writes (i.e. the scheduler reported a conflict on a path the oracle
  says nothing actually writes).
- `false_negative_conflicts` — declared write paths that **were**
  written during the run but never appeared in any conflict event
  (i.e. an arbitration the scheduler should have done, didn't).
- `corruption_detected` — any pair-mismatch detected by the corruption
  audit (`corruption.json` next to the cell).

When a prompt ships without `ground_truth_resource_access`, FP/FN are
zeroed (we can't audit what we don't have).

---

## Corpus layout

```text
prompts/
  w_fix/        # bug-fix tasks; smaller surface area, narrower oracle
    w_fix_001.yaml
    ...
  w_scaffold/   # green-field scaffolds; full ResourceAccess oracle
    w_scaffold_001.yaml
    ...
schema/
  task_spec.json        # JSON-schema for prompt files (validatable)
scripts/
  score.py              # CLI wrapper around runner.py score
runner.py               # run / replay / score subcommands
schema.py               # TaskSpec dataclass + YAML/JSON loader
scoring.py              # gates, conflict audit, report builder
VERSION                 # benchmark version pin
```

Each prompt is a single YAML/JSON file with these fields (full schema
in [schema/task_spec.json](schema/task_spec.json)):

- `id`, `group` (`w_fix` or `w_scaffold`), `version`, `title`,
  `languages`
- `prompt` — verbatim text handed to the agent
- `expected` — `min_files`, `must_pass_tests`, `forbidden_patterns`
- `budget` — `max_wall_seconds`, `max_input_tokens`,
  `max_output_tokens`
- `ground_truth_resource_access` — list of
  `{tool, args, reads, writes, appends, side_effect_level}` rows,
  one per tool call in the canonical reference solution
- `notes` — provenance (`source`, `source_ref`, `license`)

The oracle is a **lower bound**, not a script. The benchmark does not
require the agent to call tools in this exact sequence — it only
requires the *final* set of writes to cover the declared `min_files`
and the conflict events to be consistent with the declared write set.

## Versioning

`VERSION` pins the dataset:

- *any* prompt or ground-truth change → bump **minor**
- adding new prompts → bump **major**

---

## Running the benchmark

From the repo root:

```bash
# Run against the full corpus, 3 repetitions each:
PYTHONPATH=src:. python -m benchmark reins-bench run \
    --system custom --adapter path/to/adapter.py \
    --corpus benchmark/reins_bench/prompts \
    --runs 3 \
    --out results/run_v1

# Score a finished results tree (no model calls; folds existing cells
# into report.json):
PYTHONPATH=src:. python -m benchmark reins-bench score \
    --root results/run_v1 \
    --system custom \
    --corpus benchmark/reins_bench/prompts \
    --out results/run_v1/report.json

# Score a candidate against a baseline tree (computes scheduler_score):
PYTHONPATH=src:. python -m benchmark reins-bench score \
    --root results/run_v1 \
    --system custom \
    --baseline-root results/baseline_v1 \
    --baseline-system baseline \
    --corpus benchmark/reins_bench/prompts \
    --out results/run_v1/report.json
```

The `replay` subcommand re-executes a captured `trace.jsonl` against an
alternate scheduler, with no model in the loop — useful for ablation
studies once alternate scheduler variants are wired in.

### Plugging in a third-party scheduler

Implement a `SchedulerAdapter` (`name: str` + `execute(...) -> CellResult`)
in a Python file, then point `--system custom` at it:

```bash
PYTHONPATH=src:. python -m benchmark reins-bench run \
    --system custom \
    --adapter path/to/my_adapter.py \
    --corpus benchmark/reins_bench/prompts \
    --out results/my_system_v1
```

The runner imports `ADAPTER` (an instance), `adapter` (an instance), or
`build()` (a callable returning an instance) from the loaded module —
whichever is found first.

---

## Licensing

REINS-Bench is **dual-licensed**:

- **Prompts and ground-truth labels** (`prompts/`, `schema/task_spec.json`):
  [CC-BY-4.0](LICENSES/PROMPTS-CC-BY-4.0.txt) — free to use, modify,
  and redistribute, including commercially, with attribution.
- **Runner code and scoring scripts** (`runner.py`, `scoring.py`,
  `schema.py`, `scripts/`): [Apache-2.0](LICENSES/CODE-APACHE-2.0.txt).

The SWE-bench Lite 50 sample **does not redistribute upstream
instances** — only the instance-id list and the `ResourceAccess`
extension. Pull the original instances through the SWE-bench upstream
harness.

## Contributing

The W_fix corpus accepts community PRs to thicken its
`ground_truth_resource_access` coverage. The W_scaffold corpus is
fully hand-annotated; new prompts there go through the maintainers.

When opening a PR:

1. Validate the new YAML against `schema/task_spec.json`.
2. Run `pytest -q` from the repo root.
3. Bump `VERSION` per the rules above.
