""" synthetic DAG builders for the scheduler microbench.

 specifies five DAG shapes that strip the runtime down
to «pure DAG + controlled tools»: chain_n, fanout_n, diamond_n,
conflict_pair, unknown_burst. Every node is an ``AgentTaskSpec`` whose
handler sleeps ``base_ms`` milliseconds and returns
``TaskResult.completed`` — no network, no model, no real subprocess. The
shape determines the dependency edges (via ``DependencyPredicate``) and
the resource access pattern (via ``ResourceAccess``), which is what we
want to vary while holding the workload constant.

Naming convention for ``ResourceAccess`` resource keys:

- ``file:{shape}/{i}.txt`` — a synthetic file per node
- ``dir:{shape}``           — a synthetic directory used by fanout
- ``external:{shape}/u{i}`` — only used by unknown_burst probes

Files do not actually exist on disk; the runtime only sees the resource
keys and runs ``access_conflicts`` over them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from high_agent.runtime.resource_access import ResourceAccess
from high_agent.runtime.types import (
    AgentTaskSpec,
    DependencyPredicate,
    TaskContext,
    TaskResult,
)


# ─────────────────────────── handler factory ───────────────────────────────


def _sleep_handler(label: str, base_ms: int) -> Callable[[TaskContext], TaskResult]:
    """Build a handler that sleeps ``base_ms`` then completes successfully.

    ``time.sleep`` releases the GIL on CPython so under noGIL it also
    releases the worker thread; that is the desired model for «轻/中/重»
    IO-bound work in the test plan.
    """

    sleep_seconds = max(0.0, float(base_ms) / 1000.0)

    def _handler(ctx: TaskContext) -> TaskResult:
        if sleep_seconds:
            time.sleep(sleep_seconds)
        return TaskResult.completed(label)

    return _handler


# ─────────────────────────── shape builders ────────────────────────────────


@dataclass
class DagSpec:
    """A built DAG ready to feed ``CausalRuntime.submit``."""

    shape: str
    n: int
    base_ms: int
    tasks: list[AgentTaskSpec]

    @property
    def task_count(self) -> int:
        return len(self.tasks)

    @property
    def task_ids(self) -> list[str]:
        return [t.task_id for t in self.tasks]


def build_chain(n: int, base_ms: int) -> DagSpec:
    """``chain_n``: linear pipeline of n nodes, each writes its own file
    and reads the upstream file. n ∈ {8, 16, 32, 64} per
    """
    if n < 1:
        raise ValueError("chain n must be >= 1")
    tasks: list[AgentTaskSpec] = []
    prev_id: str | None = None
    for i in range(n):
        tid = f"chain-{i:03d}"
        own = f"file:chain/{i}.txt"
        deps: list[DependencyPredicate] = []
        if i == 0:
            access = ResourceAccess.write(own)
        else:
            up = f"file:chain/{i - 1}.txt"
            access = ResourceAccess(
                reads=frozenset({up}),
                writes=frozenset({own}),
                side_effect_level="local",
            )
            assert prev_id is not None
            deps = [DependencyPredicate.task_completed(prev_id)]
        tasks.append(
            AgentTaskSpec(
                kind="tool",
                goal=f"chain step {i}",
                task_id=tid,
                dependencies=deps,
                resource_access=access,
                handler=_sleep_handler(f"chain-{i}", base_ms),
            )
        )
        prev_id = tid
    return DagSpec("chain", n, base_ms, tasks)


def build_fanout(n: int, base_ms: int) -> DagSpec:
    """``fanout_n``: one parent writes ``dir:{...}``; n children read that
    directory and write their own ``file:{...}/k.txt``. Children should run
    fully in parallel under ``max_workers >= n``.
    """
    if n < 1:
        raise ValueError("fanout n must be >= 1")
    parent_dir = "dir:fanout"
    parent_id = "fanout-root"
    parent = AgentTaskSpec(
        kind="tool",
        goal="fanout root",
        task_id=parent_id,
        resource_access=ResourceAccess.write(parent_dir),
        handler=_sleep_handler("fanout-root", base_ms),
    )
    tasks: list[AgentTaskSpec] = [parent]
    for k in range(n):
        child_file = f"file:fanout/{k}.txt"
        access = ResourceAccess(
            reads=frozenset({parent_dir}),
            writes=frozenset({child_file}),
            side_effect_level="local",
        )
        tasks.append(
            AgentTaskSpec(
                kind="tool",
                goal=f"fanout child {k}",
                task_id=f"fanout-c-{k:03d}",
                dependencies=[DependencyPredicate.task_completed(parent_id)],
                resource_access=access,
                handler=_sleep_handler(f"fanout-c-{k}", base_ms),
            )
        )
    return DagSpec("fanout", n, base_ms, tasks)


def build_diamond(n: int, base_ms: int) -> DagSpec:
    """``diamond_n``: root → n parallel middle nodes → reducer.

    Middle stage uses pairs of read-only siblings on the same upstream
    resource so two reads on the same key can run in parallel. The reducer
    writes a single file and depends on every middle node.
    """
    if n < 2:
        raise ValueError("diamond n must be >= 2")
    root_id = "diamond-root"
    root_resource = "file:diamond/root.txt"
    tasks: list[AgentTaskSpec] = [
        AgentTaskSpec(
            kind="tool",
            goal="diamond root",
            task_id=root_id,
            resource_access=ResourceAccess.write(root_resource),
            handler=_sleep_handler("diamond-root", base_ms),
        )
    ]
    middle_ids: list[str] = []
    for k in range(n):
        mid_id = f"diamond-m-{k:03d}"
        middle_ids.append(mid_id)
        # Read-only siblings sharing root_resource — runtime should let
        # them run concurrently because read+read does not conflict.
        access = ResourceAccess.read(root_resource)
        tasks.append(
            AgentTaskSpec(
                kind="tool",
                goal=f"diamond mid {k}",
                task_id=mid_id,
                dependencies=[DependencyPredicate.task_completed(root_id)],
                resource_access=access,
                handler=_sleep_handler(f"diamond-m-{k}", base_ms),
            )
        )
    reducer_resource = "file:diamond/result.txt"
    tasks.append(
        AgentTaskSpec(
            kind="tool",
            goal="diamond reducer",
            task_id="diamond-reducer",
            dependencies=[
                DependencyPredicate.task_completed(mid_id) for mid_id in middle_ids
            ],
            resource_access=ResourceAccess(
                reads=frozenset({root_resource}),
                writes=frozenset({reducer_resource}),
                side_effect_level="local",
            ),
            handler=_sleep_handler("diamond-reducer", base_ms),
        )
    )
    return DagSpec("diamond", n, base_ms, tasks)


def build_conflict_pair(base_ms: int) -> DagSpec:
    """``conflict_pair``: two tasks both write the same file; runtime must
    serialize them via ``access_conflicts`` even with no explicit
    dependency. Used as the F2 conflict_density baseline.
    """
    target = "file:conflict/shared.txt"
    tasks = [
        AgentTaskSpec(
            kind="tool",
            goal="conflict left",
            task_id="conflict-left",
            resource_access=ResourceAccess.write(target),
            handler=_sleep_handler("conflict-left", base_ms),
        ),
        AgentTaskSpec(
            kind="tool",
            goal="conflict right",
            task_id="conflict-right",
            resource_access=ResourceAccess.write(target),
            handler=_sleep_handler("conflict-right", base_ms),
        ),
    ]
    return DagSpec("conflict_pair", 2, base_ms, tasks)


def build_unknown_burst(n: int, base_ms: int) -> DagSpec:
    """``unknown_burst``: 1 unknown_workspace probe + n plain reads.

    Per the unknown probe should serialize against everything via
    ``access_conflicts`` (unknown=True → global exclusive); the plain reads
    can still parallelize among themselves once the probe is done.
    """
    if n < 1:
        raise ValueError("unknown_burst n must be >= 1")
    probe_id = "unknown-probe"
    tasks: list[AgentTaskSpec] = [
        AgentTaskSpec(
            kind="tool",
            goal="unknown probe",
            task_id=probe_id,
            resource_access=ResourceAccess.unknown_workspace(),
            handler=_sleep_handler("unknown-probe", base_ms),
        )
    ]
    for k in range(n):
        target = f"file:unknown/{k}.txt"
        tasks.append(
            AgentTaskSpec(
                kind="tool",
                goal=f"unknown read {k}",
                task_id=f"unknown-r-{k:03d}",
                # Wait for the probe to finish so ablations only have to
                # reason about the read/read parallelism among children.
                dependencies=[DependencyPredicate.task_completed(probe_id)],
                resource_access=ResourceAccess.read(target),
                handler=_sleep_handler(f"unknown-r-{k}", base_ms),
            )
        )
    return DagSpec("unknown_burst", n, base_ms, tasks)


SHAPE_BUILDERS: dict[str, Callable[..., DagSpec]] = {
    "chain": build_chain,
    "fanout": build_fanout,
    "diamond": build_diamond,
    "conflict_pair": lambda n, base_ms: build_conflict_pair(base_ms),
    "unknown_burst": build_unknown_burst,
}


def build_dag(shape: str, n: int, base_ms: int) -> DagSpec:
    """Dispatch to the shape-specific builder.

    ``conflict_pair`` ignores ``n`` (always 2 nodes); the dispatcher accepts
    it anyway so the CLI matrix can stay uniform.
    """
    builder = SHAPE_BUILDERS.get(shape)
    if builder is None:
        raise ValueError(f"unknown DAG shape: {shape!r}; valid: {sorted(SHAPE_BUILDERS)}")
    if shape == "conflict_pair":
        return builder(2, base_ms)
    return builder(n, base_ms)
