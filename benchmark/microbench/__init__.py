"""__init__ for benchmark.microbench."""

from __future__ import annotations

from benchmark.microbench.dag_shapes import (
    DagSpec,
    SHAPE_BUILDERS,
    build_chain,
    build_conflict_pair,
    build_dag,
    build_diamond,
    build_fanout,
    build_unknown_burst,
)

__all__ = [
    "DagSpec",
    "SHAPE_BUILDERS",
    "build_chain",
    "build_conflict_pair",
    "build_dag",
    "build_diamond",
    "build_fanout",
    "build_unknown_burst",
]
