"""Baseline and ablation profile registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RuntimeProfile:
    """Named configuration for a baseline or ablation variant."""

    name: str
    description: str
    runtime_overrides: dict[str, Any] = field(default_factory=dict)
    controller_overrides: dict[str, Any] = field(default_factory=dict)
    adapter_hooks: dict[str, Any] = field(default_factory=dict)


BASELINES: dict[str, RuntimeProfile] = {
    "sequential": RuntimeProfile(
        name="sequential",
        description="Serial execution — single worker, single planner (ReAct-style)",
        runtime_overrides={"max_workers": 1},
        controller_overrides={"max_planner_requests": 1},
    ),
    "naive_parallel": RuntimeProfile(
        name="naive_parallel",
        description="All tasks in parallel, no conflict detection",
        runtime_overrides={"max_workers": 8},
        adapter_hooks={"disable_conflict_detection": True},
    ),
    "batch_parallel": RuntimeProfile(
        name="batch_parallel",
        description="Batch-submit per planner response, wait for completion before next batch",
        runtime_overrides={"max_workers": 8},
        controller_overrides={"max_planner_requests": 1},
        adapter_hooks={"disable_streaming_dispatch": True},
    ),
    "high_agent": RuntimeProfile(
        name="high_agent",
        description="Full REINS system — all features enabled",
    ),
}

ABLATIONS: dict[str, RuntimeProfile] = {
    "no_conflict_detection": RuntimeProfile(
        name="no_conflict_detection",
        description="Ablation: disable resource conflict detection",
        adapter_hooks={"disable_conflict_detection": True},
    ),
    "no_interleaved_planning": RuntimeProfile(
        name="no_interleaved_planning",
        description="Ablation: single planner request at a time",
        controller_overrides={"max_planner_requests": 1},
    ),
    "no_bounded_context": RuntimeProfile(
        name="no_bounded_context",
        description="Ablation: remove planner context size limits (full transcript)",
        adapter_hooks={"unbounded_context": True},
    ),
    "no_streaming_dispatch": RuntimeProfile(
        name="no_streaming_dispatch",
        description="Ablation: disable streaming early dispatch",
        adapter_hooks={"disable_streaming_dispatch": True},
    ),
    "no_adaptive_debounce": RuntimeProfile(
        name="no_adaptive_debounce",
        description="Ablation: fixed delivery debounce (no adaptive scaling)",
        runtime_overrides={"delivery_debounce": 0.05},
        adapter_hooks={"fixed_debounce": True},
    ),
}

ALL_PROFILES: dict[str, RuntimeProfile] = {**BASELINES, **ABLATIONS}


def get_profile(name: str) -> RuntimeProfile:
    """Look up a profile by name. Raises KeyError if not found."""
    if name in ALL_PROFILES:
        return ALL_PROFILES[name]
    raise KeyError(
        f"Unknown profile '{name}'. Available: {sorted(ALL_PROFILES.keys())}"
    )


def list_profiles() -> list[str]:
    """Return all available profile names."""
    return sorted(ALL_PROFILES.keys())
