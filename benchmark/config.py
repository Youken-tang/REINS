"""Benchmark configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


BENCHMARK_ROOT = Path(__file__).parent
TASKS_DIR = BENCHMARK_ROOT / "tasks"
WORKSPACES_DIR = BENCHMARK_ROOT / "workspaces"
RESULTS_DIR = BENCHMARK_ROOT / "results"


@dataclass
class BenchmarkConfig:
    max_task_timeout: float = 120.0
    max_iterations_per_task: int = 50
    workspace_root: str = str(WORKSPACES_DIR)
    parallel_workers: int = 4
    categories: list[str] = field(default_factory=lambda: ["parallel", "tool_use", "planning", "scheduling", "realistic"])
    repeat_runs: int = 3
    model: str = ""
    base_url: str = ""
    profile: str = ""
    profiles: list[str] = field(default_factory=list)
    output_format: str = "json"
    seed: int = 42
