"""Abstract base adapter that both agent implementations must satisfy."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmark.adapters import TaskTrace
from benchmark.profiles import RuntimeProfile


@dataclass
class TaskInput:
    """Standardized task input for any agent."""
    task_id: str
    prompt: str
    workspace: str
    tools_available: list[str]
    expected_outputs: dict[str, Any] | None = None
    timeout: float = 120.0
    max_iterations: int = 50


class AgentAdapter(ABC):
    """Unified interface for running benchmark tasks on different agent frameworks."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Agent framework identifier."""
        ...

    @abstractmethod
    def setup(self, model: str, base_url: str, *, profile: RuntimeProfile | None = None, **kwargs: Any) -> None:
        """Initialize the agent with model configuration and optional profile."""
        ...

    @abstractmethod
    def run_task(self, task: TaskInput) -> TaskTrace:
        """Execute a single benchmark task and return the execution trace."""
        ...

    @abstractmethod
    def teardown(self) -> None:
        """Cleanup resources after benchmark run."""
        ...

    def supports_parallel(self) -> bool:
        """Whether this agent supports parallel tool execution."""
        return False

    def supports_delegation(self) -> bool:
        """Whether this agent supports task delegation to sub-agents."""
        return False
