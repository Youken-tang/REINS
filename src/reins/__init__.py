"""REINS standalone runtime package.

Carved out of high_agent so that other agents/runtimes (e.g. hermes, opencode)
can embed the resource-aware scheduler without pulling in the high_agent CLI,
TUI, memory, or plugin shells. See ```` and
```` for the extraction rationale.
"""

from __future__ import annotations

from reins._nogil import NoGILError, ensure_nogil, is_nogil
from reins.runtime.scheduler import CausalRuntime
from reins.runtime.types import (
    AgentTaskSpec,
    ComponentWrite,
    DeliveryBatch,
    DependencyPredicate,
    TaskResult,
)

__all__ = [
    "AgentTaskSpec",
    "CausalRuntime",
    "ComponentWrite",
    "DeliveryBatch",
    "DependencyPredicate",
    "NoGILError",
    "TaskResult",
    "ensure_nogil",
    "is_nogil",
]
