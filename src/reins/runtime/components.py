"""Versioned component store."""

from __future__ import annotations

import time
from dataclasses import dataclass
from threading import RLock
from typing import Any

from reins.runtime.types import ComponentWrite


@dataclass
class Component:
    component_id: str
    version: int
    value: Any
    exists: bool
    last_writer_task_id: str | None
    updated_at: float


class ComponentStore:
    def __init__(self) -> None:
        self._lock = RLock()
        self._components: dict[str, Component] = {}

    def exists(self, component_id: str) -> bool:
        with self._lock:
            comp = self._components.get(component_id)
            return bool(comp and comp.exists)

    def version(self, component_id: str) -> int:
        with self._lock:
            comp = self._components.get(component_id)
            return comp.version if comp else 0

    def get(self, component_id: str) -> Component | None:
        with self._lock:
            return self._components.get(component_id)

    def snapshot(self) -> dict[str, Component]:
        with self._lock:
            return dict(self._components)

    def apply(self, task_id: str, write: ComponentWrite) -> Component:
        with self._lock:
            current = self._components.get(write.component_id)
            next_version = (current.version if current else 0) + 1
            if write.mode == "append" and current and current.exists:
                if isinstance(current.value, list):
                    value = [*current.value, write.value]
                else:
                    value = [current.value, write.value]
            else:
                value = write.value
            comp = Component(
                component_id=write.component_id,
                version=next_version,
                value=value,
                exists=True,
                last_writer_task_id=task_id,
                updated_at=time.monotonic(),
            )
            self._components[write.component_id] = comp
            return comp
