"""Durable storage for large tool results."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from high_agent.config import get_config_paths
from high_agent.runtime.types import new_id


@dataclass(frozen=True)
class StoredToolResult:
    result_id: str
    path: Path
    summary: str


class ToolResultStore:
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else get_config_paths().home / "tool-results"
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, value: str, summary: str = "") -> str:
        result_id = new_id("result")
        path = self.root / f"{result_id}.json"
        payload = {"id": result_id, "summary": summary or value[:500], "value": value, "created_at": time.time()}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return result_id

    def get(self, result_id: str) -> str:
        path = self.root / f"{result_id}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        return str(payload.get("value") or "")

    def metadata(self, result_id: str) -> StoredToolResult:
        path = self.root / f"{result_id}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        return StoredToolResult(result_id=result_id, path=path, summary=str(payload.get("summary") or ""))
