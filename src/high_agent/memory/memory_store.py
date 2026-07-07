"""SQLite-backed long-term memory facts."""

from __future__ import annotations

import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from high_agent.config import get_config_paths
from high_agent.runtime.types import new_id


@dataclass(frozen=True)
class MemoryFact:
    fact_id: str
    key: str
    value: str
    source: str
    created_at: float


class MemoryStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else get_config_paths().home / "memory.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def write_fact(self, key: str, value: str, *, source: str = "manual") -> str:
        fact_id = new_id("mem")
        now = time.time()
        with closing(self._connect()) as db:
            with db:
                db.execute(
                    "insert into facts(fact_id, key, value, source, created_at) values (?, ?, ?, ?, ?)",
                    (fact_id, key.strip(), value.strip(), source, now),
                )
        return fact_id

    def search(self, query: str = "", *, limit: int = 10) -> list[MemoryFact]:
        needle = f"%{query.strip()}%"
        limit = max(1, min(int(limit), 100))
        with closing(self._connect()) as db:
            if query.strip():
                rows = db.execute(
                    """
                    select fact_id, key, value, source, created_at
                    from facts
                    where key like ? or value like ?
                    order by created_at desc
                    limit ?
                    """,
                    (needle, needle, limit),
                ).fetchall()
            else:
                rows = db.execute(
                    """
                    select fact_id, key, value, source, created_at
                    from facts
                    order by created_at desc
                    limit ?
                    """,
                    (limit,),
                ).fetchall()
        return [MemoryFact(row[0], row[1], row[2], row[3], row[4]) for row in rows]

    def render_digest(self, query: str = "", *, limit: int = 8) -> str:
        facts = self.search(query, limit=limit)
        if not facts:
            return ""
        lines = ["Long-term memory digest:"]
        for fact in facts:
            lines.append(f"- {fact.key}: {fact.value}")
        return "\n".join(lines)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def _init_db(self) -> None:
        with closing(self._connect()) as db:
            with db:
                db.execute(
                    """
                    create table if not exists facts(
                        fact_id text primary key,
                        key text not null,
                        value text not null,
                        source text not null,
                        created_at real not null
                    )
                    """
                )
