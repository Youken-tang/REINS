"""SQLite-backed session persistence."""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from high_agent.config import get_config_paths
from high_agent.runtime.types import new_id


@dataclass(frozen=True)
class SessionRecord:
    session_id: str
    title: str
    created_at: float
    updated_at: float
    message_count: int


class SessionStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else get_config_paths().home / "sessions.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def create(self, title: str = "") -> str:
        session_id = new_id("session")
        now = time.time()
        with closing(self._connect()) as db:
            with db:
                db.execute(
                    "insert into sessions(session_id, title, created_at, updated_at) values (?, ?, ?, ?)",
                    (session_id, title or "untitled", now, now),
                )
        return session_id

    def append(self, session_id: str, message: dict[str, Any]) -> None:
        now = time.time()
        with closing(self._connect()) as db:
            with db:
                db.execute(
                    "insert into messages(session_id, role, content, payload, created_at) values (?, ?, ?, ?, ?)",
                    (
                        session_id,
                        str(message.get("role") or ""),
                        str(message.get("content") or ""),
                        json.dumps(message, ensure_ascii=False),
                        now,
                    ),
                )
                db.execute("update sessions set updated_at=? where session_id=?", (now, session_id))

    def load(self, session_id: str) -> list[dict[str, Any]]:
        with closing(self._connect()) as db:
            rows = db.execute(
                "select payload from messages where session_id=? order by id asc",
                (session_id,),
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def list(self, limit: int = 20) -> list[SessionRecord]:
        with closing(self._connect()) as db:
            rows = db.execute(
                """
                select s.session_id, s.title, s.created_at, s.updated_at, count(m.id)
                from sessions s
                left join messages m on m.session_id=s.session_id
                group by s.session_id
                order by s.updated_at desc
                limit ?
                """,
                (limit,),
            ).fetchall()
        return [SessionRecord(row[0], row[1], row[2], row[3], row[4]) for row in rows]

    def resume(self, session_id: str) -> list[dict[str, Any]]:
        messages = self.load(session_id)
        if not messages and not self._session_exists(session_id):
            raise KeyError(f"session not found or empty: {session_id}")
        return messages

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def _session_exists(self, session_id: str) -> bool:
        with closing(self._connect()) as db:
            row = db.execute("select 1 from sessions where session_id=?", (session_id,)).fetchone()
        return row is not None

    def _init_db(self) -> None:
        with closing(self._connect()) as db:
            with db:
                db.execute(
                    """
                    create table if not exists sessions(
                        session_id text primary key,
                        title text not null,
                        created_at real not null,
                        updated_at real not null
                    )
                    """
                )
                db.execute(
                    """
                    create table if not exists messages(
                        id integer primary key autoincrement,
                        session_id text not null,
                        role text not null,
                        content text not null,
                        payload text not null,
                        created_at real not null
                    )
                    """
                )
