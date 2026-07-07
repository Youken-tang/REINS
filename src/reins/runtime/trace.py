"""JSONL trace writer with typed event schema.

 (``) requires a stable trace schema for
all T1/T2/T3 experiments. ``TraceWriter.emit_typed`` validates events against
``TRACE_EVENT_SCHEMA`` and stamps every line with ``run_id`` so post-processing
scripts (`benchmark/scripts/parse_trace.py`) can join across runs without
re-parsing call sites. ``TraceWriter.emit`` remains for free-form events that
predate the schema (`task.discovered`, `task.cancelled`, `delivery.ready`,
`component.updated`, …); migration is incremental — adding a new typed event is
a one-line registry edit, but old events stay valid as long as they keep their
field names.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from threading import RLock
from typing import Any, Mapping


# ─────────────────────────── typed event schema ────────────────────────────
#
# Each entry maps an event name to (required_fields, optional_fields).
# Fields not in either list are still permitted (forward-compat) but will not
# be validated. An emit_typed call missing a required field raises ValueError
# in strict mode and logs a warning in non-strict mode.

TRACE_EVENT_SCHEMA: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    # ── runtime / scheduler events ──────────────────────────────────────────
    "task.submitted": (
        ("task_id", "kind", "state"),
        ("deps", "resource_access", "parent_id", "waiting_on"),
    ),
    "task.ready": (
        ("task_id",),
        ("wait_seconds",),
    ),
    "task.started": (
        ("task_id", "kind"),
        ("worker_id",),
    ),
    "task.completed": (
        ("task_id", "status"),
        ("summary", "duration_seconds", "summary_chars"),
    ),
    "task.suspended": (
        ("task_id",),
        ("awaiting", "suspend_token", "snapshot_size"),
    ),
    "task.resumed": (
        ("task_id",),
        ("triggered_by", "suspended_seconds", "trigger"),
    ),
    "task.late_after_cancel": (
        ("task_id",),
        ("status",),
    ),
    "delivery.batch": (
        ("count",),
        ("debounce_ms", "batch_seq", "task_ids"),
    ),
    "resource.conflict": (
        ("task_id",),
        ("attempted_access", "running_access", "conflict"),
    ),
    # ── controller / planner events ─────────────────────────────────────────
    "planner.requested": (
        ("planner_seq",),
        ("snapshot_seq", "kind", "fact_seq", "ledger_digest_size"),
    ),
    "planner.responded": (
        ("planner_seq",),
        (
            "snapshot_seq",
            "kind",
            "fact_seq",
            "tool_calls_count",
            "prompt_tokens",
            "completion_tokens",
            "finish_reason",
        ),
    ),
    "planner.lowered": (
        ("planner_seq",),
        ("submitted", "dropped", "deduped", "fixed", "snapshot_seq"),
    ),
    "planner.snapshot_stuck": (
        ("snapshot_seq",),
        ("attempts", "timeout_count", "stuck_threshold"),
    ),
    "controller.completion_gate": (
        ("accepted",),
        (
            "pending",
            "build_like",
            "file_writes",
            "saw_test_tool",
            "saw_passing_test",
            "saw_failed_test",
            "reason",
            "planner_seq",
        ),
    ),
    # ── model future suspend/resume (LLM RTT off worker pool) ──────────────
    "model.future.suspended": (
        ("token",),
        ("model", "est_rtt", "task_id"),
    ),
    "model.future.resumed": (
        ("token",),
        ("rtt_ms", "task_id"),
    ),
}


def _validate_payload(
    event: str, payload: Mapping[str, Any], strict: bool
) -> None:
    schema = TRACE_EVENT_SCHEMA.get(event)
    if schema is None:
        return
    required, _optional = schema
    missing = [f for f in required if f not in payload]
    if missing:
        msg = f"trace event {event!r} missing required fields: {missing}"
        if strict:
            raise ValueError(msg)


class TraceWriter:
    """Append-only JSONL writer.

    Public surface:
        emit(event, **payload)        — legacy free-form events.
        emit_typed(event, **payload)  — schema-checked events for set.
        run_id                        — uuid4-based id stamped on every line.

    Use ``HIGH_AGENT_TRACE_STRICT=1`` to make schema violations raise.
    Default (off) only validates that lookup-mode events have their required
    fields, but missing-field cases just degrade to a normal emit so a
    refactor never breaks a running benchmark mid-flight.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        run_id: str | None = None,
        strict: bool | None = None,
    ) -> None:
        self.path = Path(path) if path else None
        self.run_id = run_id or uuid.uuid4().hex
        self._strict = (
            bool(int(os.getenv("HIGH_AGENT_TRACE_STRICT", "0")))
            if strict is None
            else strict
        )
        self._lock = RLock()
        self._fh = None
        self._unflushed_events = 0
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    # ── public API ─────────────────────────────────────────────────────────

    def emit(self, event: str, **payload: Any) -> None:
        """Free-form event emit. Preferred for legacy events."""
        self._write({"event": event, **payload})

    def emit_typed(self, event: str, **payload: Any) -> None:
        """Schema-checked emit for events in TRACE_EVENT_SCHEMA.

        Unknown event names are accepted (treated as free-form) so this method
        is also safe to call as the default entry point.
        """
        _validate_payload(event, payload, self._strict)
        self._write({"event": event, **payload})

    def close(self) -> None:
        with self._lock:
            if self._fh is not None and not self._fh.closed:
                self._fh.flush()
                self._fh.close()

    # ── internal ───────────────────────────────────────────────────────────

    def _write(self, body: dict[str, Any]) -> None:
        if not self.path:
            return
        item = {"ts": time.time(), "run_id": self.run_id, **body}
        line = json.dumps(item, ensure_ascii=False, sort_keys=True)
        with self._lock:
            if self._fh is None or self._fh.closed:
                self._fh = self.path.open("a", encoding="utf-8")
            self._fh.write(line + "\n")
            # drop the per-event flush. The trace is
            # only consulted post-mortem (aggregate.py reads it after the
            # run is over) so per-line flush burns IO under the scheduler
            # lock with no observable benefit. close() still flushes, and
            # we coarse-flush every 100 events so tail-loss on crash is
            # bounded.
            self._unflushed_events += 1
            if self._unflushed_events >= 100:
                self._fh.flush()
                self._unflushed_events = 0
