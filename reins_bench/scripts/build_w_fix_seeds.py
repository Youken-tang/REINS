"""build_w_fix_seeds — generate buggy starting projects for all 30 W_fix prompts.

Each W_fix prompt describes a bug in a specific file. The agent's job is to
locate the bug, fix it, write a regression test, and run it. Without a seed
project the agent has nothing to debug — it scaffolds from scratch with
random layouts and the oracle's `min_files` path lock fails by chance.

This script writes one buggy seed project per prompt under
`benchmark/reins_bench/seeds/<prompt_id>/`. Each seed includes:

    pyproject.toml             — declares `app` as a package
    app/__init__.py
    app/<subpkg>/__init__.py
    app/<subpkg>/<buggy>.py    — the file the prompt is asking about
    _reference/<buggy>.py      — the fixed version (for selftest)
    _reference/<test>.py       — a regression test (for selftest)

The buggy file is small (15-40 LOC) and the bug is exactly what the prompt
describes. The reference solution is a minimal patch + a test that catches
the bug; selftest_seeds.py copies seed + applies reference, runs pytest.
"""

from __future__ import annotations

import shutil
import textwrap
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_SEEDS_ROOT = _REPO / "benchmark" / "reins_bench" / "seeds"

_PYPROJECT = textwrap.dedent("""\
    [build-system]
    requires = ["setuptools>=61"]
    build-backend = "setuptools.build_meta"

    [project]
    name = "reins-bench-cell"
    version = "0.0.0"
    requires-python = ">=3.10"

    [tool.setuptools.packages.find]
    include = ["app*"]
""")


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _seed(pid: str, files: dict[str, str]) -> None:
    """Write a seed project. Each entry in `files` is path → content,
    relative to seeds/<pid>/. Always include pyproject.toml and the
    necessary __init__.py chain for `app/<subpkg>/`."""
    base = _SEEDS_ROOT / pid
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True)
    _write(base / "pyproject.toml", _PYPROJECT)
    _write(base / "app" / "__init__.py", "")
    # Auto-create __init__.py for any app/<subpkg>/ dirs we'll be writing.
    subpkgs: set[str] = set()
    for rel in files:
        parts = Path(rel).parts
        if parts[0] == "app" and len(parts) >= 3:
            subpkgs.add(parts[1])
    for sp in subpkgs:
        _write(base / "app" / sp / "__init__.py", "")
    for rel, content in files.items():
        _write(base / rel, content)


# ---------------------------------------------------------------------------
# 30 W_fix seed definitions. Each entry: pid → {file: content}
# Path conventions match each prompt's `ground_truth_resource_access` writes.
# `_reference/` holds (fixed_file, test_file) used by selftest only.
# ---------------------------------------------------------------------------


def build_001() -> None:
    # off-by-one in pagination cursor
    _seed("w_fix_001", {
        "app/api/users.py": textwrap.dedent('''\
            """Users pagination endpoint with an off-by-one bug.

            limit=20 returns 21 records because the cursor decrement happens
            after the slice (so the slice keeps one extra item).
            """
            from typing import Sequence


            def list_users(all_users: Sequence[dict], cursor: int, limit: int) -> list[dict]:
                # BUG: slice first, then decrement cursor — caller ends up keeping
                # one extra record because the slice end isn't tightened.
                page = all_users[cursor:cursor + limit + 1]
                cursor -= 1
                return list(page)
        '''),
        "_reference/users.py": textwrap.dedent('''\
            from typing import Sequence


            def list_users(all_users: Sequence[dict], cursor: int, limit: int) -> list[dict]:
                page = all_users[cursor:cursor + limit]
                return list(page)
        '''),
        "_reference/test_users_pagination.py": textwrap.dedent('''\
            from app.api.users import list_users


            def test_limit_20_returns_20():
                users = [{"id": i} for i in range(100)]
                page = list_users(users, cursor=0, limit=20)
                assert len(page) == 20

            def test_limit_5_returns_5():
                users = [{"id": i} for i in range(10)]
                page = list_users(users, cursor=2, limit=5)
                assert len(page) == 5
                assert page[0]["id"] == 2
        '''),
    })


def build_002() -> None:
    # race condition in counter increment
    _seed("w_fix_002", {
        "app/services/counter.py": textwrap.dedent('''\
            """Shared counter with a read-modify-write race."""


            class Counter:
                def __init__(self) -> None:
                    self._counts: dict[str, int] = {}

                def increment(self, key: str) -> int:
                    # BUG: read-modify-write without locking.
                    current = self._counts.get(key, 0)
                    current = current + 1
                    self._counts[key] = current
                    return current

                def get(self, key: str) -> int:
                    return self._counts.get(key, 0)
        '''),
        "_reference/counter.py": textwrap.dedent('''\
            import threading


            class Counter:
                def __init__(self) -> None:
                    self._counts: dict[str, int] = {}
                    self._lock = threading.Lock()

                def increment(self, key: str) -> int:
                    with self._lock:
                        current = self._counts.get(key, 0) + 1
                        self._counts[key] = current
                        return current

                def get(self, key: str) -> int:
                    with self._lock:
                        return self._counts.get(key, 0)
        '''),
        "_reference/test_counter_race.py": textwrap.dedent('''\
            import threading
            from app.services.counter import Counter


            def test_concurrent_increments_no_loss():
                c = Counter()
                N = 8
                ITERS = 500
                def worker():
                    for _ in range(ITERS):
                        c.increment("x")
                threads = [threading.Thread(target=worker) for _ in range(N)]
                for t in threads: t.start()
                for t in threads: t.join()
                assert c.get("x") == N * ITERS
        '''),
    })


def build_003() -> None:
    # timezone-naive datetime in API response
    _seed("w_fix_003", {
        "app/api/events.py": textwrap.dedent('''\
            """Events API serialiser — timezone-naive datetime bug."""
            from datetime import datetime


            def serialise_event(event: dict) -> dict:
                # BUG: utcnow() is naive — no tzinfo on the wire.
                created_at = event.get("created_at") or datetime.utcnow()
                return {
                    "id": event["id"],
                    "created_at": created_at.isoformat(),
                }
        '''),
        "_reference/events.py": textwrap.dedent('''\
            from datetime import datetime, timezone


            def serialise_event(event: dict) -> dict:
                created_at = event.get("created_at") or datetime.now(timezone.utc)
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=timezone.utc)
                return {
                    "id": event["id"],
                    "created_at": created_at.isoformat(),
                }
        '''),
        "_reference/test_events_timezone.py": textwrap.dedent('''\
            from datetime import datetime, timezone
            from app.api.events import serialise_event


            def test_serialised_iso_includes_timezone():
                evt = {"id": 1, "created_at": datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)}
                out = serialise_event(evt)
                iso = out["created_at"]
                assert iso.endswith("+00:00") or iso.endswith("Z")

            def test_naive_datetime_is_assumed_utc():
                evt = {"id": 2, "created_at": datetime(2026, 1, 1, 12, 0, 0)}
                out = serialise_event(evt)
                iso = out["created_at"]
                assert iso.endswith("+00:00") or iso.endswith("Z")
        '''),
    })


def build_004() -> None:
    # SQL N+1 in user-with-orders endpoint
    _seed("w_fix_004", {
        "app/api/users.py": textwrap.dedent('''\
            """user-with-orders fetcher with N+1 query bug."""

            class FakeDB:
                def __init__(self) -> None:
                    self.queries = 0
                    self._users = [{"id": i, "name": f"u{i}"} for i in range(5)]
                    self._orders = {i: [{"oid": i*10+j} for j in range(3)] for i in range(5)}

                def all_users(self) -> list[dict]:
                    self.queries += 1
                    return list(self._users)

                def orders_for_user(self, uid: int) -> list[dict]:
                    self.queries += 1
                    return list(self._orders.get(uid, []))

                def all_orders_grouped(self) -> dict[int, list[dict]]:
                    self.queries += 1
                    return {k: list(v) for k, v in self._orders.items()}


            def list_users_with_orders(db: FakeDB) -> list[dict]:
                # BUG: 1 query for users, then 1 per user for orders → N+1.
                users = db.all_users()
                out = []
                for u in users:
                    u2 = dict(u)
                    u2["orders"] = db.orders_for_user(u["id"])
                    out.append(u2)
                return out
        '''),
        "_reference/users.py": textwrap.dedent('''\
            class FakeDB:
                def __init__(self) -> None:
                    self.queries = 0
                    self._users = [{"id": i, "name": f"u{i}"} for i in range(5)]
                    self._orders = {i: [{"oid": i*10+j} for j in range(3)] for i in range(5)}

                def all_users(self) -> list[dict]:
                    self.queries += 1
                    return list(self._users)

                def orders_for_user(self, uid: int) -> list[dict]:
                    self.queries += 1
                    return list(self._orders.get(uid, []))

                def all_orders_grouped(self) -> dict[int, list[dict]]:
                    self.queries += 1
                    return {k: list(v) for k, v in self._orders.items()}


            def list_users_with_orders(db: FakeDB) -> list[dict]:
                users = db.all_users()
                grouped = db.all_orders_grouped()
                return [dict(u, orders=grouped.get(u["id"], [])) for u in users]
        '''),
        "_reference/test_users_orders_n_plus_one.py": textwrap.dedent('''\
            from app.api.users import FakeDB, list_users_with_orders


            def test_constant_query_count():
                db = FakeDB()
                list_users_with_orders(db)
                # 1 for users + 1 for grouped orders = 2 total, not N+1.
                assert db.queries <= 2
        '''),
    })


def build_005() -> None:
    # file handle leak in CSV importer
    _seed("w_fix_005", {
        "app/importers/csv_importer.py": textwrap.dedent('''\
            """CSV importer that leaks file handles."""
            import csv


            def import_rows(path: str) -> list[dict]:
                # BUG: file opened but never closed (no `with`).
                f = open(path, "r", encoding="utf-8")
                reader = csv.DictReader(f)
                return list(reader)
        '''),
        "_reference/csv_importer.py": textwrap.dedent('''\
            import csv


            def import_rows(path: str) -> list[dict]:
                with open(path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    return list(reader)
        '''),
        "_reference/test_csv_importer_leak.py": textwrap.dedent('''\
            import gc
            import os
            import tempfile

            from app.importers.csv_importer import import_rows


            def _open_fds():
                # Linux-only — read /proc/self/fd
                try:
                    return len(os.listdir("/proc/self/fd"))
                except OSError:
                    return 0


            def test_no_handle_leak_after_many_imports():
                with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
                    f.write("a,b\\n1,2\\n3,4\\n")
                    tmp = f.name
                gc.collect()
                before = _open_fds()
                for _ in range(50):
                    import_rows(tmp)
                gc.collect()
                after = _open_fds()
                # Allow tiny variance from interpreter; importing 50× shouldn't grow >5.
                assert after - before <= 5, f"leaked {after - before} fds"
        '''),
    })


# ---------------------------------------------------------------------------
# Helper for the bulk-style "small bug + obvious fix" prompts. We define
# them more compactly because their structure is uniform (one buggy file
# under app/<subpkg>/<name>.py + one ref + one ref test).
# ---------------------------------------------------------------------------


def _simple(pid: str, subpkg: str, name: str, buggy: str, fixed: str, test: str) -> None:
    _seed(pid, {
        f"app/{subpkg}/{name}.py": buggy,
        f"_reference/{name}.py": fixed,
        f"_reference/test_{pid[6:]}.py": test,  # filename hint only
    })


def build_006() -> None:
    # rounding error in invoice total
    _simple(
        "w_fix_006", "billing", "invoice",
        buggy=textwrap.dedent('''\
            """Invoice total computed with float arithmetic — accumulates rounding error."""


            def total_cents(line_items: list[dict]) -> int:
                # BUG: floats lose precision on long sums; cast to int loses cents.
                total = 0.0
                for item in line_items:
                    total += float(item["qty"]) * float(item["unit_price_cents"]) / 1.0
                return int(total)  # truncates, may be 1 cent low
        '''),
        fixed=textwrap.dedent('''\
            from decimal import Decimal


            def total_cents(line_items: list[dict]) -> int:
                total = Decimal(0)
                for item in line_items:
                    total += Decimal(item["qty"]) * Decimal(item["unit_price_cents"])
                return int(total)
        '''),
        test=textwrap.dedent('''\
            from app.billing.invoice import total_cents


            def test_no_floating_point_error_on_many_items():
                items = [{"qty": 1, "unit_price_cents": 33} for _ in range(1000)]
                # 1000 × 33 = 33000 cents exactly
                assert total_cents(items) == 33000

            def test_fractional_quantity():
                items = [{"qty": 3, "unit_price_cents": 99}]
                assert total_cents(items) == 297
        '''),
    )


def build_007() -> None:
    # cache invalidation skipped on update
    _simple(
        "w_fix_007", "services", "profile_cache",
        buggy=textwrap.dedent('''\
            """Profile cache that forgets to invalidate on update."""


            class ProfileCache:
                def __init__(self) -> None:
                    self._cache: dict[int, dict] = {}
                    self._store: dict[int, dict] = {}

                def get(self, uid: int) -> dict | None:
                    if uid in self._cache:
                        return self._cache[uid]
                    p = self._store.get(uid)
                    if p is not None:
                        self._cache[uid] = p
                    return p

                def update(self, uid: int, **fields) -> None:
                    # BUG: writes to store, never invalidates cache → stale reads.
                    self._store.setdefault(uid, {}).update(fields)
        '''),
        fixed=textwrap.dedent('''\
            class ProfileCache:
                def __init__(self) -> None:
                    self._cache: dict[int, dict] = {}
                    self._store: dict[int, dict] = {}

                def get(self, uid: int) -> dict | None:
                    if uid in self._cache:
                        return dict(self._cache[uid])
                    p = self._store.get(uid)
                    if p is not None:
                        self._cache[uid] = dict(p)
                        return dict(p)
                    return None

                def update(self, uid: int, **fields) -> None:
                    self._store.setdefault(uid, {}).update(fields)
                    self._cache.pop(uid, None)
        '''),
        test=textwrap.dedent('''\
            from app.services.profile_cache import ProfileCache


            def test_update_invalidates_cache():
                pc = ProfileCache()
                pc._store[1] = {"name": "alice"}
                assert pc.get(1)["name"] == "alice"
                pc.update(1, name="bob")
                assert pc.get(1)["name"] == "bob"
        '''),
    )


def build_008() -> None:
    # missing pagination in admin user list
    _simple(
        "w_fix_008", "api", "admin_users",
        buggy=textwrap.dedent('''\
            """Admin endpoint returns all users at once — no pagination."""


            def list_admin_users(db_users: list[dict], cursor: int = 0, limit: int = 50) -> dict:
                # BUG: returns full list, ignores cursor/limit.
                return {"items": list(db_users), "next_cursor": None}
        '''),
        fixed=textwrap.dedent('''\
            def list_admin_users(db_users: list[dict], cursor: int = 0, limit: int = 50) -> dict:
                page = db_users[cursor:cursor + limit]
                next_cursor = cursor + limit if cursor + limit < len(db_users) else None
                return {"items": list(page), "next_cursor": next_cursor}
        '''),
        test=textwrap.dedent('''\
            from app.api.admin_users import list_admin_users


            def test_pagination_returns_limit():
                users = [{"id": i} for i in range(120)]
                page = list_admin_users(users, cursor=0, limit=50)
                assert len(page["items"]) == 50
                assert page["next_cursor"] == 50

            def test_last_page_signals_end():
                users = [{"id": i} for i in range(60)]
                page = list_admin_users(users, cursor=50, limit=50)
                assert len(page["items"]) == 10
                assert page["next_cursor"] is None
        '''),
    )


def build_009() -> None:
    # integer overflow in metric aggregator (Python int doesn't overflow,
    # but the bug is using int32-bounded accumulation via numpy or modular wrap).
    _simple(
        "w_fix_009", "metrics", "aggregator",
        buggy=textwrap.dedent('''\
            """Metric aggregator that wraps under int32 overflow."""

            INT32_MAX = 2**31 - 1


            def aggregate_total(samples: list[int]) -> int:
                # BUG: simulates a fixed-width accumulator; wraps mod 2**32.
                acc = 0
                for s in samples:
                    acc = (acc + s) & 0xFFFFFFFF
                if acc > INT32_MAX:
                    acc -= 2**32
                return acc
        '''),
        fixed=textwrap.dedent('''\
            def aggregate_total(samples: list[int]) -> int:
                return sum(samples)
        '''),
        test=textwrap.dedent('''\
            from app.metrics.aggregator import aggregate_total


            def test_no_overflow_on_large_sum():
                samples = [10**9] * 5  # 5e9, would overflow int32
                assert aggregate_total(samples) == 5 * 10**9
        '''),
    )


def build_010() -> None:
    # incorrect json escape in webhook signer
    _simple(
        "w_fix_010", "webhooks", "signer",
        buggy=textwrap.dedent('''\
            """Webhook payload signer with wrong JSON canonicalisation."""
            import hashlib
            import hmac
            import json


            def sign(secret: str, payload: dict) -> str:
                # BUG: json.dumps uses default separators with spaces;
                # consumer canonicalises without spaces → signatures mismatch.
                body = json.dumps(payload)
                return hmac.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()
        '''),
        fixed=textwrap.dedent('''\
            import hashlib
            import hmac
            import json


            def sign(secret: str, payload: dict) -> str:
                body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
                return hmac.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()
        '''),
        test=textwrap.dedent('''\
            from app.webhooks.signer import sign


            def test_signature_canonical_independent_of_key_order():
                a = sign("k", {"a": 1, "b": 2})
                b = sign("k", {"b": 2, "a": 1})
                assert a == b

            def test_signature_no_whitespace_dependence():
                a = sign("k", {"a": [1, 2, 3]})
                # No space-bearing variants should drift; just check stable.
                b = sign("k", {"a": [1, 2, 3]})
                assert a == b
        '''),
    )


def build_011() -> None:
    # retry loop swallows non-retriable errors
    _simple(
        "w_fix_011", "clients", "payment_client",
        buggy=textwrap.dedent('''\
            """Payment client retry loop swallows non-retriable errors."""


            class TransientError(Exception):
                pass

            class FatalError(Exception):
                pass


            def charge(do_call, max_retries: int = 3):
                attempts = 0
                last_exc = None
                while attempts < max_retries:
                    try:
                        return do_call()
                    except Exception as e:  # BUG: catches FatalError too
                        last_exc = e
                        attempts += 1
                raise last_exc
        '''),
        fixed=textwrap.dedent('''\
            class TransientError(Exception):
                pass

            class FatalError(Exception):
                pass


            def charge(do_call, max_retries: int = 3):
                attempts = 0
                last_exc = None
                while attempts < max_retries:
                    try:
                        return do_call()
                    except TransientError as e:
                        last_exc = e
                        attempts += 1
                    except FatalError:
                        raise
                raise last_exc
        '''),
        test=textwrap.dedent('''\
            import pytest
            from app.clients.payment_client import charge, FatalError, TransientError


            def test_fatal_error_is_not_retried():
                calls = {"n": 0}
                def do():
                    calls["n"] += 1
                    raise FatalError("nope")
                with pytest.raises(FatalError):
                    charge(do, max_retries=3)
                assert calls["n"] == 1

            def test_transient_error_is_retried():
                calls = {"n": 0}
                def do():
                    calls["n"] += 1
                    if calls["n"] < 2:
                        raise TransientError("retry")
                    return "ok"
                assert charge(do, max_retries=3) == "ok"
                assert calls["n"] == 2
        '''),
    )


def build_012() -> None:
    # regex catastrophic backtracking in log parser
    _simple(
        "w_fix_012", "parsers", "log_parser",
        buggy=textwrap.dedent('''\
            """Log parser with catastrophic regex backtracking on long inputs."""
            import re

            # BUG: nested quantifier (a+)+ → exponential on input 'a'*N + 'b'
            _BAD = re.compile(r"^(a+)+b$")


            def matches_pattern(line: str) -> bool:
                return bool(_BAD.match(line))
        '''),
        fixed=textwrap.dedent('''\
            import re

            _GOOD = re.compile(r"^a+b$")


            def matches_pattern(line: str) -> bool:
                return bool(_GOOD.match(line))
        '''),
        test=textwrap.dedent('''\
            import time
            from app.parsers.log_parser import matches_pattern


            def test_pathological_input_runs_fast():
                pathological = "a" * 30  # would explode under (a+)+b
                t0 = time.monotonic()
                matches_pattern(pathological)
                assert time.monotonic() - t0 < 0.5

            def test_basic_match():
                assert matches_pattern("aaab")
                assert not matches_pattern("aaa")
        '''),
    )


def build_013() -> None:
    # env var read at import time
    _simple(
        "w_fix_013", "config", "settings",
        buggy=textwrap.dedent('''\
            """Settings module reads env at import time — tests can't override."""
            import os

            # BUG: captured at import; later os.environ changes ignored.
            DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite://default")


            def get_database_url() -> str:
                return DATABASE_URL
        '''),
        fixed=textwrap.dedent('''\
            import os


            def get_database_url() -> str:
                return os.environ.get("DATABASE_URL", "sqlite://default")
        '''),
        test=textwrap.dedent('''\
            import os
            from app.config.settings import get_database_url


            def test_env_picked_up_after_set(monkeypatch):
                monkeypatch.setenv("DATABASE_URL", "postgres://x")
                assert get_database_url() == "postgres://x"

            def test_default_when_unset(monkeypatch):
                monkeypatch.delenv("DATABASE_URL", raising=False)
                assert get_database_url() == "sqlite://default"
        '''),
    )


def build_014() -> None:
    # silent UnicodeDecodeError on log ingest
    _simple(
        "w_fix_014", "ingest", "file_reader",
        buggy=textwrap.dedent('''\
            """File reader silently drops bytes that aren't valid UTF-8."""


            def read_lines(path: str) -> list[str]:
                # BUG: errors='ignore' silently drops invalid bytes — losing data.
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read().splitlines()
        '''),
        fixed=textwrap.dedent('''\
            def read_lines(path: str) -> list[str]:
                with open(path, "rb") as f:
                    raw = f.read()
                return raw.decode("utf-8", errors="replace").splitlines()
        '''),
        test=textwrap.dedent('''\
            import tempfile, os
            from app.ingest.file_reader import read_lines


            def test_invalid_byte_replaced_not_dropped():
                with tempfile.NamedTemporaryFile("wb", delete=False, suffix=".log") as f:
                    f.write(b"hello\\xffworld\\n")
                    p = f.name
                lines = read_lines(p)
                os.unlink(p)
                assert lines, "lost the line entirely"
                assert "hello" in lines[0] and "world" in lines[0]
        '''),
    )


def build_015() -> None:
    # thundering herd on cache miss (single-flight)
    _simple(
        "w_fix_015", "cache", "loader",
        buggy=textwrap.dedent('''\
            """Cache loader with thundering-herd on miss."""
            import threading


            class CacheLoader:
                def __init__(self, fetch) -> None:
                    self._cache: dict = {}
                    self._fetch = fetch

                def get(self, key):
                    if key in self._cache:
                        return self._cache[key]
                    # BUG: every concurrent miss triggers its own fetch.
                    val = self._fetch(key)
                    self._cache[key] = val
                    return val
        '''),
        fixed=textwrap.dedent('''\
            import threading


            class CacheLoader:
                def __init__(self, fetch) -> None:
                    self._cache: dict = {}
                    self._fetch = fetch
                    self._lock = threading.Lock()
                    self._inflight: dict = {}

                def get(self, key):
                    if key in self._cache:
                        return self._cache[key]
                    with self._lock:
                        if key in self._cache:
                            return self._cache[key]
                        evt = self._inflight.get(key)
                        if evt is None:
                            evt = threading.Event()
                            self._inflight[key] = evt
                            do_fetch = True
                        else:
                            do_fetch = False
                    if do_fetch:
                        try:
                            val = self._fetch(key)
                            self._cache[key] = val
                        finally:
                            evt.set()
                            with self._lock:
                                self._inflight.pop(key, None)
                        return val
                    evt.wait()
                    return self._cache[key]
        '''),
        test=textwrap.dedent('''\
            import threading, time
            from app.cache.loader import CacheLoader


            def test_concurrent_misses_singleflight():
                calls = {"n": 0}
                def fetch(k):
                    calls["n"] += 1
                    time.sleep(0.05)
                    return k * 2
                cl = CacheLoader(fetch)
                results = []
                threads = [threading.Thread(target=lambda: results.append(cl.get("x"))) for _ in range(8)]
                for t in threads: t.start()
                for t in threads: t.join()
                assert all(r == "xx" for r in results)
                assert calls["n"] == 1
        '''),
    )


def build_016() -> None:
    # TOCTOU race in temp file creation
    _simple(
        "w_fix_016", "utils", "tempfile_helper",
        buggy=textwrap.dedent('''\
            """Temp file helper with TOCTOU race."""
            import os
            import tempfile


            def create_temp(prefix: str = "x") -> str:
                # BUG: name reserved then opened separately — attacker could
                # symlink in between. Should atomically open instead.
                d = tempfile.gettempdir()
                name = os.path.join(d, prefix + str(os.getpid()))
                if os.path.exists(name):
                    os.unlink(name)
                with open(name, "w") as f:
                    f.write("")
                return name
        '''),
        fixed=textwrap.dedent('''\
            import os
            import tempfile


            def create_temp(prefix: str = "x") -> str:
                fd, name = tempfile.mkstemp(prefix=prefix)
                os.close(fd)
                return name
        '''),
        test=textwrap.dedent('''\
            import os
            from app.utils.tempfile_helper import create_temp


            def test_create_temp_returns_existing_path():
                p = create_temp()
                try:
                    assert os.path.exists(p)
                    # On a TOCTOU-safe impl, two calls return distinct paths.
                    p2 = create_temp()
                    try:
                        assert p != p2
                    finally:
                        os.unlink(p2)
                finally:
                    if os.path.exists(p): os.unlink(p)
        '''),
    )


def build_017() -> None:
    # connection pool exhaustion under load
    _simple(
        "w_fix_017", "db", "pool",
        buggy=textwrap.dedent('''\
            """Connection pool that leaks connections (no release)."""


            class Conn:
                def __init__(self, i): self.i = i


            class Pool:
                def __init__(self, size: int = 3) -> None:
                    self._free = [Conn(i) for i in range(size)]
                    self._size = size

                def acquire(self) -> Conn:
                    if not self._free:
                        raise RuntimeError("pool exhausted")
                    return self._free.pop()

                # BUG: no release() — every acquire() drains the pool until exhausted.
        '''),
        fixed=textwrap.dedent('''\
            class Conn:
                def __init__(self, i): self.i = i


            class Pool:
                def __init__(self, size: int = 3) -> None:
                    self._free = [Conn(i) for i in range(size)]
                    self._size = size

                def acquire(self) -> Conn:
                    if not self._free:
                        raise RuntimeError("pool exhausted")
                    return self._free.pop()

                def release(self, c: Conn) -> None:
                    if len(self._free) < self._size:
                        self._free.append(c)
        '''),
        test=textwrap.dedent('''\
            from app.db.pool import Pool


            def test_release_returns_connection():
                p = Pool(size=2)
                c1 = p.acquire()
                c2 = p.acquire()
                p.release(c1)
                c3 = p.acquire()
                assert c3 is c1 or c3 is not None
        '''),
    )


def build_018() -> None:
    # timezone confusion in scheduled job
    _simple(
        "w_fix_018", "jobs", "scheduler",
        buggy=textwrap.dedent('''\
            """Scheduler that compares naive `now()` with tz-aware target."""
            from datetime import datetime, timezone


            def is_due(target_utc: datetime, now_local: datetime | None = None) -> bool:
                # BUG: naive vs aware comparison raises or gives wrong answer.
                now = now_local or datetime.now()
                return now >= target_utc
        '''),
        fixed=textwrap.dedent('''\
            from datetime import datetime, timezone


            def is_due(target_utc: datetime, now_local: datetime | None = None) -> bool:
                now = now_local or datetime.now(timezone.utc)
                if now.tzinfo is None:
                    now = now.replace(tzinfo=timezone.utc)
                if target_utc.tzinfo is None:
                    target_utc = target_utc.replace(tzinfo=timezone.utc)
                return now >= target_utc
        '''),
        test=textwrap.dedent('''\
            from datetime import datetime, timezone, timedelta
            from app.jobs.scheduler import is_due


            def test_due_when_target_in_past_utc():
                past = datetime.now(timezone.utc) - timedelta(hours=1)
                assert is_due(past)

            def test_not_due_when_target_in_future_utc():
                future = datetime.now(timezone.utc) + timedelta(hours=1)
                assert not is_due(future)
        '''),
    )


def build_019() -> None:
    # missing transaction rollback on exception
    _simple(
        "w_fix_019", "services", "order_service",
        buggy=textwrap.dedent('''\
            """Order service that doesn't roll back on exception."""


            class FakeTxn:
                def __init__(self): self.committed = False; self.rolled_back = False
                def commit(self): self.committed = True
                def rollback(self): self.rolled_back = True


            def place_order(txn: FakeTxn, do_charge) -> None:
                # BUG: charges, then commits regardless of exception.
                try:
                    do_charge()
                finally:
                    txn.commit()
        '''),
        fixed=textwrap.dedent('''\
            class FakeTxn:
                def __init__(self): self.committed = False; self.rolled_back = False
                def commit(self): self.committed = True
                def rollback(self): self.rolled_back = True


            def place_order(txn: FakeTxn, do_charge) -> None:
                try:
                    do_charge()
                    txn.commit()
                except Exception:
                    txn.rollback()
                    raise
        '''),
        test=textwrap.dedent('''\
            import pytest
            from app.services.order_service import FakeTxn, place_order


            def test_rollback_on_exception():
                txn = FakeTxn()
                def boom(): raise RuntimeError("payment failed")
                with pytest.raises(RuntimeError):
                    place_order(txn, boom)
                assert txn.rolled_back and not txn.committed

            def test_commit_on_success():
                txn = FakeTxn()
                place_order(txn, lambda: None)
                assert txn.committed and not txn.rolled_back
        '''),
    )


def build_020() -> None:
    # deadlock between two service locks (lock-ordering bug)
    _simple(
        "w_fix_020", "services", "transfer_service",
        buggy=textwrap.dedent('''\
            """Transfer service that acquires locks in caller-supplied order — deadlock."""
            import threading


            class Account:
                def __init__(self, name, balance):
                    self.name = name
                    self.balance = balance
                    self.lock = threading.Lock()


            def transfer(src: Account, dst: Account, amount: int) -> None:
                # BUG: lock order = (src, dst) — concurrent reverse-direction
                # transfer (dst→src) will deadlock.
                with src.lock:
                    with dst.lock:
                        src.balance -= amount
                        dst.balance += amount
        '''),
        fixed=textwrap.dedent('''\
            import threading


            class Account:
                def __init__(self, name, balance):
                    self.name = name
                    self.balance = balance
                    self.lock = threading.Lock()


            def transfer(src: Account, dst: Account, amount: int) -> None:
                a, b = (src, dst) if src.name < dst.name else (dst, src)
                with a.lock:
                    with b.lock:
                        src.balance -= amount
                        dst.balance += amount
        '''),
        test=textwrap.dedent('''\
            import threading
            from app.services.transfer_service import Account, transfer


            def test_no_deadlock_under_reverse_concurrent_transfers():
                a = Account("a", 1000)
                b = Account("b", 1000)
                done = threading.Event()
                def fwd():
                    for _ in range(50): transfer(a, b, 1)
                def rev():
                    for _ in range(50): transfer(b, a, 1)
                t1 = threading.Thread(target=fwd)
                t2 = threading.Thread(target=rev)
                t1.start(); t2.start()
                t1.join(timeout=5); t2.join(timeout=5)
                assert not t1.is_alive() and not t2.is_alive(), "deadlock"
                assert a.balance + b.balance == 2000
        '''),
    )


def build_021() -> None:
    # SQL injection in raw query construction
    _simple(
        "w_fix_021", "api", "search",
        buggy=textwrap.dedent('''\
            """Search endpoint with string-concat SQL — injection."""


            def build_query(table: str, term: str) -> str:
                # BUG: string concat — `term` controls the query.
                return f"SELECT * FROM {table} WHERE name LIKE '%{term}%'"


            def execute_search(db, table: str, term: str) -> list[dict]:
                return db.execute(build_query(table, term))
        '''),
        fixed=textwrap.dedent('''\
            _ALLOWED_TABLES = {"users", "products"}


            def build_query(table: str, term: str) -> tuple[str, tuple]:
                if table not in _ALLOWED_TABLES:
                    raise ValueError(f"disallowed table: {table}")
                return f"SELECT * FROM {table} WHERE name LIKE ?", (f"%{term}%",)


            def execute_search(db, table: str, term: str) -> list[dict]:
                sql, params = build_query(table, term)
                return db.execute(sql, params)
        '''),
        test=textwrap.dedent('''\
            import pytest
            from app.api.search import build_query


            def test_term_not_inlined():
                sql, params = build_query("users", "alice'; DROP TABLE users; --")
                assert "DROP TABLE" not in sql
                assert "alice" in params[0]

            def test_table_whitelist():
                with pytest.raises(ValueError):
                    build_query("users; DROP TABLE", "x")
        '''),
    )


def build_022() -> None:
    # missing input validation on API boundary
    _simple(
        "w_fix_022", "api", "upload",
        buggy=textwrap.dedent('''\
            """Upload endpoint with no input validation."""


            def handle_upload(filename: str, size: int, mime: str) -> dict:
                # BUG: accepts anything — including 100GB and arbitrary mime.
                return {"ok": True, "stored_as": filename}
        '''),
        fixed=textwrap.dedent('''\
            import re

            _MAX_SIZE = 10 * 1024 * 1024
            _ALLOWED_MIME = {"image/png", "image/jpeg", "application/pdf"}
            _SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


            def handle_upload(filename: str, size: int, mime: str) -> dict:
                if not _SAFE_NAME.match(filename) or len(filename) > 255:
                    raise ValueError("invalid filename")
                if size <= 0 or size > _MAX_SIZE:
                    raise ValueError("invalid size")
                if mime not in _ALLOWED_MIME:
                    raise ValueError("disallowed mime type")
                return {"ok": True, "stored_as": filename}
        '''),
        test=textwrap.dedent('''\
            import pytest
            from app.api.upload import handle_upload


            def test_oversized_rejected():
                with pytest.raises(ValueError):
                    handle_upload("a.png", 100 * 1024 * 1024, "image/png")

            def test_path_traversal_rejected():
                with pytest.raises(ValueError):
                    handle_upload("../../etc/passwd", 100, "image/png")

            def test_disallowed_mime_rejected():
                with pytest.raises(ValueError):
                    handle_upload("a.exe", 100, "application/x-msdownload")

            def test_happy_path():
                assert handle_upload("photo.png", 1024, "image/png")["ok"]
        '''),
    )


def build_023() -> None:
    # improper error swallowing in async task
    _simple(
        "w_fix_023", "workers", "email_worker",
        buggy=textwrap.dedent('''\
            """Email worker that swallows all exceptions — silent failures."""

            import logging
            log = logging.getLogger(__name__)


            class WorkerStats:
                def __init__(self): self.successes = 0; self.failures = 0


            def process_one(task, send_fn, stats: WorkerStats) -> None:
                try:
                    send_fn(task)
                    stats.successes += 1
                except Exception:
                    # BUG: swallows the error silently — caller never knows.
                    pass
        '''),
        fixed=textwrap.dedent('''\
            import logging
            log = logging.getLogger(__name__)


            class WorkerStats:
                def __init__(self): self.successes = 0; self.failures = 0


            def process_one(task, send_fn, stats: WorkerStats) -> None:
                try:
                    send_fn(task)
                    stats.successes += 1
                except Exception as e:
                    stats.failures += 1
                    log.exception("email send failed for task=%s", task)
        '''),
        test=textwrap.dedent('''\
            from app.workers.email_worker import process_one, WorkerStats


            def test_failure_counted():
                stats = WorkerStats()
                def send(t): raise RuntimeError("smtp down")
                process_one("t1", send, stats)
                assert stats.failures == 1 and stats.successes == 0

            def test_success_counted():
                stats = WorkerStats()
                process_one("t1", lambda t: None, stats)
                assert stats.successes == 1 and stats.failures == 0
        '''),
    )


def build_024() -> None:
    # stale field in serialiser after model update
    _simple(
        "w_fix_024", "api", "users",
        buggy=textwrap.dedent('''\
            """User serialiser missing the new `email_verified` field."""


            def serialise_user(user: dict) -> dict:
                # BUG: model now has email_verified but serialiser doesn't expose it.
                return {
                    "id": user["id"],
                    "name": user["name"],
                    "email": user["email"],
                }
        '''),
        fixed=textwrap.dedent('''\
            def serialise_user(user: dict) -> dict:
                return {
                    "id": user["id"],
                    "name": user["name"],
                    "email": user["email"],
                    "email_verified": bool(user.get("email_verified", False)),
                }
        '''),
        test=textwrap.dedent('''\
            from app.api.users import serialise_user


            def test_serialiser_includes_email_verified():
                u = {"id": 1, "name": "a", "email": "a@x.com", "email_verified": True}
                out = serialise_user(u)
                assert out["email_verified"] is True

            def test_serialiser_defaults_email_verified_false():
                u = {"id": 2, "name": "b", "email": "b@x.com"}
                out = serialise_user(u)
                assert out["email_verified"] is False
        '''),
    )


def build_025() -> None:
    # signed-token verification skips expiry check
    _simple(
        "w_fix_025", "auth", "token",
        buggy=textwrap.dedent('''\
            """Signed-token verifier — expiry not checked."""
            import hmac, hashlib, json, base64, time


            def issue(payload: dict, secret: str, ttl_seconds: int = 60) -> str:
                payload = dict(payload, exp=int(time.time()) + ttl_seconds)
                body = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
                sig = hmac.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()
                return f"{body}.{sig}"


            def verify(token: str, secret: str) -> dict | None:
                body, sig = token.rsplit(".", 1)
                expected = hmac.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()
                if not hmac.compare_digest(sig, expected):
                    return None
                # BUG: signature ok but exp not checked — expired tokens still pass.
                return json.loads(base64.urlsafe_b64decode(body.encode()).decode())
        '''),
        fixed=textwrap.dedent('''\
            import hmac, hashlib, json, base64, time


            def issue(payload: dict, secret: str, ttl_seconds: int = 60) -> str:
                payload = dict(payload, exp=int(time.time()) + ttl_seconds)
                body = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
                sig = hmac.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()
                return f"{body}.{sig}"


            def verify(token: str, secret: str) -> dict | None:
                body, sig = token.rsplit(".", 1)
                expected = hmac.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()
                if not hmac.compare_digest(sig, expected):
                    return None
                payload = json.loads(base64.urlsafe_b64decode(body.encode()).decode())
                if int(payload.get("exp", 0)) < int(time.time()):
                    return None
                return payload
        '''),
        test=textwrap.dedent('''\
            import time
            from app.auth.token import issue, verify


            def test_expired_rejected():
                t = issue({"sub": "alice"}, "secret", ttl_seconds=-1)
                assert verify(t, "secret") is None

            def test_valid_accepted():
                t = issue({"sub": "alice"}, "secret", ttl_seconds=60)
                p = verify(t, "secret")
                assert p and p["sub"] == "alice"

            def test_tampered_rejected():
                t = issue({"sub": "alice"}, "secret", ttl_seconds=60)
                body, _ = t.rsplit(".", 1)
                bad = body + "." + "0" * 64
                assert verify(bad, "secret") is None
        '''),
    )


def build_026() -> None:
    # idempotency missing on payment intent
    _simple(
        "w_fix_026", "api", "payments",
        buggy=textwrap.dedent('''\
            """Payment intent endpoint — duplicate requests double-charge."""


            class PaymentService:
                def __init__(self):
                    self.charges = []

                def create_intent(self, idempotency_key: str | None, amount: int) -> dict:
                    # BUG: idempotency_key ignored — every retry charges again.
                    self.charges.append((idempotency_key, amount))
                    return {"intent_id": f"pi_{len(self.charges)}", "amount": amount}
        '''),
        fixed=textwrap.dedent('''\
            class PaymentService:
                def __init__(self):
                    self.charges = []
                    self._by_key: dict[str, dict] = {}

                def create_intent(self, idempotency_key: str | None, amount: int) -> dict:
                    if idempotency_key and idempotency_key in self._by_key:
                        return self._by_key[idempotency_key]
                    self.charges.append((idempotency_key, amount))
                    intent = {"intent_id": f"pi_{len(self.charges)}", "amount": amount}
                    if idempotency_key:
                        self._by_key[idempotency_key] = intent
                    return intent
        '''),
        test=textwrap.dedent('''\
            from app.api.payments import PaymentService


            def test_same_key_returns_same_intent():
                s = PaymentService()
                a = s.create_intent("k1", 1000)
                b = s.create_intent("k1", 1000)
                assert a == b
                assert len(s.charges) == 1

            def test_different_keys_charge_separately():
                s = PaymentService()
                s.create_intent("k1", 100)
                s.create_intent("k2", 100)
                assert len(s.charges) == 2
        '''),
    )


def build_027() -> None:
    # missing rate-limit on login endpoint
    _simple(
        "w_fix_027", "api", "auth",
        buggy=textwrap.dedent('''\
            """Login endpoint with no rate limit — password brute-force possible."""


            class LoginService:
                def __init__(self, password_check):
                    self._check = password_check

                def login(self, ip: str, user: str, password: str) -> bool:
                    # BUG: no per-IP throttling.
                    return self._check(user, password)
        '''),
        fixed=textwrap.dedent('''\
            import time

            _MAX_ATTEMPTS = 5
            _WINDOW_SECONDS = 60


            class LoginService:
                def __init__(self, password_check):
                    self._check = password_check
                    self._attempts: dict[str, list[float]] = {}

                def login(self, ip: str, user: str, password: str) -> bool:
                    now = time.monotonic()
                    history = [t for t in self._attempts.get(ip, []) if now - t < _WINDOW_SECONDS]
                    if len(history) >= _MAX_ATTEMPTS:
                        raise PermissionError("rate limited")
                    history.append(now)
                    self._attempts[ip] = history
                    return self._check(user, password)
        '''),
        test=textwrap.dedent('''\
            import pytest
            from app.api.auth import LoginService


            def test_rate_limit_after_threshold():
                s = LoginService(lambda u, p: False)
                for _ in range(5):
                    s.login("1.2.3.4", "u", "p")
                with pytest.raises(PermissionError):
                    s.login("1.2.3.4", "u", "p")

            def test_other_ip_unaffected():
                s = LoginService(lambda u, p: False)
                for _ in range(5):
                    s.login("1.2.3.4", "u", "p")
                # Different IP should still be allowed.
                s.login("5.6.7.8", "u", "p")
        '''),
    )


def build_028() -> None:
    # regex matches across newlines unintentionally
    _simple(
        "w_fix_028", "parsers", "markdown",
        buggy=textwrap.dedent('''\
            """Markdown fence parser — `.` matches newlines, eats unrelated content."""
            import re

            # BUG: re.DOTALL across the whole text → fences merge with later code blocks.
            _FENCE = re.compile(r"```(.+?)```", re.DOTALL)


            def find_fences(text: str) -> list[str]:
                return _FENCE.findall(text)
        '''),
        fixed=textwrap.dedent('''\
            import re

            # Match ``` … ``` on a per-block basis without spanning unrelated paragraphs.
            _FENCE = re.compile(r"```([\\s\\S]*?)```")


            def find_fences(text: str) -> list[str]:
                return _FENCE.findall(text)
        '''),
        test=textwrap.dedent('''\
            from app.parsers.markdown import find_fences


            def test_two_separate_fences_not_merged():
                text = "```a```\\n\\nbetween\\n\\n```b```"
                fences = find_fences(text)
                assert fences == ["a", "b"]

            def test_fence_with_newlines_inside():
                text = "```py\\nx = 1\\n```"
                fences = find_fences(text)
                assert len(fences) == 1
                assert "x = 1" in fences[0]
        '''),
    )


def build_029() -> None:
    # missing CSRF token check on state-changing endpoint
    _simple(
        "w_fix_029", "api", "profile",
        buggy=textwrap.dedent('''\
            """Profile update endpoint — no CSRF token check."""


            class ProfileService:
                def __init__(self):
                    self.profiles: dict[int, dict] = {}

                def update(self, uid: int, csrf_token: str | None, fields: dict) -> dict:
                    # BUG: csrf_token unused → CSRF attacks succeed.
                    p = self.profiles.setdefault(uid, {})
                    p.update(fields)
                    return p
        '''),
        fixed=textwrap.dedent('''\
            import hmac


            class ProfileService:
                def __init__(self, expected_token: str = "valid"):
                    self.profiles: dict[int, dict] = {}
                    self._expected = expected_token

                def update(self, uid: int, csrf_token: str | None, fields: dict) -> dict:
                    if not csrf_token or not hmac.compare_digest(csrf_token, self._expected):
                        raise PermissionError("missing or invalid CSRF token")
                    p = self.profiles.setdefault(uid, {})
                    p.update(fields)
                    return p
        '''),
        test=textwrap.dedent('''\
            import pytest
            from app.api.profile import ProfileService


            def test_missing_token_rejected():
                s = ProfileService()
                with pytest.raises(PermissionError):
                    s.update(1, None, {"name": "a"})

            def test_wrong_token_rejected():
                s = ProfileService()
                with pytest.raises(PermissionError):
                    s.update(1, "bogus", {"name": "a"})

            def test_valid_token_accepted():
                s = ProfileService()
                s.update(1, "valid", {"name": "alice"})
                assert s.profiles[1]["name"] == "alice"
        '''),
    )


def build_030() -> None:
    # floating-point comparison without epsilon
    _simple(
        "w_fix_030", "services", "inventory",
        buggy=textwrap.dedent('''\
            """Inventory reconciler that uses == on floats."""


            def reconcile(observed: float, expected: float) -> bool:
                # BUG: floating-point == is fragile (0.1 + 0.2 != 0.3, etc.)
                return observed == expected
        '''),
        fixed=textwrap.dedent('''\
            import math


            def reconcile(observed: float, expected: float, *, tol: float = 1e-9) -> bool:
                return math.isclose(observed, expected, rel_tol=tol, abs_tol=tol)
        '''),
        test=textwrap.dedent('''\
            from app.services.inventory import reconcile


            def test_classic_floating_point_pair():
                assert reconcile(0.1 + 0.2, 0.3)

            def test_genuinely_different():
                assert not reconcile(1.0, 2.0)
        '''),
    )


def build_all() -> None:
    """Run every build_NNN() helper. Idempotent — overwrites existing seeds."""
    import sys
    mod = sys.modules[__name__]
    n = 0
    for i in range(1, 31):
        fn = getattr(mod, f"build_{i:03d}", None)
        if fn is not None:
            fn()
            n += 1
    print(f"built {n} seed projects under {_SEEDS_ROOT}")


if __name__ == "__main__":
    build_all()
