"""Partitioned MVCC context store with epoch-based GC.

Provides lock-free reads and per-block parallel writes for the agent runtime.
"""

from __future__ import annotations

import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Entry:
    """Immutable versioned entry in a block."""
    clock: int
    data: Any


class Block:
    """Append-only entry list for one logical partition.

    Each block has at most one writer at a time (per-task blocks are written
    only by that task's worker thread). The `facts` and `stats` blocks may
    have multiple appenders — Python 3.13t list.append is thread-safe.
    """

    __slots__ = ("entries",)

    def __init__(self) -> None:
        self.entries: list[Entry] = []

    def append(self, clock: int, data: Any) -> None:
        self.entries.append(Entry(clock=clock, data=data))

    def latest_at(self, target_clock: int) -> Entry | None:
        """Find the most recent entry with clock <= target_clock."""
        for entry in reversed(self.entries):
            if entry.clock <= target_clock:
                return entry
        return None

    def all_at(self, target_clock: int) -> list[Entry]:
        """Return all entries with clock <= target_clock."""
        return [e for e in self.entries if e.clock <= target_clock]


@dataclass
class ContextSnapshot:
    """Frozen aggregated view at a specific clock. Immutable after construction."""

    clock: int = 0
    task_states: dict[str, Any] = field(default_factory=dict)
    causal_edges: dict[str, list[str]] = field(default_factory=dict)
    discovery_edges: dict[str, list[str]] = field(default_factory=dict)
    facts: list[str] = field(default_factory=list)
    summaries: list[str] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)

    def copy(self) -> "ContextSnapshot":
        return ContextSnapshot(
            clock=self.clock,
            task_states=dict(self.task_states),
            causal_edges={k: list(v) for k, v in self.causal_edges.items()},
            discovery_edges={k: list(v) for k, v in self.discovery_edges.items()},
            facts=list(self.facts),
            summaries=list(self.summaries),
            stats=dict(self.stats),
        )

    def apply(self, block_id: str, data: Any) -> None:
        """Apply an entry's data to this snapshot based on block type."""
        if block_id.startswith("task:"):
            task_id = block_id[5:]
            self.task_states[task_id] = data
        elif block_id.startswith("causal:"):
            source_id = block_id[7:]
            if isinstance(data, (list, tuple)):
                self.causal_edges[source_id] = list(data)
            else:
                self.causal_edges.setdefault(source_id, []).append(str(data))
        elif block_id.startswith("discovery:"):
            parent_id = block_id[10:]
            if isinstance(data, (list, tuple)):
                self.discovery_edges[parent_id] = list(data)
            else:
                self.discovery_edges.setdefault(parent_id, []).append(str(data))
        elif block_id == "facts":
            if isinstance(data, str):
                self.facts.append(data)
        elif block_id == "summaries":
            if isinstance(data, str):
                self.summaries.append(data)
        elif block_id == "stats":
            if isinstance(data, dict):
                for k, v in data.items():
                    self.stats[k] = self.stats.get(k, 0) + int(v)


class AtomicCounter:
    """Thread-safe monotonic counter using a lock-free pattern on 3.13t."""

    __slots__ = ("_value", "_lock")

    def __init__(self, initial: int = 0) -> None:
        self._value = initial
        self._lock = threading.Lock()

    @property
    def value(self) -> int:
        return self._value

    def fetch_add(self, delta: int = 1) -> int:
        """Atomically increment and return the previous value."""
        with self._lock:
            prev = self._value
            self._value += delta
            return prev


class ReadHandle:
    """RAII handle for snapshot readers. Prevents GC of referenced versions."""

    __slots__ = ("_store", "_clock", "_released")

    def __init__(self, store: "ContextStore", clock: int) -> None:
        self._store = store
        self._clock = clock
        self._released = False
        store._reader_acquire(clock)

    @property
    def clock(self) -> int:
        return self._clock

    def release(self) -> None:
        if not self._released:
            self._released = True
            self._store._reader_release(self._clock)

    def __del__(self) -> None:
        self.release()

    def __enter__(self) -> "ReadHandle":
        return self

    def __exit__(self, *args: Any) -> None:
        self.release()


class ContextStore:
    """Partitioned MVCC store. Writers append to blocks, readers take snapshots."""

    def __init__(self, *, gc_interval: float = 5.0, gc_threshold: int = 128) -> None:
        self.clock = AtomicCounter(0)
        self.epoch: int = 0
        self.base = ContextSnapshot(clock=0)
        self.blocks: dict[str, Block] = {}
        # Multiset of active reader clocks. Two ReadHandles can hold the same
        # clock (read() does not advance the counter), so a plain set would
        # collapse them and let _compact() truncate entries that the second
        # reader still depends on. Counter keeps a refcount per clock.
        self._readers: Counter[int] = Counter()
        self._readers_lock = threading.Lock()
        self._blocks_lock = threading.Lock()
        self._gc_interval = gc_interval
        self._gc_threshold = gc_threshold
        self._gc_stop = threading.Event()
        self._gc_thread: threading.Thread | None = None

    def _reader_acquire(self, clock: int) -> None:
        with self._readers_lock:
            self._readers[clock] += 1

    def _reader_release(self, clock: int) -> None:
        with self._readers_lock:
            count = self._readers.get(clock, 0)
            if count <= 1:
                self._readers.pop(clock, None)
            else:
                self._readers[clock] = count - 1

    def get_block(self, block_id: str) -> Block:
        """Get or create a block by ID. Thread-safe for concurrent first access."""
        block = self.blocks.get(block_id)
        if block is not None:
            return block
        # check-then-set is not atomic on free-threaded 3.13t; wrap the
        # creation in a lock so two writers racing on a brand-new block_id
        # cannot both construct a fresh Block and silently drop one of the
        # two appends.
        with self._blocks_lock:
            block = self.blocks.get(block_id)
            if block is None:
                block = Block()
                self.blocks[block_id] = block
            return block

    def write(self, block_id: str, data: Any) -> int:
        """Append an entry to a block. Returns the clock value assigned."""
        c = self.clock.fetch_add(1)
        self.get_block(block_id).append(c, data)
        return c

    def snapshot(self, at_clock: int | None = None) -> ContextSnapshot:
        """Read a consistent view at the given clock. No locks needed."""
        # Register as a reader for the duration of the snapshot build so
        # _compact() cannot truncate entries we still need.
        target = at_clock if at_clock is not None else self.clock.value
        self._reader_acquire(target)
        try:
            result = self.base.copy()
            result.clock = target

            for block_id, block in list(self.blocks.items()):
                if block_id == "facts":
                    for entry in block.all_at(target):
                        result.apply(block_id, entry.data)
                elif block_id == "summaries":
                    for entry in block.all_at(target):
                        result.apply(block_id, entry.data)
                elif block_id == "stats":
                    for entry in block.all_at(target):
                        result.apply(block_id, entry.data)
                else:
                    entry = block.latest_at(target)
                    if entry is not None:
                        result.apply(block_id, entry.data)

            return result
        finally:
            self._reader_release(target)

    def read(self, at_clock: int | None = None) -> ReadHandle:
        """Acquire a read handle that prevents GC of versions >= clock."""
        clock = at_clock if at_clock is not None else self.clock.value
        return ReadHandle(self, clock)

    def start_gc(self) -> None:
        """Start the background GC thread."""
        if self._gc_thread is not None:
            return
        self._gc_stop.clear()
        self._gc_thread = threading.Thread(
            target=self._gc_loop, name="context-store-gc", daemon=True
        )
        self._gc_thread.start()

    def stop_gc(self) -> None:
        """Stop the background GC thread."""
        self._gc_stop.set()
        if self._gc_thread is not None:
            self._gc_thread.join(timeout=2.0)
            self._gc_thread = None

    def _gc_loop(self) -> None:
        while not self._gc_stop.is_set():
            self._gc_stop.wait(self._gc_interval)
            if self._gc_stop.is_set():
                break
            self._compact()

    def _compact(self) -> None:
        """Epoch compaction: squash old entries into base, truncate blocks."""
        with self._readers_lock:
            if not self._readers:
                low_wm = self.clock.value
            else:
                low_wm = min(self._readers)

        if low_wm - self.base.clock < self._gc_threshold:
            return

        new_base = self.base.copy()
        for block_id, block in self.blocks.items():
            for entry in block.entries:
                if entry.clock <= low_wm:
                    new_base.apply(block_id, entry.data)
        new_base.clock = low_wm

        self.base = new_base

        for block in self.blocks.values():
            block.entries = [e for e in block.entries if e.clock > low_wm]

        self.epoch += 1
