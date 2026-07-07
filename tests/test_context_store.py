from __future__ import annotations

import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from high_agent.runtime.context_store import (
    AtomicCounter,
    Block,
    ContextSnapshot,
    ContextStore,
    Entry,
    ReadHandle,
)


class AtomicCounterTests(unittest.TestCase):
    def test_fetch_add_returns_previous_value(self) -> None:
        c = AtomicCounter(10)
        self.assertEqual(c.fetch_add(1), 10)
        self.assertEqual(c.value, 11)

    def test_concurrent_fetch_add(self) -> None:
        c = AtomicCounter(0)
        n_threads = 30
        increments_per_thread = 100

        def worker():
            for _ in range(increments_per_thread):
                c.fetch_add(1)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(c.value, n_threads * increments_per_thread)


class BlockTests(unittest.TestCase):
    def test_append_and_latest_at(self) -> None:
        b = Block()
        b.append(1, "first")
        b.append(5, "second")
        b.append(10, "third")

        self.assertIsNone(b.latest_at(0))
        self.assertEqual(b.latest_at(1).data, "first")
        self.assertEqual(b.latest_at(3).data, "first")
        self.assertEqual(b.latest_at(5).data, "second")
        self.assertEqual(b.latest_at(99).data, "third")

    def test_all_at(self) -> None:
        b = Block()
        b.append(1, "a")
        b.append(2, "b")
        b.append(5, "c")

        entries = b.all_at(3)
        self.assertEqual(len(entries), 2)
        self.assertEqual([e.data for e in entries], ["a", "b"])

    def test_concurrent_append(self) -> None:
        b = Block()
        n_threads = 30
        appends_per_thread = 100

        def worker(thread_id):
            for i in range(appends_per_thread):
                b.append(thread_id * 1000 + i, f"t{thread_id}-{i}")

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(b.entries), n_threads * appends_per_thread)


class ContextSnapshotTests(unittest.TestCase):
    def test_apply_task_block(self) -> None:
        snap = ContextSnapshot()
        snap.apply("task:task-a", {"state": "completed", "summary": "done"})
        self.assertEqual(snap.task_states["task-a"]["state"], "completed")

    def test_apply_causal_block(self) -> None:
        snap = ContextSnapshot()
        snap.apply("causal:task-a", ["task-b", "task-c"])
        self.assertEqual(snap.causal_edges["task-a"], ["task-b", "task-c"])

    def test_apply_facts_block(self) -> None:
        snap = ContextSnapshot()
        snap.apply("facts", "fact-1")
        snap.apply("facts", "fact-2")
        self.assertEqual(snap.facts, ["fact-1", "fact-2"])

    def test_apply_stats_block(self) -> None:
        snap = ContextSnapshot()
        snap.apply("stats", {"completed": 1, "running": 2})
        snap.apply("stats", {"completed": 3})
        self.assertEqual(snap.stats["completed"], 4)
        self.assertEqual(snap.stats["running"], 2)

    def test_copy_is_independent(self) -> None:
        snap = ContextSnapshot(clock=5, facts=["a"])
        copy = snap.copy()
        copy.facts.append("b")
        self.assertEqual(snap.facts, ["a"])
        self.assertEqual(copy.facts, ["a", "b"])


class ContextStoreTests(unittest.TestCase):
    def test_write_and_snapshot(self) -> None:
        store = ContextStore()
        store.write("task:task-a", {"state": "running"})
        store.write("task:task-a", {"state": "completed"})
        store.write("task:task-b", {"state": "running"})

        snap = store.snapshot()
        self.assertEqual(snap.task_states["task-a"]["state"], "completed")
        self.assertEqual(snap.task_states["task-b"]["state"], "running")

    def test_snapshot_at_specific_clock(self) -> None:
        store = ContextStore()
        c1 = store.write("task:task-a", {"state": "running"})
        c2 = store.write("task:task-a", {"state": "completed"})

        snap_early = store.snapshot(at_clock=c1)
        self.assertEqual(snap_early.task_states["task-a"]["state"], "running")

        snap_late = store.snapshot(at_clock=c2)
        self.assertEqual(snap_late.task_states["task-a"]["state"], "completed")

    def test_facts_accumulate(self) -> None:
        store = ContextStore()
        store.write("facts", "fact-1")
        store.write("facts", "fact-2")
        store.write("facts", "fact-3")

        snap = store.snapshot()
        self.assertEqual(snap.facts, ["fact-1", "fact-2", "fact-3"])

    def test_read_handle_prevents_gc(self) -> None:
        store = ContextStore(gc_threshold=2)
        store.write("task:a", "v1")
        store.write("task:a", "v2")
        store.write("task:a", "v3")

        handle = store.read(at_clock=1)
        store._compact()
        self.assertIn(handle.clock, store._readers)
        handle.release()
        self.assertNotIn(1, store._readers)

    def test_gc_compacts_old_entries(self) -> None:
        store = ContextStore(gc_threshold=2)
        store.write("task:a", "v1")
        store.write("task:a", "v2")
        store.write("task:a", "v3")
        store.write("task:a", "v4")

        store._compact()

        self.assertEqual(store.epoch, 1)
        self.assertEqual(store.base.task_states.get("a"), "v4")
        block = store.blocks.get("task:a")
        self.assertEqual(len(block.entries), 0)

    def test_gc_does_not_remove_reader_referenced_entries(self) -> None:
        store = ContextStore(gc_threshold=2)
        c1 = store.write("task:a", "v1")
        store.write("task:a", "v2")
        store.write("task:a", "v3")

        handle = store.read(at_clock=c1)
        store._compact()
        self.assertEqual(store.epoch, 0)
        handle.release()

        store._compact()
        self.assertEqual(store.epoch, 1)

    def test_concurrent_write_different_blocks(self) -> None:
        store = ContextStore()
        n_threads = 30
        writes_per_thread = 50

        def worker(thread_id):
            block_id = f"task:task-{thread_id}"
            for i in range(writes_per_thread):
                store.write(block_id, {"state": f"step-{i}"})

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(store.clock.value, n_threads * writes_per_thread)
        snap = store.snapshot()
        for t in range(n_threads):
            self.assertEqual(
                snap.task_states[f"task-{t}"]["state"],
                f"step-{writes_per_thread - 1}",
            )

    def test_concurrent_read_while_writing(self) -> None:
        store = ContextStore()
        stop = threading.Event()
        errors: list[str] = []

        def writer():
            i = 0
            while not stop.is_set():
                store.write(f"task:w-{i % 10}", {"v": i})
                i += 1
                time.sleep(0.001)

        def reader():
            while not stop.is_set():
                clock = store.clock.value
                with store.read(at_clock=clock) as handle:
                    snap = store.snapshot(at_clock=handle.clock)
                    if snap.clock != handle.clock:
                        errors.append(f"clock mismatch: {snap.clock} != {handle.clock}")
                time.sleep(0.002)

        writers = [threading.Thread(target=writer) for _ in range(5)]
        readers = [threading.Thread(target=reader) for _ in range(5)]
        for t in writers + readers:
            t.start()

        time.sleep(0.5)
        stop.set()

        for t in writers + readers:
            t.join()

        self.assertEqual(errors, [])
        self.assertGreater(store.clock.value, 0)

    def test_gc_thread_lifecycle(self) -> None:
        store = ContextStore(gc_interval=0.1, gc_threshold=2)
        store.start_gc()
        for i in range(10):
            store.write("task:x", f"v{i}")
        time.sleep(0.3)
        store.stop_gc()
        self.assertGreater(store.epoch, 0)


if __name__ == "__main__":
    unittest.main()
