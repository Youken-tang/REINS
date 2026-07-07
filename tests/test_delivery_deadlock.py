"""Regression test for ABBA deadlock between delivery_lock and _lock.

The deadlock occurred when:
- Controller thread: delivery_lock → wait for _lock (in wait_next_delivery)
- Callback thread: _lock → wait for delivery_lock (in _on_future_done)

Fix: wait_next_delivery no longer holds delivery_lock while acquiring _lock.
"""

import threading
import time

from high_agent.runtime.scheduler import CausalRuntime
from high_agent.runtime.types import AgentTaskSpec, TaskContext, TaskResult


def test_no_deadlock_fast_tasks():
    """Many fast tasks complete without deadlock between delivery and scheduling."""
    runtime = CausalRuntime(max_workers=8, workspace_root="/tmp")
    completed = []

    def fast_handler(ctx: TaskContext) -> TaskResult:
        return TaskResult.completed("done")

    tasks = [
        AgentTaskSpec(
            task_id=f"fast-{i}",
            kind="tool",
            goal=f"fast task {i}",
            handler=fast_handler,
        )
        for i in range(50)
    ]
    runtime.submit(tasks)

    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        batch = runtime.wait_next_delivery(timeout=1.0)
        if batch:
            completed.extend(batch.events)
        if runtime.pending_count() == 0:
            break

    runtime.shutdown()
    assert len(completed) >= 50, f"Only {len(completed)}/50 delivered (possible deadlock)"


def test_no_deadlock_concurrent_delivery_and_completion():
    """Controller reading deliveries while tasks complete rapidly doesn't deadlock."""
    runtime = CausalRuntime(max_workers=16, workspace_root="/tmp")
    delivered_count = 0
    stop = threading.Event()

    def fast_handler(ctx: TaskContext) -> TaskResult:
        return TaskResult.completed("ok")

    def consumer():
        nonlocal delivered_count
        while not stop.is_set():
            batch = runtime.wait_next_delivery(timeout=0.5)
            if batch:
                delivered_count += len(batch.events)

    consumer_thread = threading.Thread(target=consumer, daemon=True)
    consumer_thread.start()

    for wave in range(5):
        tasks = [
            AgentTaskSpec(
                task_id=f"w{wave}-{i}",
                kind="tool",
                goal=f"wave {wave} task {i}",
                handler=fast_handler,
            )
            for i in range(20)
        ]
        runtime.submit(tasks)
        time.sleep(0.05)

    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if runtime.pending_count() == 0:
            break
        time.sleep(0.1)

    stop.set()
    consumer_thread.join(timeout=3.0)
    runtime.shutdown()

    assert delivered_count >= 100, f"Only {delivered_count}/100 delivered (possible deadlock)"
