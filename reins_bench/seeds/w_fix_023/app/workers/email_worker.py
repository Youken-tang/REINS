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
