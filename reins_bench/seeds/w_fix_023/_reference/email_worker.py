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
