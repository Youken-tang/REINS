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
