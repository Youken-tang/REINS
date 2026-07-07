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
