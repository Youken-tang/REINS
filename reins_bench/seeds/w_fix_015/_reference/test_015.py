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
