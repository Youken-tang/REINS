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
