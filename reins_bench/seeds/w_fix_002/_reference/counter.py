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
