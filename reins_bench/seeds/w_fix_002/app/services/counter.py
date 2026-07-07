"""Shared counter with a read-modify-write race."""


class Counter:
    def __init__(self) -> None:
        self._counts: dict[str, int] = {}

    def increment(self, key: str) -> int:
        # BUG: read-modify-write without locking.
        current = self._counts.get(key, 0)
        current = current + 1
        self._counts[key] = current
        return current

    def get(self, key: str) -> int:
        return self._counts.get(key, 0)
