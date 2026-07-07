"""Cache loader with thundering-herd on miss."""
import threading


class CacheLoader:
    def __init__(self, fetch) -> None:
        self._cache: dict = {}
        self._fetch = fetch

    def get(self, key):
        if key in self._cache:
            return self._cache[key]
        # BUG: every concurrent miss triggers its own fetch.
        val = self._fetch(key)
        self._cache[key] = val
        return val
