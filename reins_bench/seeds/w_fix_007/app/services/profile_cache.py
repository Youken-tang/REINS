"""Profile cache that forgets to invalidate on update."""


class ProfileCache:
    def __init__(self) -> None:
        self._cache: dict[int, dict] = {}
        self._store: dict[int, dict] = {}

    def get(self, uid: int) -> dict | None:
        if uid in self._cache:
            return self._cache[uid]
        p = self._store.get(uid)
        if p is not None:
            self._cache[uid] = p
        return p

    def update(self, uid: int, **fields) -> None:
        # BUG: writes to store, never invalidates cache → stale reads.
        self._store.setdefault(uid, {}).update(fields)
