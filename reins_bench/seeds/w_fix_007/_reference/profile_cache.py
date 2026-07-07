class ProfileCache:
    def __init__(self) -> None:
        self._cache: dict[int, dict] = {}
        self._store: dict[int, dict] = {}

    def get(self, uid: int) -> dict | None:
        if uid in self._cache:
            return dict(self._cache[uid])
        p = self._store.get(uid)
        if p is not None:
            self._cache[uid] = dict(p)
            return dict(p)
        return None

    def update(self, uid: int, **fields) -> None:
        self._store.setdefault(uid, {}).update(fields)
        self._cache.pop(uid, None)
