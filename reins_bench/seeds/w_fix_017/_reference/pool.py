class Conn:
    def __init__(self, i): self.i = i


class Pool:
    def __init__(self, size: int = 3) -> None:
        self._free = [Conn(i) for i in range(size)]
        self._size = size

    def acquire(self) -> Conn:
        if not self._free:
            raise RuntimeError("pool exhausted")
        return self._free.pop()

    def release(self, c: Conn) -> None:
        if len(self._free) < self._size:
            self._free.append(c)
