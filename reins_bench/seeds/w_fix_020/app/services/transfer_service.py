"""Transfer service that acquires locks in caller-supplied order — deadlock."""
import threading


class Account:
    def __init__(self, name, balance):
        self.name = name
        self.balance = balance
        self.lock = threading.Lock()


def transfer(src: Account, dst: Account, amount: int) -> None:
    # BUG: lock order = (src, dst) — concurrent reverse-direction
    # transfer (dst→src) will deadlock.
    with src.lock:
        with dst.lock:
            src.balance -= amount
            dst.balance += amount
