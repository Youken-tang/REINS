import threading


class Account:
    def __init__(self, name, balance):
        self.name = name
        self.balance = balance
        self.lock = threading.Lock()


def transfer(src: Account, dst: Account, amount: int) -> None:
    a, b = (src, dst) if src.name < dst.name else (dst, src)
    with a.lock:
        with b.lock:
            src.balance -= amount
            dst.balance += amount
