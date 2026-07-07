import threading
from app.services.transfer_service import Account, transfer


def test_no_deadlock_under_reverse_concurrent_transfers():
    a = Account("a", 1000)
    b = Account("b", 1000)
    done = threading.Event()
    def fwd():
        for _ in range(50): transfer(a, b, 1)
    def rev():
        for _ in range(50): transfer(b, a, 1)
    t1 = threading.Thread(target=fwd)
    t2 = threading.Thread(target=rev)
    t1.start(); t2.start()
    t1.join(timeout=5); t2.join(timeout=5)
    assert not t1.is_alive() and not t2.is_alive(), "deadlock"
    assert a.balance + b.balance == 2000
