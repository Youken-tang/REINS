"""Order service that doesn't roll back on exception."""


class FakeTxn:
    def __init__(self): self.committed = False; self.rolled_back = False
    def commit(self): self.committed = True
    def rollback(self): self.rolled_back = True


def place_order(txn: FakeTxn, do_charge) -> None:
    # BUG: charges, then commits regardless of exception.
    try:
        do_charge()
    finally:
        txn.commit()
