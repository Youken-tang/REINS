class FakeTxn:
    def __init__(self): self.committed = False; self.rolled_back = False
    def commit(self): self.committed = True
    def rollback(self): self.rolled_back = True


def place_order(txn: FakeTxn, do_charge) -> None:
    try:
        do_charge()
        txn.commit()
    except Exception:
        txn.rollback()
        raise
