import pytest
from app.services.order_service import FakeTxn, place_order


def test_rollback_on_exception():
    txn = FakeTxn()
    def boom(): raise RuntimeError("payment failed")
    with pytest.raises(RuntimeError):
        place_order(txn, boom)
    assert txn.rolled_back and not txn.committed

def test_commit_on_success():
    txn = FakeTxn()
    place_order(txn, lambda: None)
    assert txn.committed and not txn.rolled_back
