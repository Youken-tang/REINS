import pytest
from app.clients.payment_client import charge, FatalError, TransientError


def test_fatal_error_is_not_retried():
    calls = {"n": 0}
    def do():
        calls["n"] += 1
        raise FatalError("nope")
    with pytest.raises(FatalError):
        charge(do, max_retries=3)
    assert calls["n"] == 1

def test_transient_error_is_retried():
    calls = {"n": 0}
    def do():
        calls["n"] += 1
        if calls["n"] < 2:
            raise TransientError("retry")
        return "ok"
    assert charge(do, max_retries=3) == "ok"
    assert calls["n"] == 2
