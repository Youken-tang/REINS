import time
from app.auth.token import issue, verify


def test_expired_rejected():
    t = issue({"sub": "alice"}, "secret", ttl_seconds=-1)
    assert verify(t, "secret") is None

def test_valid_accepted():
    t = issue({"sub": "alice"}, "secret", ttl_seconds=60)
    p = verify(t, "secret")
    assert p and p["sub"] == "alice"

def test_tampered_rejected():
    t = issue({"sub": "alice"}, "secret", ttl_seconds=60)
    body, _ = t.rsplit(".", 1)
    bad = body + "." + "0" * 64
    assert verify(bad, "secret") is None
