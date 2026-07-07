import pytest
from app.api.auth import LoginService


def test_rate_limit_after_threshold():
    s = LoginService(lambda u, p: False)
    for _ in range(5):
        s.login("1.2.3.4", "u", "p")
    with pytest.raises(PermissionError):
        s.login("1.2.3.4", "u", "p")

def test_other_ip_unaffected():
    s = LoginService(lambda u, p: False)
    for _ in range(5):
        s.login("1.2.3.4", "u", "p")
    # Different IP should still be allowed.
    s.login("5.6.7.8", "u", "p")
