import pytest
from app.api.profile import ProfileService


def test_missing_token_rejected():
    s = ProfileService()
    with pytest.raises(PermissionError):
        s.update(1, None, {"name": "a"})

def test_wrong_token_rejected():
    s = ProfileService()
    with pytest.raises(PermissionError):
        s.update(1, "bogus", {"name": "a"})

def test_valid_token_accepted():
    s = ProfileService()
    s.update(1, "valid", {"name": "alice"})
    assert s.profiles[1]["name"] == "alice"
