import os
from app.config.settings import get_database_url


def test_env_picked_up_after_set(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgres://x")
    assert get_database_url() == "postgres://x"

def test_default_when_unset(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert get_database_url() == "sqlite://default"
