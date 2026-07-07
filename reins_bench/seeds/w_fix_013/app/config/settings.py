"""Settings module reads env at import time — tests can't override."""
import os

# BUG: captured at import; later os.environ changes ignored.
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite://default")


def get_database_url() -> str:
    return DATABASE_URL
