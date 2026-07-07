import os


def get_database_url() -> str:
    return os.environ.get("DATABASE_URL", "sqlite://default")
