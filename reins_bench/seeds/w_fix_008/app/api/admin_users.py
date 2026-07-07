"""Admin endpoint returns all users at once — no pagination."""


def list_admin_users(db_users: list[dict], cursor: int = 0, limit: int = 50) -> dict:
    # BUG: returns full list, ignores cursor/limit.
    return {"items": list(db_users), "next_cursor": None}
