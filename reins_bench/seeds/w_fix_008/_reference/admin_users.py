def list_admin_users(db_users: list[dict], cursor: int = 0, limit: int = 50) -> dict:
    page = db_users[cursor:cursor + limit]
    next_cursor = cursor + limit if cursor + limit < len(db_users) else None
    return {"items": list(page), "next_cursor": next_cursor}
