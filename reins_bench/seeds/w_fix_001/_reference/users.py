from typing import Sequence


def list_users(all_users: Sequence[dict], cursor: int, limit: int) -> list[dict]:
    page = all_users[cursor:cursor + limit]
    return list(page)
