"""Users pagination endpoint with an off-by-one bug.

limit=20 returns 21 records because the cursor decrement happens
after the slice (so the slice keeps one extra item).
"""
from typing import Sequence


def list_users(all_users: Sequence[dict], cursor: int, limit: int) -> list[dict]:
    # BUG: slice first, then decrement cursor — caller ends up keeping
    # one extra record because the slice end isn't tightened.
    page = all_users[cursor:cursor + limit + 1]
    cursor -= 1
    return list(page)
