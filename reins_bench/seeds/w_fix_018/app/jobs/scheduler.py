"""Scheduler that compares naive `now()` with tz-aware target."""
from datetime import datetime, timezone


def is_due(target_utc: datetime, now_local: datetime | None = None) -> bool:
    # BUG: naive vs aware comparison raises or gives wrong answer.
    now = now_local or datetime.now()
    return now >= target_utc
