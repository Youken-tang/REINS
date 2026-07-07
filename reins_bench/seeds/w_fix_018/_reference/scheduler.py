from datetime import datetime, timezone


def is_due(target_utc: datetime, now_local: datetime | None = None) -> bool:
    now = now_local or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    if target_utc.tzinfo is None:
        target_utc = target_utc.replace(tzinfo=timezone.utc)
    return now >= target_utc
