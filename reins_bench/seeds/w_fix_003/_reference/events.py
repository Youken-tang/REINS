from datetime import datetime, timezone


def serialise_event(event: dict) -> dict:
    created_at = event.get("created_at") or datetime.now(timezone.utc)
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    return {
        "id": event["id"],
        "created_at": created_at.isoformat(),
    }
