"""Events API serialiser — timezone-naive datetime bug."""
from datetime import datetime


def serialise_event(event: dict) -> dict:
    # BUG: utcnow() is naive — no tzinfo on the wire.
    created_at = event.get("created_at") or datetime.utcnow()
    return {
        "id": event["id"],
        "created_at": created_at.isoformat(),
    }
