from datetime import datetime, timezone
from app.api.events import serialise_event


def test_serialised_iso_includes_timezone():
    evt = {"id": 1, "created_at": datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)}
    out = serialise_event(evt)
    iso = out["created_at"]
    assert iso.endswith("+00:00") or iso.endswith("Z")

def test_naive_datetime_is_assumed_utc():
    evt = {"id": 2, "created_at": datetime(2026, 1, 1, 12, 0, 0)}
    out = serialise_event(evt)
    iso = out["created_at"]
    assert iso.endswith("+00:00") or iso.endswith("Z")
