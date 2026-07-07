from datetime import datetime, timezone, timedelta
from app.jobs.scheduler import is_due


def test_due_when_target_in_past_utc():
    past = datetime.now(timezone.utc) - timedelta(hours=1)
    assert is_due(past)

def test_not_due_when_target_in_future_utc():
    future = datetime.now(timezone.utc) + timedelta(hours=1)
    assert not is_due(future)
