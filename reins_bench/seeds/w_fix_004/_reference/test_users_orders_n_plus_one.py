from app.api.users import FakeDB, list_users_with_orders


def test_constant_query_count():
    db = FakeDB()
    list_users_with_orders(db)
    # 1 for users + 1 for grouped orders = 2 total, not N+1.
    assert db.queries <= 2
