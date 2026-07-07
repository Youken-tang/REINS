from app.api.users import list_users


def test_limit_20_returns_20():
    users = [{"id": i} for i in range(100)]
    page = list_users(users, cursor=0, limit=20)
    assert len(page) == 20

def test_limit_5_returns_5():
    users = [{"id": i} for i in range(10)]
    page = list_users(users, cursor=2, limit=5)
    assert len(page) == 5
    assert page[0]["id"] == 2
