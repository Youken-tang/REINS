from app.api.admin_users import list_admin_users


def test_pagination_returns_limit():
    users = [{"id": i} for i in range(120)]
    page = list_admin_users(users, cursor=0, limit=50)
    assert len(page["items"]) == 50
    assert page["next_cursor"] == 50

def test_last_page_signals_end():
    users = [{"id": i} for i in range(60)]
    page = list_admin_users(users, cursor=50, limit=50)
    assert len(page["items"]) == 10
    assert page["next_cursor"] is None
