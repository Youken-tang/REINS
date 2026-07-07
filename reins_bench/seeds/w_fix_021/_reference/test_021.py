import pytest
from app.api.search import build_query


def test_term_not_inlined():
    sql, params = build_query("users", "alice'; DROP TABLE users; --")
    assert "DROP TABLE" not in sql
    assert "alice" in params[0]

def test_table_whitelist():
    with pytest.raises(ValueError):
        build_query("users; DROP TABLE", "x")
