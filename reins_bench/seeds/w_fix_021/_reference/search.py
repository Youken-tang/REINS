_ALLOWED_TABLES = {"users", "products"}


def build_query(table: str, term: str) -> tuple[str, tuple]:
    if table not in _ALLOWED_TABLES:
        raise ValueError(f"disallowed table: {table}")
    return f"SELECT * FROM {table} WHERE name LIKE ?", (f"%{term}%",)


def execute_search(db, table: str, term: str) -> list[dict]:
    sql, params = build_query(table, term)
    return db.execute(sql, params)
