"""Search endpoint with string-concat SQL — injection."""


def build_query(table: str, term: str) -> str:
    # BUG: string concat — `term` controls the query.
    return f"SELECT * FROM {table} WHERE name LIKE '%{term}%'"


def execute_search(db, table: str, term: str) -> list[dict]:
    return db.execute(build_query(table, term))
