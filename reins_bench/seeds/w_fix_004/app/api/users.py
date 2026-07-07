"""user-with-orders fetcher with N+1 query bug."""

class FakeDB:
    def __init__(self) -> None:
        self.queries = 0
        self._users = [{"id": i, "name": f"u{i}"} for i in range(5)]
        self._orders = {i: [{"oid": i*10+j} for j in range(3)] for i in range(5)}

    def all_users(self) -> list[dict]:
        self.queries += 1
        return list(self._users)

    def orders_for_user(self, uid: int) -> list[dict]:
        self.queries += 1
        return list(self._orders.get(uid, []))

    def all_orders_grouped(self) -> dict[int, list[dict]]:
        self.queries += 1
        return {k: list(v) for k, v in self._orders.items()}


def list_users_with_orders(db: FakeDB) -> list[dict]:
    # BUG: 1 query for users, then 1 per user for orders → N+1.
    users = db.all_users()
    out = []
    for u in users:
        u2 = dict(u)
        u2["orders"] = db.orders_for_user(u["id"])
        out.append(u2)
    return out
