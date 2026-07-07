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
    users = db.all_users()
    grouped = db.all_orders_grouped()
    return [dict(u, orders=grouped.get(u["id"], [])) for u in users]
