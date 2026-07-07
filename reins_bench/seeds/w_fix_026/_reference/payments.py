class PaymentService:
    def __init__(self):
        self.charges = []
        self._by_key: dict[str, dict] = {}

    def create_intent(self, idempotency_key: str | None, amount: int) -> dict:
        if idempotency_key and idempotency_key in self._by_key:
            return self._by_key[idempotency_key]
        self.charges.append((idempotency_key, amount))
        intent = {"intent_id": f"pi_{len(self.charges)}", "amount": amount}
        if idempotency_key:
            self._by_key[idempotency_key] = intent
        return intent
