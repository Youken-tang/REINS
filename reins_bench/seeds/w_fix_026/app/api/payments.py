"""Payment intent endpoint — duplicate requests double-charge."""


class PaymentService:
    def __init__(self):
        self.charges = []

    def create_intent(self, idempotency_key: str | None, amount: int) -> dict:
        # BUG: idempotency_key ignored — every retry charges again.
        self.charges.append((idempotency_key, amount))
        return {"intent_id": f"pi_{len(self.charges)}", "amount": amount}
