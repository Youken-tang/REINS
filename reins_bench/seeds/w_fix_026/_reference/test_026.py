from app.api.payments import PaymentService


def test_same_key_returns_same_intent():
    s = PaymentService()
    a = s.create_intent("k1", 1000)
    b = s.create_intent("k1", 1000)
    assert a == b
    assert len(s.charges) == 1

def test_different_keys_charge_separately():
    s = PaymentService()
    s.create_intent("k1", 100)
    s.create_intent("k2", 100)
    assert len(s.charges) == 2
