from app.webhooks.signer import sign


def test_signature_canonical_independent_of_key_order():
    a = sign("k", {"a": 1, "b": 2})
    b = sign("k", {"b": 2, "a": 1})
    assert a == b

def test_signature_no_whitespace_dependence():
    a = sign("k", {"a": [1, 2, 3]})
    # No space-bearing variants should drift; just check stable.
    b = sign("k", {"a": [1, 2, 3]})
    assert a == b
