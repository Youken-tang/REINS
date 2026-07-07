from app.billing.invoice import total_cents


def test_no_floating_point_error_on_many_items():
    items = [{"qty": 1, "unit_price_cents": 33} for _ in range(1000)]
    # 1000 × 33 = 33000 cents exactly
    assert total_cents(items) == 33000

def test_fractional_quantity():
    items = [{"qty": 3, "unit_price_cents": 99}]
    assert total_cents(items) == 297
