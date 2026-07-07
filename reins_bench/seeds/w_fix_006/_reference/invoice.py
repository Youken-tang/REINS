from decimal import Decimal


def total_cents(line_items: list[dict]) -> int:
    total = Decimal(0)
    for item in line_items:
        total += Decimal(item["qty"]) * Decimal(item["unit_price_cents"])
    return int(total)
