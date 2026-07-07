"""Invoice total computed with float arithmetic — accumulates rounding error."""


def total_cents(line_items: list[dict]) -> int:
    # BUG: floats lose precision on long sums; cast to int loses cents.
    total = 0.0
    for item in line_items:
        total += float(item["qty"]) * float(item["unit_price_cents"]) / 1.0
    return int(total)  # truncates, may be 1 cent low
