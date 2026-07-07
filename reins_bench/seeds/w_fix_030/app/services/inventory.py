"""Inventory reconciler that uses == on floats."""


def reconcile(observed: float, expected: float) -> bool:
    # BUG: floating-point == is fragile (0.1 + 0.2 != 0.3, etc.)
    return observed == expected
