import math


def reconcile(observed: float, expected: float, *, tol: float = 1e-9) -> bool:
    return math.isclose(observed, expected, rel_tol=tol, abs_tol=tol)
