"""Compatibility shim — code lives in :mod:`reins.controller`.

Forwards every public *and* private name so existing tests that deep-import
private symbols (``_PlannerRequest``, ``_PlannerResult``, …) keep working.
"""

from reins import controller as _controller

globals().update(
    {name: getattr(_controller, name) for name in dir(_controller) if not name.startswith("__")}
)

del _controller
