"""Compatibility shim — code lives in :mod:`reins.runtime.scheduler`.

Forwards every public *and* private name so existing callers and tests that
deep-import private names continue to work transparently.
"""

from reins.runtime import scheduler as _scheduler

globals().update(
    {name: getattr(_scheduler, name) for name in dir(_scheduler) if not name.startswith("__")}
)

del _scheduler
