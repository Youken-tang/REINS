"""Compatibility shim — code lives in :mod:`reins.tool_calls`."""

from reins import tool_calls as _tool_calls

globals().update(
    {name: getattr(_tool_calls, name) for name in dir(_tool_calls) if not name.startswith("__")}
)

del _tool_calls
