"""Compatibility shim — code lives in :mod:`reins._nogil`."""

from reins._nogil import (  # noqa: F401
    NoGILError,
    ensure_nogil,
    is_nogil,
)
