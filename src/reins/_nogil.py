"""Runtime checks for CPython free-threaded builds."""

from __future__ import annotations

import sys
import sysconfig


class NoGILError(RuntimeError):
    """Raised when high-agent is started without a free-threaded Python."""


def is_nogil() -> bool:
    """Return True when the interpreter was built with the GIL disabled."""

    return sysconfig.get_config_var("Py_GIL_DISABLED") == 1


def ensure_nogil(*, strict: bool = True) -> bool:
    """Validate that the process is running on Python 3.13t or newer.

    Args:
        strict: When True, raise NoGILError if the interpreter is not a
            free-threaded CPython build. When False, return the check result.
    """

    ok = is_nogil()
    if ok or not strict:
        return ok
    raise NoGILError(
        "high-agent requires CPython 3.13t free-threading "
        f"(Py_GIL_DISABLED=1); current interpreter is {sys.version.split()[0]}."
    )
