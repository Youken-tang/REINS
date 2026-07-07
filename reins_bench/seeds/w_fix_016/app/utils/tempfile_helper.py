"""Temp file helper with TOCTOU race."""
import os
import tempfile


def create_temp(prefix: str = "x") -> str:
    # BUG: name reserved then opened separately — attacker could
    # symlink in between. Should atomically open instead.
    d = tempfile.gettempdir()
    name = os.path.join(d, prefix + str(os.getpid()))
    if os.path.exists(name):
        os.unlink(name)
    with open(name, "w") as f:
        f.write("")
    return name
