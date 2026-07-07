import os
import tempfile


def create_temp(prefix: str = "x") -> str:
    fd, name = tempfile.mkstemp(prefix=prefix)
    os.close(fd)
    return name
