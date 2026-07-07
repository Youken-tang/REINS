import re

_MAX_SIZE = 10 * 1024 * 1024
_ALLOWED_MIME = {"image/png", "image/jpeg", "application/pdf"}
_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


def handle_upload(filename: str, size: int, mime: str) -> dict:
    if not _SAFE_NAME.match(filename) or len(filename) > 255:
        raise ValueError("invalid filename")
    if size <= 0 or size > _MAX_SIZE:
        raise ValueError("invalid size")
    if mime not in _ALLOWED_MIME:
        raise ValueError("disallowed mime type")
    return {"ok": True, "stored_as": filename}
