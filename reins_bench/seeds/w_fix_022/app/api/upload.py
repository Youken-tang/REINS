"""Upload endpoint with no input validation."""


def handle_upload(filename: str, size: int, mime: str) -> dict:
    # BUG: accepts anything — including 100GB and arbitrary mime.
    return {"ok": True, "stored_as": filename}
