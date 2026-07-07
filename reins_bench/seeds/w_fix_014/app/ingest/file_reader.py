"""File reader silently drops bytes that aren't valid UTF-8."""


def read_lines(path: str) -> list[str]:
    # BUG: errors='ignore' silently drops invalid bytes — losing data.
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().splitlines()
