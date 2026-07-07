import re

_GOOD = re.compile(r"^a+b$")


def matches_pattern(line: str) -> bool:
    return bool(_GOOD.match(line))
