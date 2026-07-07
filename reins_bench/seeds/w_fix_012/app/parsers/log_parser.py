"""Log parser with catastrophic regex backtracking on long inputs."""
import re

# BUG: nested quantifier (a+)+ → exponential on input 'a'*N + 'b'
_BAD = re.compile(r"^(a+)+b$")


def matches_pattern(line: str) -> bool:
    return bool(_BAD.match(line))
