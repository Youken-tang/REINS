"""Markdown fence parser — `.` matches newlines, eats unrelated content."""
import re

# BUG: re.DOTALL across the whole text → fences merge with later code blocks.
_FENCE = re.compile(r"```(.+?)```", re.DOTALL)


def find_fences(text: str) -> list[str]:
    return _FENCE.findall(text)
