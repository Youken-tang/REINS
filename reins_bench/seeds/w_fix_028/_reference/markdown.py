import re

# Match ``` … ``` on a per-block basis without spanning unrelated paragraphs.
_FENCE = re.compile(r"```([\s\S]*?)```")


def find_fences(text: str) -> list[str]:
    return _FENCE.findall(text)
