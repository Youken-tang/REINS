def read_lines(path: str) -> list[str]:
    with open(path, "rb") as f:
        raw = f.read()
    return raw.decode("utf-8", errors="replace").splitlines()
