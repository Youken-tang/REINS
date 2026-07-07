"""CSV importer that leaks file handles."""
import csv


def import_rows(path: str) -> list[dict]:
    # BUG: file opened but never closed (no `with`).
    f = open(path, "r", encoding="utf-8")
    reader = csv.DictReader(f)
    return list(reader)
