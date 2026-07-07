import time
from app.parsers.log_parser import matches_pattern


def test_pathological_input_runs_fast():
    pathological = "a" * 30  # would explode under (a+)+b
    t0 = time.monotonic()
    matches_pattern(pathological)
    assert time.monotonic() - t0 < 0.5

def test_basic_match():
    assert matches_pattern("aaab")
    assert not matches_pattern("aaa")
