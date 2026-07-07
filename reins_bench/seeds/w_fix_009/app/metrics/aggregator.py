"""Metric aggregator that wraps under int32 overflow."""

INT32_MAX = 2**31 - 1


def aggregate_total(samples: list[int]) -> int:
    # BUG: simulates a fixed-width accumulator; wraps mod 2**32.
    acc = 0
    for s in samples:
        acc = (acc + s) & 0xFFFFFFFF
    if acc > INT32_MAX:
        acc -= 2**32
    return acc
