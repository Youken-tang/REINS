from app.metrics.aggregator import aggregate_total


def test_no_overflow_on_large_sum():
    samples = [10**9] * 5  # 5e9, would overflow int32
    assert aggregate_total(samples) == 5 * 10**9
