import pytest

from llm_latency_bench.stats import percentile

# Unsorted, unevenly spaced, with one large outlier. Expected values are computed by hand with
# linear interpolation between closest ranks: rank = (n - 1) * q / 100 over the sorted values [1, 2, 4, 8, 100].
UNEVEN = [8.0, 1.0, 100.0, 4.0, 2.0]


@pytest.mark.parametrize(
    ("q", "expected"),
    [
        (0, 1.0),
        (10, 1.4),  # rank 0.4: 1 + 0.4 * (2 - 1)
        (25, 2.0),  # rank 1.0
        (50, 4.0),  # rank 2.0
        (75, 8.0),  # rank 3.0
        (95, 81.6),  # rank 3.8: 8 + 0.8 * (100 - 8)
        (99, 96.32),  # rank 3.96: 8 + 0.96 * 92
        (100, 100.0),
    ],
)
def test_percentile_interpolates_between_closest_ranks(q, expected):
    assert percentile(UNEVEN, q) == pytest.approx(expected)


def test_even_count_median_is_midpoint():
    assert percentile([3.0, 1.0, 10.0, 2.0], 50) == pytest.approx(2.5)


def test_no_data_is_none_not_zero():
    assert percentile([], 50) is None


def test_single_value_is_every_percentile():
    assert percentile([0.7], 99) == 0.7


def test_out_of_range_percentile_is_rejected():
    with pytest.raises(ValueError):
        percentile([1.0], 101)
