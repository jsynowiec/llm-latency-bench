"""Percentiles used by the aggregated report."""

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

PERCENTILES = (10, 25, 50, 75, 95, 99)


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear interpolation between closest ranks (Hyndman-Fan type 7, the numpy default).

    Returns None for no data, so an empty metric stays an empty CSV cell instead of a zero.
    """
    if not 0 <= q <= 100:
        raise ValueError(f"percentile must be within [0, 100], got {q}")
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q / 100
    lower = math.floor(rank)
    upper = math.ceil(rank)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def present_values(rows: Iterable[Mapping[str, Any]], column: str) -> list[float]:
    """The column's values as floats, skipping rows where the value is missing."""
    return [float(row[column]) for row in rows if row.get(column) is not None]
