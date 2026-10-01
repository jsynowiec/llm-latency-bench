"""Plain-text summary of a run. It prints measurements only; interpreting them is left to the reader."""

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence

from llm_latency_bench.results import Row, is_valid_sample, scenario_key
from llm_latency_bench.stats import percentile, present_values


def _fmt(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _p50(rows: Iterable[Row], column: str) -> float | None:
    return percentile(present_values(rows, column), 50)


def _table(title: str, header: Sequence[str], body: Sequence[Sequence[str]]) -> list[str]:
    widths = [max(len(cell) for cell in column) for column in zip(header, *body, strict=True)]
    lines = ["  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)) for row in [header, *body]]
    return ["", f"== {title} ==", lines[0], "  ".join("-" * width for width in widths), *lines[1:]]


def _pair_key(row: Row) -> str:
    return f"{row['model_slug']}/{row['reasoning_level']}"


def _by_pair(rows: Iterable[Row]) -> dict[str, list[Row]]:
    grouped: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        grouped[_pair_key(row)].append(row)
    return dict(sorted(grouped.items()))


def _levels(rows: Iterable[Row], label_column: str, tokens_column: str) -> list[str]:
    """Size labels present in the rows, smallest target size first."""
    return [label for label, _ in sorted({(row[label_column], row[tokens_column]) for row in rows}, key=lambda x: x[1])]


def _ttft_p50_by_level(rows: list[Row], column: str, levels: Sequence[object]) -> list[str]:
    return [_fmt(_p50([row for row in rows if row[column] == level], "ttft_s")) for level in levels]


def _tukey_outliers(values: list[float]) -> int:
    q1, q3 = percentile(values, 25), percentile(values, 75)
    if q1 is None or q3 is None:
        return 0
    fence = q3 + 1.5 * (q3 - q1)
    return sum(value > fence for value in values)


def _totals(rows: list[Row], measure: list[Row], samples: list[Row]) -> list[str]:
    sent = [row for row in rows if row["status"] != "not_sent"]
    lines = [
        "== Run totals ==",
        f"scenario runs: {len({scenario_key(row) for row in measure})}, valid final-turn samples: {len(samples)}, "
        f"requests sent: {len(sent)} (incl. warm-up and retries), "
        f"failed attempts: {sum(row['status'] == 'error' for row in sent)}, "
        f"retries: {sum((row['retry_index'] or 0) > 0 for row in sent)}, "
        f"not sent: {len(rows) - len(sent)}",
    ]
    providers = Counter(row["upstream_provider"] or "not reported" for row in sent)
    lines.append("upstream providers: " + ", ".join(f"{name}={n}" for name, n in providers.most_common()))
    categories = Counter(row["error_category"] for row in rows if row["status"] != "ok")
    if categories:
        lines.append("errors by category: " + ", ".join(f"{name}={n}" for name, n in categories.most_common()))
    return lines


def _distribution(samples: list[Row], min_samples_p95: int) -> list[str]:
    body = []
    for pair, rows in _by_pair(samples).items():
        ttft = present_values(rows, "ttft_s")
        latency = present_values(rows, "request_latency_s")
        body.append(
            [
                pair,
                str(len(rows)),
                *(_fmt(percentile(ttft, q)) for q in (10, 50, 95)),
                *(_fmt(percentile(latency, q)) for q in (50, 95)),
                _fmt(_p50(rows, "tps"), 1),
                str(_tukey_outliers(ttft)),
            ]
        )
    header = ["pair", "n", "TTFT p10", "TTFT p50", "TTFT p95", "latency p50", "latency p95", "TPS p50", "TTFT outliers"]
    title = "Final-turn distribution per model/reasoning pair (seconds; TPS in visible tokens/s)"
    note = f"p95 over fewer than {min_samples_p95} samples is a low-confidence estimate. Outliers: > p75 + 1.5 x IQR."
    return [*_table(title, header, body), note]


def _ttft_by_factor(
    samples: list[Row], prompt_levels: list[str], context_levels: list[str], turn_counts: Sequence[int]
) -> list[str]:
    body = [
        [
            pair,
            *_ttft_p50_by_level(rows, "prompt_size", prompt_levels),
            *_ttft_p50_by_level(rows, "context_size", context_levels),
            *_ttft_p50_by_level(rows, "turn_count", turn_counts),
        ]
        for pair, rows in _by_pair(samples).items()
    ]
    header = [
        "pair",
        *(f"prompt {size}" for size in prompt_levels),
        *(f"context {size}" for size in context_levels),
        *(f"turns {count}" for count in turn_counts),
    ]
    return _table("TTFT p50 by factor, pooled over the other two factors (final turn, seconds)", header, body)


def _context_scaling(samples: list[Row], context_levels: list[str]) -> list[str]:
    body = []
    for pair, rows in _by_pair(samples).items():
        cells = []
        for level in context_levels:
            at_level = [row for row in rows if row["context_size"] == level]
            cells.append(f"{_fmt(_p50(at_level, 'ttft_s'))}/{_fmt(_p50(at_level, 'reasoning_tokens'), 0)}")
        body.append([pair, *cells])
    header = ["pair", *(f"context {level} TTFT/reasoning" for level in context_levels)]
    title = "TTFT p50 (s) and hidden reasoning tokens p50 by context size (final turn)"
    return _table(title, header, body)


def _tokens(samples: list[Row], turn_counts: Sequence[int]) -> list[str]:
    body = []
    for pair, rows in _by_pair(samples).items():
        cells = []
        for count in turn_counts:
            at_count = [row for row in rows if row["turn_count"] == count]
            cells.append(f"{_fmt(_p50(at_count, 'input_tokens'), 0)}/{_fmt(_p50(at_count, 'cache_read_tokens'), 0)}")
        body.append([pair, *cells, _fmt(_p50(rows, "output_tokens"), 0), _fmt(_p50(rows, "reasoning_tokens"), 0)])
    header = ["pair", *(f"turns {count} in/cached" for count in turn_counts), "output tokens", "reasoning tokens"]
    return _table("Input tokens and cache reads at the final turn (p50)", header, body)


def _first_turn_control(first_turns: list[Row], turn_counts: Sequence[int]) -> list[str]:
    multi_turn = [count for count in turn_counts if count > 1]
    body = [[pair, *_ttft_p50_by_level(rows, "turn_count", multi_turn)] for pair, rows in _by_pair(first_turns).items()]
    title = (
        "Control: TTFT p50 of turn 1, by turn count (the first request is identical when there is more than one turn)"
    )
    return _table(title, ["pair", *(f"turns {count}" for count in multi_turn)], body)


def _warmup(rows: list[Row], first_turns: list[Row]) -> list[str]:
    # The last warm-up row of a pair in a run is its final attempt. Combined data holds several runs.
    finals = {(row["run_id"], _pair_key(row)): row for row in rows if row["phase"] == "warmup"}
    first_by_pair = _by_pair(first_turns)
    body = []
    for pair, warmups in _by_pair(finals.values()).items():
        status = ", ".join(f"{name}={n}" for name, n in sorted(Counter(row["status"] for row in warmups).items()))
        body.append([pair, status, _fmt(_p50(warmups, "ttft_s")), _fmt(_p50(first_by_pair.get(pair, []), "ttft_s"))])
    header = ["pair", "warm-up status", "warm-up TTFT p50", "turn-1 TTFT p50"]
    return _table("Warm-up (first request per pair) vs measured turn-1 TTFT p50", header, body)


def _failures(measure: list[Row]) -> list[str]:
    body = []
    for pair, rows in _by_pair(measure).items():
        runs = len({scenario_key(row) for row in rows})
        missing = runs - sum(is_valid_sample(row) for row in rows)
        body.append([pair, str(runs), str(missing), _fmt(missing / runs)])
    header = ["pair", "scenario runs", "failed", "rate"]
    return _table("Failure rate per pair (scenario runs without a valid final-turn sample)", header, body)


def summarize(rows: list[Row], min_samples_p95: int) -> str:
    measure = [row for row in rows if row["phase"] == "measure"]
    prompt_levels = _levels(measure, "prompt_size", "prompt_target_tokens")
    context_levels = _levels(measure, "context_size", "context_target_tokens")
    turn_counts = sorted({row["turn_count"] for row in measure})
    samples = [row for row in measure if is_valid_sample(row)]
    first_turns = [row for row in measure if row["turn_index"] == 1 and row["status"] == "ok"]
    return "\n".join(
        [
            *_totals(rows, measure, samples),
            *_distribution(samples, min_samples_p95),
            *_ttft_by_factor(samples, prompt_levels, context_levels, turn_counts),
            *_context_scaling(samples, context_levels),
            *_tokens(samples, turn_counts),
            *_first_turn_control(first_turns, turn_counts),
            *_warmup(rows, first_turns),
            *_failures(measure),
        ]
    )
