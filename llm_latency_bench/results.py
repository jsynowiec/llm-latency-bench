"""raw.csv rows and the aggregated.csv report built from them."""

import csv
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import IO, Any

from llm_latency_bench.measure import Attempt
from llm_latency_bench.models_csv import ModelPair
from llm_latency_bench.scenarios import Scenario
from llm_latency_bench.stats import PERCENTILES, percentile, present_values

RAW_COLUMNS = (
    "run_id",
    "phase",
    "request_seq",
    "started_at_utc",
    "scenario_id",
    "scenario_order",
    "repetition",
    "seed",
    "model_name",
    "model_slug",
    "reasoning_level_name",
    "reasoning_level",
    "prompt_size",
    "prompt_target_tokens",
    "context_size",
    "context_target_tokens",
    "turn_count",
    "turn_index",
    "is_final_turn",
    "retry_index",
    "status",
    "error_category",
    "http_status",
    "error_message",
    "finish_reason",
    "response_model",
    "upstream_provider",
    "generation_id",
    "input_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "output_tokens",
    "reasoning_tokens",
    "visible_output_tokens",
    "visible_chars",
    "ttft_s",
    "post_ttft_s",
    "request_latency_s",
    "tps",
    "scenario_status",
    "scenario_total_s",
    "scenario_backoff_s",
    "scenario_started_at_utc",
    "scenario_ended_at_utc",
)

GROUP_COLUMNS = (
    "model_name",
    "model_slug",
    "reasoning_level_name",
    "reasoning_level",
    "prompt_size",
    "prompt_target_tokens",
    "context_size",
    "context_target_tokens",
    "turn_count",
)
# Sizes sort by their target tokens, not by label.
_SORT_COLUMNS = ("model_slug", "reasoning_level", "prompt_target_tokens", "context_target_tokens", "turn_count")
METRICS = ("ttft_s", "request_latency_s", "tps", "scenario_total_s")
TOKEN_MEDIANS = ("input_tokens", "cache_read_tokens", "output_tokens", "reasoning_tokens", "visible_output_tokens")

AGGREGATED_COLUMNS = (
    *GROUP_COLUMNS,
    "scenario_runs",
    "successful_samples",
    "failure_count",
    "failure_rate",
    "failed_attempts",
    "retried_scenarios",
    "tps_samples",
    *(f"{metric}_p{q}" for metric in METRICS for q in PERCENTILES),
    *(f"{name}_p50" for name in TOKEN_MEDIANS),
    "p95_low_confidence",
    "p99_low_confidence",
)

type Row = dict[str, Any]


def format_cell(value: object) -> str:
    """CSV text for one value. None stays an empty cell, never a zero."""
    match value:
        case None:
            return ""
        case bool():
            return "true" if value else "false"
        case float():
            return f"{value:.6f}"
    return str(value)


class RawWriter:
    """Appends rows to raw.csv and flushes each one, so an interrupted run keeps everything sent so far."""

    def __init__(self, handle: IO[str]) -> None:
        self._handle = handle
        self._writer = csv.DictWriter(handle, fieldnames=RAW_COLUMNS)
        self._writer.writeheader()
        self.rows: list[Row] = []

    def write(self, row: Row) -> None:
        unknown = row.keys() - set(RAW_COLUMNS)
        if unknown:
            raise ValueError(f"unknown raw.csv column(s): {', '.join(sorted(unknown))}")
        # Store every column, so readers never meet a missing key.
        full = {column: row.get(column) for column in RAW_COLUMNS}
        self._writer.writerow({column: format_cell(value) for column, value in full.items()})
        self._handle.flush()
        self.rows.append(full)


def pair_columns(pair: ModelPair) -> Row:
    return {
        "model_name": pair.model_name,
        "model_slug": pair.model_slug,
        "reasoning_level_name": pair.reasoning_level_name,
        "reasoning_level": pair.reasoning_level,
    }


def scenario_columns(scenario: Scenario, order: int) -> Row:
    return pair_columns(scenario.pair) | {
        "phase": "measure",
        "scenario_id": scenario.scenario_id,
        "scenario_order": order,
        "repetition": scenario.repetition,
        "prompt_size": scenario.prompt_size,
        "prompt_target_tokens": scenario.prompt_tokens,
        "context_size": scenario.context_size,
        "context_target_tokens": scenario.context_tokens,
        "turn_count": scenario.turn_count,
    }


def attempt_columns(attempt: Attempt, retry_index: int) -> Row:
    usage = attempt.usage
    return {
        "started_at_utc": attempt.started_at_utc.isoformat(),
        "retry_index": retry_index,
        "status": "ok" if attempt.ok else "error",
        "error_category": attempt.error_category,
        "http_status": attempt.http_status,
        "error_message": attempt.error_message,
        "finish_reason": attempt.finish_reason,
        "response_model": attempt.response_model,
        "upstream_provider": attempt.upstream_provider,
        "generation_id": attempt.generation_id,
        "input_tokens": usage.input_tokens,
        "cache_read_tokens": usage.cache_read_tokens,
        "cache_write_tokens": usage.cache_write_tokens,
        "output_tokens": usage.output_tokens,
        "reasoning_tokens": usage.reasoning_tokens,
        "visible_output_tokens": usage.visible_output_tokens,
        "visible_chars": attempt.visible_chars,
        "ttft_s": attempt.ttft_s,
        "post_ttft_s": attempt.post_ttft_s,
        "request_latency_s": attempt.request_latency_s,
        "tps": attempt.tps,
    }


def scenario_key(row: Mapping[str, Any]) -> tuple[str, str]:
    """Identifies one scenario run. Combined results can hold the same scenario ID from several runs."""
    return row["run_id"], row["scenario_id"]


def is_valid_sample(row: Mapping[str, Any]) -> bool:
    """A final-turn attempt that succeeded."""
    return row["phase"] == "measure" and bool(row["is_final_turn"]) and row["status"] == "ok"


def aggregate(rows: Iterable[Row], min_samples_p95: int, min_samples_p99: int) -> list[Row]:
    """One row per model/reasoning/prompt size/context size/turn count, from final-turn samples."""
    groups: dict[tuple[Any, ...], list[Row]] = defaultdict(list)
    for row in rows:
        if row["phase"] == "measure":
            groups[tuple(row[column] for column in GROUP_COLUMNS)].append(row)

    def by_size(item: tuple[tuple[Any, ...], list[Row]]) -> list[Any]:
        first = item[1][0]
        return [first[column] for column in _SORT_COLUMNS]

    report: list[Row] = []
    for key, group in sorted(groups.items(), key=by_size):
        scenario_ids = {scenario_key(row) for row in group}
        samples = [row for row in group if is_valid_sample(row)]
        failures = len(scenario_ids) - len(samples)
        retried = {scenario_key(row) for row in group if (row["retry_index"] or 0) > 0}

        out: Row = dict(zip(GROUP_COLUMNS, key, strict=True))
        out |= {
            "scenario_runs": len(scenario_ids),
            "successful_samples": len(samples),
            "failure_count": failures,
            "failure_rate": failures / len(scenario_ids),
            "failed_attempts": sum(row["status"] == "error" for row in group),
            "retried_scenarios": len(retried),
            "tps_samples": len(present_values(samples, "tps")),
        }
        for metric in METRICS:
            values = present_values(samples, metric)
            out |= {f"{metric}_p{q}": percentile(values, q) for q in PERCENTILES}
        for name in TOKEN_MEDIANS:
            out[f"{name}_p50"] = percentile(present_values(samples, name), 50)
        out["p95_low_confidence"] = len(samples) < min_samples_p95
        out["p99_low_confidence"] = len(samples) < min_samples_p99
        report.append(out)
    return report


_INT_COLUMNS = frozenset(
    {
        "request_seq",
        "scenario_order",
        "repetition",
        "seed",
        "prompt_target_tokens",
        "context_target_tokens",
        "turn_count",
        "turn_index",
        "retry_index",
        "http_status",
        "input_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "output_tokens",
        "reasoning_tokens",
        "visible_output_tokens",
        "visible_chars",
    }
)
_FLOAT_COLUMNS = frozenset(
    {"ttft_s", "post_ttft_s", "request_latency_s", "tps", "scenario_total_s", "scenario_backoff_s"}
)


def parse_raw_row(cells: Mapping[str, str]) -> Row:
    """The inverse of `format_cell` for one raw.csv row: empty cells become None, numbers and flags get their type.

    Columns missing from older files are None.
    """
    row: Row = {}
    for column in RAW_COLUMNS:
        text = cells.get(column) or ""
        if not text:
            row[column] = None
        elif column in _INT_COLUMNS:
            row[column] = int(text)
        elif column in _FLOAT_COLUMNS:
            row[column] = float(text)
        elif column == "is_final_turn":
            row[column] = text == "true"
        else:
            row[column] = text
    return row


def read_raw_csv(path: Path) -> list[Row]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [parse_raw_row(cells) for cells in csv.DictReader(handle)]


def write_csv(path: Path, columns: tuple[str, ...], rows: Iterable[Row]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: format_cell(row.get(column)) for column in columns})
