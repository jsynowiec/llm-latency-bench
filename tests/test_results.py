import csv
import io

import pytest

from llm_latency_bench.results import AGGREGATED_COLUMNS, RAW_COLUMNS, RawWriter, aggregate, read_raw_csv, write_csv


def final_row(scenario_id: str, *, status: str = "ok", ttft: float | None = 1.0, tps: float | None = None) -> dict:
    return {
        "phase": "measure",
        "run_id": "run-1",
        "scenario_id": scenario_id,
        "model_name": "M",
        "model_slug": "m",
        "reasoning_level_name": "High",
        "reasoning_level": "high",
        "prompt_size": "short",
        "prompt_target_tokens": 40,
        "context_size": "long",
        "context_target_tokens": 10000,
        "turn_count": 1,
        "is_final_turn": True,
        "retry_index": 0,
        "status": status,
        "error_category": None,
        "upstream_provider": None,
        "ttft_s": ttft,
        "request_latency_s": ttft,
        "tps": tps,
        "scenario_total_s": ttft,
    }


def turn_row(
    scenario_id: str,
    turn_index: int,
    *,
    run_id: str = "run-1",
    retry_index: int = 0,
    status: str = "ok",
    ttft: float | None = None,
    input_tokens: int | None = None,
) -> dict:
    """One attempt at a turn of a three-turn scenario."""
    return final_row(scenario_id, status=status, ttft=ttft) | {
        "run_id": run_id,
        "turn_count": 3,
        "turn_index": turn_index,
        "is_final_turn": turn_index == 3,
        "retry_index": retry_index,
        "input_tokens": input_tokens,
    }


def three_turn_cell() -> list[dict]:
    return [
        turn_row("s1", 1, ttft=0.5, input_tokens=100),
        turn_row("s1", 2, status="error"),
        turn_row("s1", 2, retry_index=1, ttft=0.5, input_tokens=220),
        turn_row("s1", 3, ttft=2.0, input_tokens=3000),
        turn_row("s2", 1, ttft=0.5, input_tokens=150),
        turn_row("s2", 2, ttft=0.5, input_tokens=250),
        turn_row("s2", 3, ttft=7.0, input_tokens=3400),
        # Aborted at turn 2 after two retries.
        turn_row("s3", 1, ttft=0.5, input_tokens=120),
        turn_row("s3", 2, status="error"),
        turn_row("s3", 2, retry_index=1, status="error"),
        turn_row("s3", 2, retry_index=2, status="error"),
        # Combined results: the same scenario ID from another run is another scenario run.
        turn_row("s1", 1, run_id="run-2", ttft=0.5, input_tokens=110),
        turn_row("s1", 2, run_id="run-2", ttft=0.5, input_tokens=230),
        turn_row("s1", 3, run_id="run-2", ttft=3.0, input_tokens=3100),
        turn_row("s4", 3, status="not_sent") | {"retry_index": None, "error_category": "run_aborted"},
    ]


def test_aggregate_counts_scenario_runs_and_takes_metrics_from_final_turn_samples():
    [cell] = aggregate(three_turn_cell(), min_samples_p95=3, min_samples_p99=4)

    assert cell["turn_count"] == 3
    assert cell["scenario_runs"] == 5
    assert cell["successful_samples"] == 3
    assert cell["failure_count"] == 2
    assert cell["failure_rate"] == pytest.approx(0.4)
    assert cell["failed_attempts"] == 4
    assert cell["retried_scenarios"] == 2
    # Final-turn TTFTs are 2, 3 and 7 seconds.
    assert cell["ttft_s_p10"] == pytest.approx(2.2)
    assert cell["ttft_s_p50"] == pytest.approx(3.0)
    assert cell["ttft_s_p95"] == pytest.approx(6.6)
    assert cell["input_tokens_p50"] == pytest.approx(3100)
    assert cell["p95_low_confidence"] is False
    assert cell["p99_low_confidence"] is True


def test_aggregate_sorts_cells_by_target_tokens_not_by_label():
    def cell(prompt: str, prompt_tokens: int, context: str, context_tokens: int) -> dict:
        return final_row(f"{prompt}-{context}") | {
            "prompt_size": prompt,
            "prompt_target_tokens": prompt_tokens,
            "context_size": context,
            "context_target_tokens": context_tokens,
        }

    rows = [
        cell("long", 2000, "none", 0),
        cell("short", 40, "large", 20000),
        cell("long", 2000, "large", 20000),
        cell("short", 40, "none", 0),
    ]

    order = [(row["prompt_size"], row["context_size"]) for row in aggregate(rows, 20, 100)]
    assert order == [("short", "none"), ("short", "large"), ("long", "none"), ("long", "large")]


def test_raw_writer_flushes_each_row_and_round_trips_through_read_raw_csv(tmp_path):
    rows = three_turn_cell()
    path = tmp_path / "raw.csv"

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = RawWriter(handle)
        for written, row in enumerate(rows, start=1):
            writer.write(row)
            # A separate reader sees every row so far, as it would after an interrupted run.
            assert len(read_raw_csv(path)) == written

    assert all(stored.keys() == set(RAW_COLUMNS) for stored in writer.rows)
    assert aggregate(read_raw_csv(path), 3, 4) == aggregate(rows, 3, 4)


def test_raw_writer_rejects_unknown_columns():
    handle = io.StringIO()
    writer = RawWriter(handle)

    with pytest.raises(ValueError, match="ttft_ms"):
        writer.write(final_row("a") | {"ttft_ms": 1000})

    assert writer.rows == []
    assert handle.getvalue().splitlines() == [",".join(RAW_COLUMNS)]


def test_metrics_without_samples_are_written_as_empty_cells_not_zero(tmp_path):
    rows = [final_row("a", tps=None), final_row("b", status="error", ttft=None)]
    path = tmp_path / "aggregated.csv"

    write_csv(path, AGGREGATED_COLUMNS, aggregate(rows, 20, 100))

    [cell] = list(csv.DictReader(path.open(newline="")))
    assert cell["successful_samples"] == "1"
    assert cell["failure_count"] == "1"
    assert cell["failure_rate"] == "0.500000"
    assert cell["ttft_s_p50"] == "1.000000"
    assert cell["tps_samples"] == "0"
    assert cell["tps_p50"] == ""
    assert cell["input_tokens_p50"] == ""


def test_aggregate_has_no_cells_without_measured_rows():
    warmup_only = [{**final_row("m/high/warmup", status="error", ttft=None), "phase": "warmup", "turn_index": 1}]

    assert aggregate(warmup_only, 20, 100) == []
    assert aggregate([], 20, 100) == []
