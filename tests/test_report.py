import re

from llm_latency_bench.report import summarize

TABLE_TITLES = (
    "Final-turn distribution",
    "TTFT p50 by factor",
    "TTFT p50 (s) and hidden reasoning",
    "Input tokens and cache reads",
    "Control: TTFT p50 of turn 1",
    "Warm-up",
    "Failure rate per pair",
)


def attempt(pair: str, scenario_id: str, **columns) -> dict:
    """A successful single-turn attempt of `pair` ("<model_slug>/<reasoning_level>"), with `columns` overridden."""
    model_slug, reasoning_level = pair.split("/")
    return {
        "phase": "measure",
        "run_id": "run-1",
        "scenario_id": f"{pair}/{scenario_id}",
        "model_slug": model_slug,
        "reasoning_level": reasoning_level,
        "prompt_size": "short",
        "prompt_target_tokens": 40,
        "context_size": "none",
        "context_target_tokens": 0,
        "turn_count": 1,
        "turn_index": 1,
        "is_final_turn": True,
        "retry_index": 0,
        "status": "ok",
        "error_category": None,
        "upstream_provider": "Azure",
        "ttft_s": None,
        "request_latency_s": None,
        "tps": None,
    } | columns


def table(text: str, title: str) -> tuple[list[str], dict[str, list[str]]]:
    """The header and the body rows, keyed by pair, of the one summary table whose title starts with `title`."""
    lines = text.splitlines()
    [start] = [index for index, line in enumerate(lines) if line.startswith(f"== {title}")]
    header = re.split(r" {2,}", lines[start + 1].strip())
    body: dict[str, list[str]] = {}
    for line in lines[start + 3 :]:
        cells = re.split(r" {2,}", line.strip())
        if not line or len(cells) != len(header):  # the blank line before the next table, or a note
            break
        body[cells[0]] = cells
    return header, body


def test_distribution_table_has_percentiles_and_tukey_outliers_of_final_turn_samples():
    a_samples = [  # (TTFT, latency, TPS)
        (30.0, 35.0, None),
        (1.0, 2.0, 50.0),
        (5.0, 9.0, 100.0),
        (40.0, 50.0, None),
        (3.0, 5.0, 70.0),
        (2.0, 3.0, 60.0),
        (6.0, 10.0, 130.0),
        (4.0, 8.0, 80.0),
    ]
    b_samples = [(7.0, 8.0), (1.0, 2.0), (4.0, 5.0), (2.0, 3.0), (3.0, 4.0)]
    rows = [
        *(
            attempt("a/high", f"a{i}", ttft_s=t, request_latency_s=lat, tps=tps)
            for i, (t, lat, tps) in enumerate(a_samples)
        ),
        *(attempt("b/low", f"b{i}", ttft_s=t, request_latency_s=lat) for i, (t, lat) in enumerate(b_samples)),
        # Not samples: a truncated final turn, a turn before the final one, and a warm-up.
        attempt("a/high", "a-truncated", status="error", ttft_s=0.2, request_latency_s=0.3),
        attempt("a/high", "a-aborted", turn_count=2, is_final_turn=False, ttft_s=0.1, request_latency_s=0.2),
        attempt("a/high", "warmup", phase="warmup", ttft_s=0.3, request_latency_s=0.4),
    ]

    text = summarize(rows, 7)

    header, body = table(text, "Final-turn distribution")
    assert header == [
        "pair",
        "n",
        "TTFT p10",
        "TTFT p50",
        "TTFT p95",
        "latency p50",
        "latency p95",
        "TPS p50",
        "TTFT outliers",
    ]
    # a: fence = 12 + 1.5 x (12 - 2.75) = 25.875. b: fence = 4 + 1.5 x 2 = 7, and 7 is not above it.
    assert body == {
        "a/high": ["a/high", "8", "1.70", "4.50", "36.50", "8.50", "44.75", "75.0", "2"],
        "b/low": ["b/low", "5", "1.40", "3.00", "6.40", "4.00", "7.40", "-", "0"],
    }
    assert "p95 over fewer than 7 samples" in text


def factor_rows() -> list[dict]:
    """Final-turn samples over prompt x context x turn count, plus the earlier turns of the three-turn scenarios."""
    cells = [  # (prompt, context, turns, TTFT, reasoning, input, cache read, output tokens)
        ("short", "none", 1, 1.0, 100, 50, 0, 300),
        ("short", "large", 1, 5.0, 400, 20050, 1024, 500),
        ("short", "large", 3, 6.0, 600, 20300, 20000, 700),
        ("long", "none", 1, 2.0, 200, 2010, 512, 400),
        ("long", "none", 3, 3.0, 300, 2400, 2200, 450),
        ("long", "large", 3, 10.0, 900, 22500, 22000, 900),
        ("short", "none", 3, 4.0, 250, 300, 100, 350),
    ]
    target_tokens = {"short": 40, "long": 2000, "none": 0, "large": 20000}
    rows = []
    for index, (prompt, context, turns, ttft, reasoning, input_tokens, cached, output) in enumerate(cells):
        scenario = {
            "prompt_size": prompt,
            "prompt_target_tokens": target_tokens[prompt],
            "context_size": context,
            "context_target_tokens": target_tokens[context],
            "turn_count": turns,
        }
        for turn in range(1, turns):
            rows.append(
                attempt(
                    "a/high",
                    f"a{index}",
                    **scenario,
                    turn_index=turn,
                    is_final_turn=False,
                    ttft_s=0.1,
                    reasoning_tokens=5,
                    input_tokens=100,
                    cache_read_tokens=0,
                    output_tokens=10,
                )
            )
        rows.append(
            attempt(
                "a/high",
                f"a{index}",
                **scenario,
                turn_index=turns,
                ttft_s=ttft,
                reasoning_tokens=reasoning,
                input_tokens=input_tokens,
                cache_read_tokens=cached,
                output_tokens=output,
            )
        )
    rows.append(
        attempt("b/low", "b0", ttft_s=0.4, reasoning_tokens=50, input_tokens=45, cache_read_tokens=0, output_tokens=120)
    )
    return rows


def test_ttft_by_factor_pools_over_the_other_factors_and_orders_sizes_by_target_tokens():
    header, body = table(summarize(factor_rows(), 20), "TTFT p50 by factor")

    assert header == ["pair", "prompt short", "prompt long", "context none", "context large", "turns 1", "turns 3"]
    assert body == {
        "a/high": ["a/high", "4.50", "3.00", "2.50", "6.00", "2.00", "5.00"],
        "b/low": ["b/low", "0.40", "-", "0.40", "-", "0.40", "-"],
    }


def test_context_scaling_table_has_ttft_and_reasoning_medians_per_context_size():
    header, body = table(summarize(factor_rows(), 20), "TTFT p50 (s) and hidden reasoning")

    assert header == ["pair", "context none TTFT/reasoning", "context large TTFT/reasoning"]
    assert body == {
        "a/high": ["a/high", "2.50/225", "6.00/600"],
        "b/low": ["b/low", "0.40/50", "-/-"],
    }


def test_token_table_has_final_turn_medians_per_turn_count():
    header, body = table(summarize(factor_rows(), 20), "Input tokens and cache reads")

    assert header == ["pair", "turns 1 in/cached", "turns 3 in/cached", "output tokens", "reasoning tokens"]
    assert body == {
        "a/high": ["a/high", "2010/512", "11350/11100", "450", "300"],
        "b/low": ["b/low", "45/0", "-/-", "120", "50"],
    }


def test_first_turn_control_uses_successful_measured_turn_one_attempts():
    def scenario(scenario_id: str, turns: int, first_ttft: float) -> list[dict]:
        first = attempt("a/high", scenario_id, turn_count=turns, is_final_turn=turns == 1, ttft_s=first_ttft)
        if turns == 1:
            return [first]
        final = attempt("a/high", scenario_id, turn_count=turns, turn_index=turns, ttft_s=0.2)
        return [first, final]

    rows = [
        *scenario("one-a", 1, 0.7),
        *scenario("one-b", 1, 0.9),
        *scenario("two-a", 2, 1.0),
        *scenario("two-b", 2, 6.0),
        *scenario("four-a", 4, 11.0),
        *scenario("four-b", 4, 3.0),
        *scenario("four-c", 4, 4.0),
        # A truncated turn 1 has a TTFT but is not a successful attempt. Its retry is.
        attempt("a/high", "two-c", turn_count=2, is_final_turn=False, status="error", ttft_s=50.0),
        attempt("a/high", "two-c", turn_count=2, is_final_turn=False, retry_index=1, ttft_s=2.0),
        attempt("a/high", "two-c", turn_count=2, turn_index=2, ttft_s=0.2),
        attempt("a/high", "warmup", phase="warmup", ttft_s=9.0),
    ]

    text = summarize(rows, 20)

    header, body = table(text, "Control: TTFT p50 of turn 1")
    assert header == ["pair", "turns 2", "turns 4"]
    assert body == {"a/high": ["a/high", "2.00", "4.00"]}
    # Turn 1 pooled over every turn count: 0.7, 0.9, 1, 2, 3, 4, 6, 11.
    _, warmup = table(text, "Warm-up")
    assert warmup == {"a/high": ["a/high", "ok=1", "9.00", "2.50"]}


def test_failure_table_and_totals_count_scenario_runs_across_combined_runs():
    rows = [
        attempt("a/high", "s1"),
        attempt("a/high", "s2", turn_count=2, is_final_turn=False, status="error", error_category="timeout"),
        attempt("a/high", "s3", status="not_sent", retry_index=None, error_category="run_aborted"),
        attempt("a/high", "s4", status="error", error_category="timeout"),
        attempt("a/high", "s4", status="error", error_category="timeout", retry_index=1),
        attempt("a/high", "s4", retry_index=2),
        attempt("a/high", "s1", run_id="run-2"),
        attempt("b/low", "s1"),
        attempt("b/low", "s2"),
        attempt("b/low", "s3", status="error", error_category="timeout"),
    ]
    for row in rows:
        if row["status"] != "ok":
            row["upstream_provider"] = None

    text = summarize(rows, 20)

    _, body = table(text, "Failure rate per pair")
    assert body == {
        "a/high": ["a/high", "5", "2", "0.40"],
        "b/low": ["b/low", "3", "1", "0.33"],
    }
    assert text.splitlines()[1:4] == [
        "scenario runs: 8, valid final-turn samples: 5, requests sent: 9 (incl. warm-up and retries), "
        "failed attempts: 4, retries: 2, not sent: 1",
        "upstream providers: Azure=5, not reported=4",
        "errors by category: timeout=4, run_aborted=1",
    ]


def test_summary_of_runs_without_samples_has_totals_and_empty_tables():
    warmup_only = [
        attempt("m/high", "warmup", phase="warmup", status="error", error_category="timeout", upstream_provider=None)
    ]

    text = summarize(warmup_only, 20)

    assert text.splitlines()[1:4] == [
        "scenario runs: 0, valid final-turn samples: 0, requests sent: 1 (incl. warm-up and retries), "
        "failed attempts: 1, retries: 0, not sent: 0",
        "upstream providers: not reported=1",
        "errors by category: timeout=1",
    ]
    assert table(text, "Warm-up")[1] == {"m/high": ["m/high", "error=1", "-", "-"]}
    for title in TABLE_TITLES:
        if title != "Warm-up":
            assert table(text, title)[1] == {}

    empty = summarize([], 20)
    assert empty.splitlines()[:3] == [
        "== Run totals ==",
        "scenario runs: 0, valid final-turn samples: 0, requests sent: 0 (incl. warm-up and retries), "
        "failed attempts: 0, retries: 0, not sent: 0",
        "upstream providers: ",
    ]
    for title in TABLE_TITLES:
        assert table(empty, title)[1] == {}


def test_warmup_table_counts_the_final_warmup_of_every_combined_run():
    def warmup(run_id: str, status: str, ttft: float | None, retry_index: int = 0) -> dict:
        return attempt(
            "m/high",
            "warmup",
            run_id=run_id,
            phase="warmup",
            status=status,
            ttft_s=ttft,
            retry_index=retry_index,
        )

    rows = [
        warmup("run-a", "error", None),
        warmup("run-b", "error", None),
        warmup("run-b", "ok", 3.0, retry_index=1),
        warmup("run-c", "ok", 5.0),
    ]

    _, body = table(summarize(rows, 20), "Warm-up")
    # run-b retried, so only its last attempt counts: one failed run and two successful ones.
    assert body == {"m/high": ["m/high", "error=1, ok=2", "4.00", "-"]}
