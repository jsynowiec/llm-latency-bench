import asyncio
import dataclasses
import io
import time

import pytest

from llm_latency_bench.credentials import Redactor
from llm_latency_bench.models_csv import ModelPair
from llm_latency_bench.results import RawWriter, aggregate
from llm_latency_bench.runner import BenchmarkRunner, run_matrix
from llm_latency_bench.scenarios import Scenario
from tests.fake_openrouter import FakeOpenRouter, echo_turns, error_reply, generation_id, text_reply

PAIR = ModelPair("GPT-6 Sol", "High", "openai/gpt-6-sol", "high")
OTHER = ModelPair("GPT-6 Luna", "High", "openai/gpt-6-luna", "high")
THIRD = ModelPair("GPT-6 Terra", "High", "openai/gpt-6-terra", "high")


class VirtualClock:
    """perf_counter plus virtual seconds, so retry waits and slow replies cost no real time."""

    def __init__(self) -> None:
        self.offset = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return time.perf_counter() + self.offset

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.offset += seconds


class OverlapTrackingServer(FakeOpenRouter):
    """Holds each request open briefly and records the most requests in flight at once."""

    def __init__(self, script, hold_s: float) -> None:
        super().__init__(script)
        self._hold_s = hold_s
        self._in_flight = 0
        self.peak_in_flight = 0

    async def handle(self, request):
        self._in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self._in_flight)
        try:
            await asyncio.sleep(self._hold_s)
        finally:
            self._in_flight -= 1
        return await super().handle(request)


def run(openrouter_models, settings, server, scenarios, pairs=(PAIR,), warmup=False, concurrency=1, clock=None):
    writer = RawWriter(io.StringIO())
    clock = clock or VirtualClock()
    logs: list[str] = []

    async def go():
        async with openrouter_models(server) as models:
            runner = BenchmarkRunner(
                run_id="run-1",
                seed=5,
                settings=settings,
                model_factory=models,
                writer=writer,
                redact=Redactor(()),
                log=logs.append,
                sleep=clock.sleep,
                clock=clock,
            )
            return await run_matrix(
                runner, list(pairs), scenarios, concurrency=concurrency, warmup=warmup, log=logs.append
            )

    outcome = asyncio.run(go())
    return outcome, writer.rows, clock.sleeps, logs


def with_backoff(settings, base_s: float, max_s: float = 30.0, max_retries: int = 2):
    retries = dataclasses.replace(settings.retries, backoff_base_s=base_s, backoff_max_s=max_s, max_retries=max_retries)
    return dataclasses.replace(settings, retries=retries)


def scenario(turns: int, pair: ModelPair = PAIR, repetition: int = 1) -> Scenario:
    return Scenario(pair, "short", 40, "short", 250, turns, repetition)


def message_texts(body: dict) -> list[tuple[str, str]]:
    return [(message["role"], message["content"]) for message in body["messages"]]


def test_multi_turn_history_carries_the_actual_replies_in_order(openrouter_models, settings):
    server = FakeOpenRouter(echo_turns)
    outcome, rows, _, _ = run(openrouter_models, settings, server, [scenario(3)])

    assert outcome.completed == 1
    third = message_texts(server.bodies[2])
    assert [role for role, _ in third] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert third[2] == ("assistant", "Answer 1.")
    assert third[4] == ("assistant", "Answer 2.")
    # The reference context stays in the system message; user messages are the per-turn prompts.
    assert "<asset_register>" in third[0][1]
    assert all("<asset_register>" not in content for role, content in third if role == "user")
    assert [row["turn_index"] for row in rows] == [1, 2, 3]
    assert [row["is_final_turn"] for row in rows] == [False, False, True]
    assert rows[-1]["scenario_status"] == "completed"
    assert rows[0]["scenario_total_s"] is None


def test_scenario_total_spans_exactly_its_own_turns(openrouter_models, settings):
    clock = VirtualClock()

    def script(body, number):
        # Request N takes 10*N virtual seconds, so each scenario's total is known in advance.
        clock.offset += 10.0 * number
        return text_reply("ok")

    server = FakeOpenRouter(script)
    _, rows, _, _ = run(openrouter_models, settings, server, [scenario(3), scenario(2, repetition=2)], clock=clock)

    assert [row["request_latency_s"] for row in rows] == pytest.approx([10, 20, 30, 40, 50], abs=0.5)
    finals = [row for row in rows if row["is_final_turn"]]
    assert [row["scenario_total_s"] for row in finals] == pytest.approx([60, 90], abs=0.5)
    assert [row["scenario_backoff_s"] for row in finals] == [0.0, 0.0]


def test_retry_after_rate_limit_keeps_the_failed_attempt_visible(openrouter_models, settings):
    server = FakeOpenRouter(lambda body, number: error_reply(429, "slow") if number == 1 else text_reply("ok"))
    scenarios = [scenario(1), scenario(1, repetition=2)]
    _, rows, sleeps, logs = run(openrouter_models, with_backoff(settings, 2.0), server, scenarios)

    assert [(row["status"], row["retry_index"], row["error_category"]) for row in rows] == [
        ("error", 0, "rate_limited"),
        ("ok", 1, None),
        ("ok", 0, None),
    ]
    [wait] = sleeps
    assert 1.0 <= wait <= 2.0
    assert rows[1]["scenario_backoff_s"] == wait
    assert rows[2]["scenario_backoff_s"] == 0.0
    assert any("retry 1/2" in line for line in logs)

    [cell] = aggregate(rows, 20, 100)
    assert cell["scenario_runs"] == 2
    assert cell["successful_samples"] == 2
    assert cell["failure_count"] == 0
    assert cell["failed_attempts"] == 1
    assert cell["retried_scenarios"] == 1


def test_retries_stop_at_the_configured_limit_and_sum_their_waits(openrouter_models, settings):
    server = FakeOpenRouter(lambda body, number: error_reply(503, "down"))
    _, rows, sleeps, _ = run(openrouter_models, with_backoff(settings, 1.0), server, [scenario(1)])

    assert [row["retry_index"] for row in rows] == [0, 1, 2]
    assert len(sleeps) == 2
    assert rows[-1]["scenario_status"] == "aborted"
    assert rows[-1]["scenario_backoff_s"] == pytest.approx(sleeps[0] + sleeps[1])
    assert rows[-1]["scenario_total_s"] >= rows[-1]["scenario_backoff_s"]


def test_backoff_doubles_per_retry_up_to_the_cap_with_jitter(openrouter_models, settings):
    server = FakeOpenRouter(lambda body, number: error_reply(503, "down"))
    settings = with_backoff(settings, 1.0, max_s=3.0, max_retries=8)
    _, _, sleeps, _ = run(openrouter_models, settings, server, [scenario(1)])

    # Uncapped delays 1, 2, 4, 8, ... capped at 3, each scaled by a jitter factor in [0.5, 1.0].
    bounds = [(0.5, 1.0), (1.0, 2.0), *[(1.5, 3.0)] * 6]
    assert len(sleeps) == len(bounds)
    for wait, (low, high) in zip(sleeps, bounds, strict=True):
        assert low <= wait <= high
    assert len(set(sleeps[2:])) == 6


def test_scenario_backoff_and_total_include_waits_from_every_turn(openrouter_models, settings):
    clock = VirtualClock()
    failures = {1: error_reply(429, "slow"), 4: error_reply(503, "down")}

    def script(body, number):
        clock.offset += 1.0
        return failures.get(number) or text_reply("ok")

    server = FakeOpenRouter(script)
    _, rows, sleeps, _ = run(openrouter_models, with_backoff(settings, 4.0), server, [scenario(3)], clock=clock)

    assert [(row["turn_index"], row["retry_index"]) for row in rows] == [(1, 0), (1, 1), (2, 0), (3, 0), (3, 1)]
    assert len(sleeps) == 2
    assert all(2.0 <= wait <= 4.0 for wait in sleeps)
    final = rows[-1]
    assert final["scenario_backoff_s"] == pytest.approx(sleeps[0] + sleeps[1])
    # Five requests of 1 virtual second each, plus both waits.
    assert final["scenario_total_s"] == pytest.approx(5.0 + sleeps[0] + sleeps[1], abs=0.5)
    assert all(row["scenario_backoff_s"] is None for row in rows[:-1])


def test_request_seq_follows_send_order_across_retries(openrouter_models, settings):
    failures = {2: error_reply(429, "slow"), 4: error_reply(503, "down"), 5: error_reply(503, "down")}
    server = FakeOpenRouter(lambda body, number: failures.get(number) or text_reply("ok"))
    _, rows, _, _ = run(openrouter_models, settings, server, [scenario(2)], warmup=True)

    assert [(row["phase"], row["turn_index"], row["retry_index"]) for row in rows] == [
        ("warmup", 1, 0),
        ("measure", 1, 0),
        ("measure", 1, 1),
        ("measure", 2, 0),
        ("measure", 2, 1),
        ("measure", 2, 2),
    ]
    assert [row["request_seq"] for row in rows] == [1, 2, 3, 4, 5, 6]


def test_non_retryable_failure_aborts_the_scenario_and_sends_no_later_turns(openrouter_models, settings):
    server = FakeOpenRouter(lambda body, number: error_reply(400, "bad") if number == 2 else text_reply("fine"))
    outcome, rows, sleeps, _ = run(openrouter_models, settings, server, [scenario(5)])

    assert outcome.aborted == 1
    assert len(server.requests) == 2
    assert sleeps == []
    assert rows[-1]["turn_index"] == 2
    assert rows[-1]["scenario_status"] == "aborted"

    [cell] = aggregate(rows, 20, 100)
    assert cell["scenario_runs"] == 1
    assert cell["successful_samples"] == 0
    assert cell["failure_count"] == 1
    assert cell["failure_rate"] == 1.0
    assert cell["ttft_s_p50"] is None


@pytest.mark.parametrize(
    ("status", "category"), [(400, "bad_request"), (403, "forbidden"), (404, "not_found"), (422, "bad_request")]
)
def test_pair_rejected_at_warmup_is_recorded_and_not_benchmarked(openrouter_models, settings, status, category):
    def script(body, number):
        if body["model"] == "openai/gpt-6-luna":
            return error_reply(status, "this model or setting is not available")
        return text_reply("ok")

    server = FakeOpenRouter(script)
    scenarios = [scenario(1, OTHER), scenario(1, PAIR), scenario(5, OTHER)]
    outcome, rows, _, _ = run(openrouter_models, settings, server, scenarios, pairs=(PAIR, OTHER), warmup=True)

    assert outcome.rejected_pairs == ["openai/gpt-6-luna/high"]
    luna_requests = [body for body in server.bodies if body["model"] == "openai/gpt-6-luna"]
    assert len(luna_requests) == 1
    luna_rows = [row for row in rows if row["model_slug"] == "openai/gpt-6-luna"]
    assert [(row["phase"], row["status"], row["error_category"]) for row in luna_rows] == [
        ("warmup", "error", category),
        ("measure", "not_sent", "pair_rejected"),
        ("measure", "not_sent", "pair_rejected"),
    ]
    cells = {cell["turn_count"]: cell for cell in aggregate(rows, 20, 100) if cell["model_slug"] == "openai/gpt-6-luna"}
    assert cells[5]["failure_count"] == 1


@pytest.mark.parametrize(("status", "category"), [(401, "auth"), (402, "insufficient_credits")])
def test_account_failure_stops_the_run_and_marks_the_rest_not_sent(openrouter_models, settings, status, category):
    server = FakeOpenRouter(lambda body, number: error_reply(status, "account problem"))
    scenarios = [scenario(1), scenario(1, repetition=2), scenario(1, repetition=3)]
    outcome, rows, _, _ = run(openrouter_models, settings, server, scenarios)

    assert outcome.fatal_error is not None
    assert category in outcome.fatal_error
    assert len(server.requests) == 1
    assert [row["status"] for row in rows] == ["error", "not_sent", "not_sent"]
    assert [row["error_category"] for row in rows[1:]] == ["run_aborted", "run_aborted"]

    # Rows never sent are failed scenarios but not failed attempts.
    [cell] = aggregate(rows, 20, 100)
    assert cell["failure_count"] == 3
    assert cell["failed_attempts"] == 1


def test_account_failure_at_warmup_skips_the_remaining_warmups(openrouter_models, settings):
    server = FakeOpenRouter(lambda body, number: error_reply(401, "bad key"))
    pairs = (PAIR, OTHER, THIRD)
    scenarios = [scenario(1, pair) for pair in pairs]
    outcome, rows, _, _ = run(openrouter_models, settings, server, scenarios, pairs=pairs, warmup=True)

    assert len(server.requests) == 1
    assert outcome.rejected_pairs == []
    assert "auth" in outcome.fatal_error
    assert [(row["phase"], row["status"], row["error_category"]) for row in rows] == [
        ("warmup", "error", "auth"),
        ("measure", "not_sent", "run_aborted"),
        ("measure", "not_sent", "run_aborted"),
        ("measure", "not_sent", "run_aborted"),
    ]


def test_every_turn_sends_the_reasoning_object_of_its_pair(openrouter_models, settings):
    server = FakeOpenRouter(echo_turns)
    run(openrouter_models, settings, server, [scenario(2)])

    assert [body["reasoning"] for body in server.bodies] == [{"effort": "high", "exclude": True}] * 2


def test_aggregate_uses_only_final_turns_and_groups_by_cell(openrouter_models, settings):
    clock = VirtualClock()
    # Virtual seconds before the first token, by request number. Requests 4 and 6 are non-final turns.
    delays = {1: 1.0, 2: 2.0, 3: 4.0, 4: 50.0, 5: 3.0, 6: 60.0, 7: 7.0}

    def script(body, number):
        clock.offset += delays[number]
        return text_reply("ok")

    server = FakeOpenRouter(script)
    scenarios = [*(scenario(1, repetition=rep) for rep in (1, 2, 3)), *(scenario(2, repetition=rep) for rep in (1, 2))]
    _, rows, _, _ = run(openrouter_models, settings, server, scenarios, clock=clock)

    cells = {cell["turn_count"]: cell for cell in aggregate(rows, 2, 3)}
    assert (cells[1]["scenario_runs"], cells[1]["successful_samples"]) == (3, 3)
    assert (cells[2]["scenario_runs"], cells[2]["successful_samples"]) == (2, 2)
    assert cells[1]["ttft_s_p50"] == pytest.approx(2.0, abs=0.05)
    assert cells[2]["ttft_s_p50"] == pytest.approx(5.0, abs=0.05)
    assert cells[2]["ttft_s_p10"] == pytest.approx(3.4, abs=0.05)
    assert cells[2]["scenario_total_s_p50"] == pytest.approx(60.0, abs=0.1)
    assert (cells[1]["p95_low_confidence"], cells[1]["p99_low_confidence"]) == (False, False)
    assert (cells[2]["p95_low_confidence"], cells[2]["p99_low_confidence"]) == (False, True)


def test_unclassified_http_error_at_warmup_does_not_reject_the_pair(openrouter_models, settings):
    server = FakeOpenRouter(lambda body, number: error_reply(409, "conflict") if number == 1 else text_reply("ok"))
    outcome, rows, _, _ = run(openrouter_models, settings, server, [scenario(1)], warmup=True)

    assert outcome.rejected_pairs == []
    assert [(row["phase"], row["status"]) for row in rows] == [("warmup", "error"), ("measure", "ok")]


def nonce_of(body: dict) -> str:
    return body["messages"][0]["content"].split(".")[0].removeprefix("Session ")


@pytest.mark.parametrize("concurrency", [1, 2, 3])
def test_concurrency_runs_that_many_scenarios_at_once_and_no_more(openrouter_models, settings, concurrency):
    server = OverlapTrackingServer(echo_turns, hold_s=0.02)
    scenarios = [scenario(1, repetition=rep) for rep in range(1, 6)]
    outcome, _, _, _ = run(openrouter_models, settings, server, scenarios, concurrency=concurrency)

    assert outcome.completed == 5
    assert server.peak_in_flight == concurrency


def test_concurrent_scenarios_keep_their_own_history_and_rows(openrouter_models, settings):
    def script(body, number):
        # Replies name the scenario (by nonce) and turn and arrive with uneven delays,
        # so a history or row mixed up between overlapping scenarios is detectable.
        nonce = nonce_of(body)
        turn = sum(message["role"] == "user" for message in body["messages"])
        reply = text_reply(f"{nonce} turn {turn}")
        reply.chunks[1] = (0.02 * (number % 3), reply.chunks[1][1])
        return reply

    server = FakeOpenRouter(script)
    scenarios = [scenario(5), scenario(5, repetition=2), scenario(1)]
    outcome, rows, _, _ = run(openrouter_models, settings, server, scenarios, concurrency=2)

    assert outcome.completed == 3
    assert len(rows) == 11
    for body in server.bodies:
        assistant = [message["content"] for message in body["messages"] if message["role"] == "assistant"]
        assert assistant == [f"{nonce_of(body)} turn {turn}" for turn in range(1, len(assistant) + 1)]
    nonce_by_generation = {generation_id(number): nonce_of(body) for number, body in enumerate(server.bodies, 1)}
    nonce_by_scenario: dict[str, set[str]] = {}
    for row in rows:
        nonce_by_scenario.setdefault(row["scenario_id"], set()).add(nonce_by_generation[row["generation_id"]])
    assert all(len(nonces) == 1 for nonces in nonce_by_scenario.values())
    assert len(set().union(*nonce_by_scenario.values())) == 3
