"""Run the scenario matrix: warm-up, scenarios, retries, and one raw row per attempt."""

import asyncio
import itertools
import random
import time
import uuid
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import cast

import logfire
from pydantic_ai import ModelMessage, ModelRequest
from pydantic_ai.models.openrouter import OpenRouterModelSettings, OpenRouterReasoning

from llm_latency_bench.credentials import Redactor
from llm_latency_bench.measure import Attempt, ErrorCategory, stream_attempt
from llm_latency_bench.models_csv import ModelPair
from llm_latency_bench.openrouter import ModelFactory
from llm_latency_bench.results import RawWriter, Row, attempt_columns, pair_columns, scenario_columns
from llm_latency_bench.scenarios import Conversation, Scenario, scenario_texts
from llm_latency_bench.settings import Settings

WARMUP_PROMPT = "Reply with the single word: ready."
# Warm-up errors that mean the model or setting was rejected. Such a pair is not benchmarked.
PAIR_REJECTED = frozenset({ErrorCategory.BAD_REQUEST, ErrorCategory.NOT_FOUND, ErrorCategory.FORBIDDEN})
# Errors that affect every pair, so the run stops.
RUN_FATAL = frozenset({ErrorCategory.AUTH, ErrorCategory.INSUFFICIENT_CREDITS})


@dataclass
class TurnOutcome:
    """The last attempt of a turn and its row, not yet written, so scenario totals can be added to it."""

    attempt: Attempt
    row: Row
    backoff_s: float


@dataclass
class RunOutcome:
    completed: int = 0
    aborted: int = 0
    not_sent: int = 0
    rejected_pairs: list[str] = field(default_factory=list)
    fatal_error: str | None = None


class BenchmarkRunner:
    def __init__(
        self,
        *,
        run_id: str,
        seed: int,
        settings: Settings,
        model_factory: ModelFactory,
        writer: RawWriter,
        redact: Redactor,
        log: Callable[[str], None] = print,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._run_id = run_id
        self._seed = seed
        self._settings = settings
        self._model_factory = model_factory
        self._writer = writer
        self._redact = redact
        self._log = log
        self._sleep = sleep
        self._clock = clock
        self._request_seq = itertools.count(1)
        self._jitter = random.Random(seed)
        self._fatal: Attempt | None = None

    def _model_settings(self, pair: ModelPair) -> OpenRouterModelSettings:
        return OpenRouterModelSettings(
            openrouter_reasoning=cast(
                OpenRouterReasoning,
                # Sent verbatim, so the provider decides which efforts a model accepts and rejections are recorded.
                # `exclude` keeps reasoning text out of the stream and the history.
                {"effort": pair.reasoning_level, "exclude": True},
            ),
            max_tokens=self._settings.output.max_tokens,
        )

    def _backoff_s(self, retry_index: int) -> float:
        retry = self._settings.retries
        delay = min(retry.backoff_max_s, retry.backoff_base_s * 2**retry_index)
        return delay * self._jitter.uniform(0.5, 1.0)

    def _common(self, row: Row) -> Row:
        return {"run_id": self._run_id, "seed": self._seed} | row

    async def _send_turn(self, pair: ModelPair, messages: list[ModelMessage], base_row: Row) -> TurnOutcome:
        """Send one turn, retrying retryable failures. Earlier attempts are written; the last one is returned."""
        backoff_s = 0.0
        max_retries = self._settings.retries.max_retries
        for retry_index in itertools.count():
            request_seq = next(self._request_seq)
            with logfire.span(
                "attempt {scenario_id} turn {turn_index} retry {retry_index}",
                scenario_id=base_row["scenario_id"],
                turn_index=base_row["turn_index"],
                retry_index=retry_index,
                request_seq=request_seq,
            ) as span:
                attempt = await stream_attempt(
                    self._model_factory(pair.model_slug),
                    messages,
                    self._model_settings(pair),
                    self._settings.client.request_timeout_s,
                    self._redact,
                    self._clock,
                )
                row = base_row | {"request_seq": request_seq} | attempt_columns(attempt, retry_index)
                span.set_attributes(
                    {
                        key: row[key]
                        for key in ("status", "error_category", "http_status", "ttft_s", "request_latency_s")
                        if row[key] is not None
                    }
                )
            if attempt.ok or not attempt.retryable or retry_index >= max_retries:
                if attempt.error_category in RUN_FATAL:
                    self._fatal = attempt
                return TurnOutcome(attempt, row, backoff_s)

            self._writer.write(row)
            delay = self._backoff_s(retry_index)
            self._log(
                f"  retry {retry_index + 1}/{max_retries} for {base_row['scenario_id']} turn {base_row['turn_index']}"
                f" after {attempt.error_category} ({attempt.error_message}); waiting {delay:.1f}s"
            )
            backoff_s += delay
            await self._sleep(delay)
        raise AssertionError("unreachable")

    async def warmup(self, pair: ModelPair) -> bool:
        """One small request per pair. Returns False if the pair was rejected and must not be benchmarked."""
        base_row = self._common(
            pair_columns(pair)
            | {
                "phase": "warmup",
                "scenario_id": f"{pair.key}/warmup",
                "turn_count": 1,
                "turn_index": 1,
                "is_final_turn": True,
            }
        )
        outcome = await self._send_turn(pair, [ModelRequest.user_text_prompt(WARMUP_PROMPT)], base_row)
        self._writer.write(outcome.row)
        attempt = outcome.attempt
        ttft = f"TTFT {attempt.ttft_s:.2f}s" if attempt.ttft_s is not None else "no visible text"
        status = f"ok, {ttft}" if attempt.ok else f"FAILED {attempt.error_category}: {attempt.error_message}"
        self._log(f"warm-up {pair.key}: {status}")
        return attempt.error_category not in PAIR_REJECTED

    def write_not_sent(self, scenario: Scenario, order: int, reason: str) -> None:
        """One visible row for a planned scenario that was never sent."""
        now = datetime.now(UTC).isoformat()
        self._writer.write(
            self._common(scenario_columns(scenario, order))
            | {
                "turn_index": scenario.turn_count,
                "is_final_turn": True,
                "status": "not_sent",
                "error_category": reason,
                "scenario_status": "not_sent",
                "scenario_started_at_utc": now,
                "scenario_ended_at_utc": now,
            }
        )

    async def run_scenario(self, scenario: Scenario, order: int) -> TurnOutcome:
        """Run the turns of one scenario in order. Stops at the first turn that fails after retries."""
        system, prompts = scenario_texts(scenario, self._settings.output.answer_words, uuid.uuid4().hex)
        conversation = Conversation(system)
        base = self._common(scenario_columns(scenario, order))
        started_at = datetime.now(UTC)
        start = self._clock()
        backoff_s = 0.0
        with logfire.span("scenario {scenario_id}", scenario_id=scenario.scenario_id, scenario_order=order):
            for turn_index, prompt in enumerate(prompts, start=1):
                messages = conversation.with_user_turn(prompt)
                is_final_turn = turn_index == scenario.turn_count
                outcome = await self._send_turn(
                    scenario.pair, messages, base | {"turn_index": turn_index, "is_final_turn": is_final_turn}
                )
                backoff_s += outcome.backoff_s
                if is_final_turn or not outcome.attempt.ok:
                    outcome.row |= {
                        "scenario_status": "completed" if outcome.attempt.ok else "aborted",
                        "scenario_total_s": self._clock() - start,
                        "scenario_backoff_s": backoff_s,
                        "scenario_started_at_utc": started_at.isoformat(),
                        "scenario_ended_at_utc": datetime.now(UTC).isoformat(),
                    }
                    self._writer.write(outcome.row)
                    return outcome
                self._writer.write(outcome.row)
                assert outcome.attempt.response is not None
                conversation.record_turn(messages, outcome.attempt.response)
        raise AssertionError("a scenario has at least one turn")

    @property
    def fatal_error(self) -> str | None:
        if self._fatal is None:
            return None
        return f"{self._fatal.error_category} (HTTP {self._fatal.http_status}): {self._fatal.error_message}"


def _progress_line(done: int, total: int, scenario: Scenario, outcome: TurnOutcome) -> str:
    attempt = outcome.attempt
    prefix = f"[{done}/{total}] {scenario.scenario_id}"
    if not attempt.ok:
        return f"{prefix} ABORTED at turn {outcome.row['turn_index']}: {attempt.error_category} {attempt.error_message}"
    ttft = f"{attempt.ttft_s:.2f}s" if attempt.ttft_s is not None else "-"
    return f"{prefix} ok, final-turn TTFT {ttft}, scenario {outcome.row['scenario_total_s']:.1f}s"


async def run_matrix(
    runner: BenchmarkRunner,
    pairs: list[ModelPair],
    scenarios: list[Scenario],
    *,
    concurrency: int,
    warmup: bool,
    log: Callable[[str], None] = print,
) -> RunOutcome:
    outcome = RunOutcome()
    if warmup:
        for pair in pairs:
            if not await runner.warmup(pair):
                outcome.rejected_pairs.append(pair.key)
            if runner.fatal_error:
                break

    total = len(scenarios)
    queue: Iterator[tuple[int, Scenario]] = iter(enumerate(scenarios, start=1))

    async def worker() -> None:
        for order, scenario in queue:
            if runner.fatal_error:
                runner.write_not_sent(scenario, order, "run_aborted")
                outcome.not_sent += 1
            elif scenario.pair.key in outcome.rejected_pairs:
                runner.write_not_sent(scenario, order, "pair_rejected")
                outcome.not_sent += 1
            else:
                result = await runner.run_scenario(scenario, order)
                if result.attempt.ok:
                    outcome.completed += 1
                else:
                    outcome.aborted += 1
                log(_progress_line(outcome.completed + outcome.aborted + outcome.not_sent, total, scenario, result))

    # Workers share one iterator: scenarios start in seeded order, at most `concurrency` at a time.
    async with asyncio.TaskGroup() as group:
        for _ in range(concurrency):
            group.create_task(worker())
    outcome.fatal_error = runner.fatal_error
    return outcome
