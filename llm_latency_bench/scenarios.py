"""The scenario matrix: every model/reasoning pair x prompt size x context size x turn count x repetition."""

import itertools
import random
from dataclasses import dataclass

from pydantic_ai import ModelMessage, ModelRequest, ModelResponse
from pydantic_ai.messages import SystemPromptPart, UserPromptPart

from llm_latency_bench import fixtures
from llm_latency_bench.models_csv import ModelPair
from llm_latency_bench.settings import SizeSettings


@dataclass(frozen=True)
class Scenario:
    pair: ModelPair
    prompt_size: str
    prompt_tokens: int
    context_size: str
    context_tokens: int
    turn_count: int
    repetition: int

    @property
    def scenario_id(self) -> str:
        return f"{self.pair.key}/p-{self.prompt_size}/c-{self.context_size}/t-{self.turn_count}/r-{self.repetition}"


def build_scenarios(pairs: list[ModelPair], sizes: SizeSettings, repetitions: int, seed: int) -> list[Scenario]:
    """All scenarios in a seeded random order, so time-of-day and warm-up effects are not tied to one model."""
    scenarios = [
        Scenario(pair, prompt_size, prompt_tokens, context_size, context_tokens, turn_count, repetition)
        for pair, (prompt_size, prompt_tokens), (context_size, context_tokens), turn_count, repetition in (
            itertools.product(
                pairs,
                sizes.prompt_tokens.items(),
                sizes.context_tokens.items(),
                sizes.turn_counts,
                range(1, repetitions + 1),
            )
        )
    ]
    random.Random(seed).shuffle(scenarios)
    return scenarios


def expected_requests(scenarios: list[Scenario]) -> int:
    """Requests needed if nothing fails and nothing is retried."""
    return sum(scenario.turn_count for scenario in scenarios)


def estimate_input_tokens(scenarios: list[Scenario], answer_words: int) -> int:
    """Rough total input tokens for the plan, from text length. Each reply is assumed to be `answer_words` long."""
    reply_chars = answer_words * 6
    total_chars = 0
    for scenario in scenarios:
        system, prompts = scenario_texts(scenario, answer_words, nonce="0" * 32)
        history = len(system)
        for prompt in prompts:
            history += len(prompt)
            total_chars += history
            history += reply_chars
    return total_chars // fixtures.CHARS_PER_TOKEN


class Conversation:
    """The message history of one scenario. Only successful replies are ever appended."""

    def __init__(self, system_prompt: str) -> None:
        self._system_prompt = system_prompt
        self._messages: list[ModelMessage] = []

    def with_user_turn(self, prompt: str) -> list[ModelMessage]:
        """Messages to send for the next turn: the history so far plus the new user prompt."""
        if not self._messages:
            return [ModelRequest(parts=[SystemPromptPart(self._system_prompt), UserPromptPart(prompt)])]
        return [*self._messages, ModelRequest(parts=[UserPromptPart(prompt)])]

    def record_turn(self, sent: list[ModelMessage], reply: ModelResponse) -> None:
        self._messages = [*sent, reply]


def scenario_texts(scenario: Scenario, answer_words: int, nonce: str) -> tuple[str, list[str]]:
    """The system prompt (with the reference context) and one user prompt per turn."""
    context = fixtures.reference_context(scenario.context_tokens)
    prompts = fixtures.conversation_prompts(scenario.prompt_tokens, scenario.turn_count, answer_words)
    return fixtures.system_prompt(nonce, context), prompts
