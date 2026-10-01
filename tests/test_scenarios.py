import subprocess
import sys
from collections import Counter

import pytest

from llm_latency_bench import fixtures
from llm_latency_bench.models_csv import ModelPair
from llm_latency_bench.scenarios import build_scenarios, expected_requests, scenario_texts
from llm_latency_bench.settings import load_settings
from tests.conftest import REPO_ROOT

PAIRS = [ModelPair("A", "Low", "a", "low"), ModelPair("B", "High", "b", "high")]


def test_every_pair_gets_the_full_factorial_once_per_repetition(settings):
    scenarios = build_scenarios(PAIRS, settings.sizes, repetitions=2, seed=1)

    assert len(scenarios) == 2 * 3 * 3 * 3 * 2
    assert len({scenario.scenario_id for scenario in scenarios}) == len(scenarios)
    per_pair = Counter(scenario.pair.key for scenario in scenarios)
    assert per_pair == {"a/low": 54, "b/high": 54}
    combos = Counter((s.prompt_size, s.context_size, s.turn_count) for s in scenarios if s.pair.key == "a/low")
    assert set(combos.values()) == {2}


def test_expected_requests_count_every_turn(settings):
    scenarios = build_scenarios(PAIRS, settings.sizes, repetitions=1, seed=1)

    # Per pair: 9 prompt/context cells x (1 + 5 + 10) turns.
    assert expected_requests(scenarios) == 2 * 9 * 16


def test_order_is_reproducible_from_the_seed_and_mixes_models(settings):
    first = [s.scenario_id for s in build_scenarios(PAIRS, settings.sizes, repetitions=1, seed=7)]
    again = [s.scenario_id for s in build_scenarios(PAIRS, settings.sizes, repetitions=1, seed=7)]
    other = [s.scenario_id for s in build_scenarios(PAIRS, settings.sizes, repetitions=1, seed=8)]

    assert first == again
    assert first != other
    # Not grouped by model: the first half is not all one pair.
    assert len({scenario_id.split("/p-")[0] for scenario_id in first[: len(first) // 2]}) == 2


def test_context_and_prompt_sizes_grow_and_stay_separate(settings):
    scenarios = {
        (s.prompt_size, s.context_size): s
        for s in build_scenarios(PAIRS[:1], settings.sizes, repetitions=1, seed=1)
        if s.turn_count == 5
    }
    system_short, prompts_short = scenario_texts(scenarios["short", "short"], 80, "n1")
    system_long, prompts_long = scenario_texts(scenarios["long", "long"], 80, "n1")

    assert len(system_short) < len(system_long)
    assert len(prompts_short[0]) < len(prompts_long[0])
    assert len(prompts_long) == 5
    # One distinct question per turn, and no user prompt text leaks into the reference context.
    assert len(set(prompts_long)) == 5
    assert all(prompt[-60:] not in system_long for prompt in prompts_long)


def test_final_turn_sends_the_same_prompt_for_every_turn_count(settings):
    scenarios = build_scenarios(PAIRS[:1], settings.sizes, repetitions=1, seed=1)
    prompts = {
        (s.prompt_size, s.turn_count): scenario_texts(s, 80, "n")[1] for s in scenarios if s.context_size == "short"
    }

    for size in settings.sizes.prompt_tokens:
        one, five, ten = prompts[size, 1], prompts[size, 5], prompts[size, 10]
        # Only the history before the final prompt differs between turn counts.
        assert one[-1] == five[-1] == ten[-1]
        assert five[:4] == ten[:4]
        assert one[-1] not in ten[:-1]
        assert len(set(ten)) == 10


def test_turn_counts_beyond_the_question_set_are_rejected():
    with pytest.raises(ValueError, match="between 1 and 10"):
        fixtures.conversation_prompts(40, 11, 80)


def test_size_targets_are_approximately_met(settings):
    for context_tokens in (*settings.sizes.context_tokens.values(), 128000):
        context = fixtures.reference_context(context_tokens)
        assert 0.8 <= len(context) / (context_tokens * fixtures.CHARS_PER_TOKEN) <= 1.2
    for prompt_tokens in settings.sizes.prompt_tokens.values():
        prompt = fixtures.user_prompt(prompt_tokens, 0, 80)
        assert 0.6 <= len(prompt) / (prompt_tokens * fixtures.CHARS_PER_TOKEN) <= 1.3


def test_fixture_text_is_identical_in_every_process():
    # A fresh interpreter with another hash seed must build the same text, so every run sends the same prompts.
    expression = "fixtures.reference_context(2500) + fixtures.user_prompt(2000, 3, 80)"
    code = f"from llm_latency_bench import fixtures; print({expression}, end='')"
    child = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, env={"PYTHONHASHSEED": "12345"}
    )

    assert child.stdout == fixtures.reference_context(2500) + fixtures.user_prompt(2000, 3, 80)


def test_nonce_leads_the_system_prompt(settings):
    system, _ = scenario_texts(build_scenarios(PAIRS, settings.sizes, 1, 1)[0], 80, "abc123")

    assert system.startswith("Session abc123.")


def test_sweep_with_two_context_levels_plans_only_those_levels():
    sizes = load_settings(REPO_ROOT / "settings.context-sweep-32k-64k.toml").sizes
    scenarios = build_scenarios(PAIRS[:1], sizes, repetitions=5, seed=1)

    assert len(scenarios) == 3 * 2 * 1 * 5
    assert {(s.context_size, s.context_tokens) for s in scenarios} == {("32k", 32000), ("64k", 64000)}
    assert {(s.prompt_size, s.prompt_tokens) for s in scenarios} == {("short", 40), ("medium", 400), ("long", 2000)}
