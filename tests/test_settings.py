from pathlib import Path

import pytest

from llm_latency_bench.settings import SettingsError, load_settings
from tests.conftest import REPO_ROOT

SWEEP_FILES = ["settings.context-sweep.toml", "settings.context-sweep-32k-64k.toml"]


def settings_with(tmp_path: Path, old: str, new: str) -> Path:
    text = (REPO_ROOT / "settings.toml").read_text(encoding="utf-8")
    assert old in text
    path = tmp_path / "settings.toml"
    path.write_text(text.replace(old, new), encoding="utf-8")
    return path


def test_default_settings_match_the_documented_design():
    settings = load_settings(REPO_ROOT / "settings.toml")

    assert settings.sizes.prompt_tokens == {"short": 40, "medium": 400, "long": 2000}
    assert settings.sizes.context_tokens == {"short": 250, "medium": 2500, "long": 10000}
    assert settings.sizes.turn_counts == (1, 5, 10)
    assert (settings.retries.max_retries, settings.run.concurrency, settings.output.max_tokens) == (2, 2, 32000)


@pytest.mark.parametrize("name", SWEEP_FILES)
def test_sweep_settings_can_be_combined_with_the_default_run(name):
    default = load_settings(REPO_ROOT / "settings.toml")
    sweep = load_settings(REPO_ROOT / name)

    # Single-turn, so no request reads a cached prefix. Combining needs the same [output] and prompt sizes.
    assert sweep.sizes.turn_counts == (1,)
    assert sweep.output == default.output
    assert sweep.sizes.prompt_tokens == default.sizes.prompt_tokens
    assert all(label == f"{tokens // 1000}k" for label, tokens in sweep.sizes.context_tokens.items())
    assert sweep.run.repetitions == 5


@pytest.mark.parametrize(
    ("table", "message"),
    [
        ("", "non-empty table"),
        ("a = 0", "positive integer"),
        ('a = "big"', "positive integer"),
        ("a = true", "positive"),
    ],
)
def test_invalid_context_levels_are_rejected(tmp_path, table, message):
    text = (REPO_ROOT / "settings.toml").read_text(encoding="utf-8")
    start = text.index("[sizes.context_tokens]")
    end = text.index("[stats]")
    path = tmp_path / "settings.toml"
    path.write_text(text[:start] + f"[sizes.context_tokens]\n{table}\n\n" + text[end:], encoding="utf-8")

    with pytest.raises(SettingsError, match=message):
        load_settings(path)


@pytest.mark.parametrize("counts", ["[1, 1]", "[1, true]", "[1.0]", "[]", "[0, 5]", "[5, 11]"])
def test_turn_counts_must_be_distinct_integers_from_1_to_10(tmp_path, counts):
    # Duplicates would give two scenarios one ID and a negative failure count in aggregated.csv.
    path = settings_with(tmp_path, "turn_counts = [1, 5, 10]", f"turn_counts = {counts}")

    with pytest.raises(SettingsError, match="distinct integers from 1 to 10"):
        load_settings(path)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("concurrency = 2", "concurrency = 2.5", "run.concurrency must be a int"),
        ("concurrency = 2", 'concurrency = "2"', "run.concurrency must be a int"),
        ("warmup = true", "warmup = 1", "run.warmup must be a bool"),
        ("max_retries = 2", "max_retries = true", "retries.max_retries must be a int"),
        ("connect_timeout_s = 10.0", 'connect_timeout_s = "10"', "client.connect_timeout_s must be a float"),
    ],
)
def test_settings_values_of_the_wrong_type_are_rejected(tmp_path, old, new, message):
    with pytest.raises(SettingsError, match=message):
        load_settings(settings_with(tmp_path, old, new))


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("max_retries = 2", "max_retries = -1", "retries.max_retries must not be negative"),
        ("concurrency = 2", "concurrency = 0", "run.concurrency and run.repetitions must be at least 1"),
        ("repetitions = 1", "repetitions = 0", "run.concurrency and run.repetitions must be at least 1"),
    ],
)
def test_out_of_range_counts_are_rejected(tmp_path, old, new, message):
    with pytest.raises(SettingsError, match=message):
        load_settings(settings_with(tmp_path, old, new))


def test_the_smallest_allowed_counts_are_accepted(tmp_path):
    text = (
        (REPO_ROOT / "settings.toml")
        .read_text(encoding="utf-8")
        .replace("max_retries = 2", "max_retries = 0")
        .replace("concurrency = 2", "concurrency = 1")
        .replace("turn_counts = [1, 5, 10]", "turn_counts = [10, 1]")
    )
    path = tmp_path / "settings.toml"
    path.write_text(text, encoding="utf-8")

    settings = load_settings(path)

    assert (settings.retries.max_retries, settings.run.concurrency, settings.run.repetitions) == (0, 1, 1)
    assert settings.sizes.turn_counts == (10, 1)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("seed = 20260930\n", "", r"invalid \[run\] table: .*'seed'"),
        ("warmup = true", "warmup = true\nwarm_up = false", r"invalid \[run\] table: .*'warm_up'"),
        (
            "turn_counts = [1, 5, 10]",
            "turn_counts = [1, 5, 10]\nrepeats = 3",
            r"unknown key\(s\) in \[sizes\]: repeats",
        ),
    ],
    ids=["missing-key", "unknown-key", "unknown-sizes-key"],
)
def test_missing_or_unknown_keys_are_settings_errors(tmp_path, old, new, message):
    with pytest.raises(SettingsError, match=message):
        load_settings(settings_with(tmp_path, old, new))


def test_integer_value_for_a_float_setting_is_accepted(tmp_path):
    path = settings_with(tmp_path, "connect_timeout_s = 10.0", "connect_timeout_s = 10")

    assert load_settings(path).client.connect_timeout_s == 10


def test_invalid_toml_is_a_settings_error(tmp_path):
    with pytest.raises(SettingsError, match="not valid TOML"):
        load_settings(settings_with(tmp_path, "[run]", "[run"))
