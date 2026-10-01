from pathlib import Path

import pytest

from llm_latency_bench.models_csv import ModelsCsvError, filter_pairs, load_model_pairs
from tests.conftest import REPO_ROOT

HEADER = "model_name,reasoning_level_name,model_slug,reasoning_level\n"


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "models.csv"
    path.write_text(text, encoding="utf-8")
    return path


def test_repository_models_csv_loads_in_file_order():
    pairs = load_model_pairs(REPO_ROOT / "models.csv")

    assert len(pairs) == 8
    assert pairs[0].key == "openai/gpt-6-astra/low"
    assert pairs[-1].key == "openai/gpt-6-astra/xhigh"
    assert pairs[3].model_name == "GPT-6 Luna"


def test_missing_column_is_rejected(tmp_path):
    path = write(tmp_path, "model_name,model_slug,reasoning_level\nA,a,low\n")
    with pytest.raises(ModelsCsvError, match="reasoning_level_name"):
        load_model_pairs(path)


def test_blank_value_is_rejected_with_line_number(tmp_path):
    path = write(tmp_path, HEADER + "A,Low,a,low\nB,High, ,high\n")
    with pytest.raises(ModelsCsvError, match=r":3: blank value\(s\) in column\(s\): model_slug"):
        load_model_pairs(path)


def test_duplicate_model_reasoning_pair_is_rejected(tmp_path):
    path = write(tmp_path, HEADER + "A,Low,a,low\nA again,Low,a,low\n")
    with pytest.raises(ModelsCsvError, match="duplicate model/reasoning pair a/low"):
        load_model_pairs(path)


def test_same_model_with_different_reasoning_is_allowed(tmp_path):
    path = write(tmp_path, HEADER + "A,Low,a,low\nA,High,a,high\n")
    assert [pair.key for pair in load_model_pairs(path)] == ["a/low", "a/high"]


def test_filters_combine_model_and_reasoning():
    pairs = load_model_pairs(REPO_ROOT / "models.csv")

    selected = filter_pairs(pairs, models=["openai/gpt-6-sol"], reasoning_levels=["high", "xhigh"])

    assert [pair.key for pair in selected] == ["openai/gpt-6-sol/high", "openai/gpt-6-sol/xhigh"]


def test_unknown_filter_value_is_an_error_not_an_empty_run():
    pairs = load_model_pairs(REPO_ROOT / "models.csv")
    with pytest.raises(ModelsCsvError, match="gpt-7"):
        filter_pairs(pairs, models=["gpt-7"])


def test_filters_that_match_nothing_are_an_error():
    pairs = load_model_pairs(REPO_ROOT / "models.csv")
    with pytest.raises(ModelsCsvError, match="matched no"):
        filter_pairs(pairs, models=["openai/gpt-6-luna"], reasoning_levels=["low"])
