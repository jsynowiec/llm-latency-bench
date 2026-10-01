import pytest

from llm_latency_bench.credentials import REDACTED, CredentialsError, Redactor, load_api_key

GOOD = {"OPENROUTER_API_KEY": " sk-or-live-abc "}


def test_environment_value_is_loaded_and_stripped():
    assert load_api_key(None, GOOD) == "sk-or-live-abc"


def test_environment_value_wins_over_env_file(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("OPENROUTER_API_KEY=sk-or-from-file\n", encoding="utf-8")

    assert load_api_key(env_file, GOOD) == "sk-or-live-abc"


def test_env_file_is_used_when_the_environment_lacks_the_key(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("OPENROUTER_API_KEY=sk-or-from-file\n", encoding="utf-8")

    assert load_api_key(env_file, {}) == "sk-or-from-file"


@pytest.mark.parametrize("environ", [{}, {"OPENROUTER_API_KEY": "   "}])
def test_missing_key_is_named(environ):
    with pytest.raises(CredentialsError, match="missing environment variable: OPENROUTER_API_KEY"):
        load_api_key(None, environ)


def test_env_example_placeholder_is_rejected():
    with pytest.raises(CredentialsError, match="placeholder"):
        load_api_key(None, {"OPENROUTER_API_KEY": "replace-with-your-openrouter-api-key"})


def test_redactor_replaces_every_occurrence_and_longest_secret_first():
    redact = Redactor(("abc", "abcdef"))

    assert redact("key=abcdef other=abc") == f"key={REDACTED} other={REDACTED}"
