import json

import logfire

from llm_latency_bench.credentials import Redactor
from llm_latency_bench.tracing import local_tracing
from tests.conftest import FAKE_API_KEY


def test_spans_go_to_the_local_file_and_credential_values_are_scrubbed(tmp_path):
    trace_path = tmp_path / "traces.jsonl"
    with local_tracing(trace_path, Redactor((FAKE_API_KEY,))):
        with logfire.span("plain", note="plain text survives"):
            pass
        try:
            with logfire.span("failing"):
                raise RuntimeError(f"error echoed {FAKE_API_KEY}")
        except RuntimeError:
            pass

    text = trace_path.read_text(encoding="utf-8")
    spans = {span["name"]: span for span in map(json.loads, text.splitlines())}
    assert spans["plain"]["attributes"]["note"] == "plain text survives"
    assert "failing" in spans
    assert FAKE_API_KEY not in text
    assert "error echoed [REDACTED]" in text


def test_logfire_itself_scrubs_attributes_that_contain_the_key(tmp_path):
    # Logfire scrubs before any exporter, so this is also what an OTLP endpoint receives. The key avoids
    # Logfire's default patterns (such as "secret" or "api key"), so only the key-specific pattern can match.
    key = "sk-or-v1-0123456789abcdef"
    trace_path = tmp_path / "traces.jsonl"
    with local_tracing(trace_path, Redactor((key,))), logfire.span("call", note=f"sent with {key} today"):
        pass

    [span] = map(json.loads, trace_path.read_text(encoding="utf-8").splitlines())
    note = span["attributes"]["note"]
    assert note.startswith("[Scrubbed")
    assert "sent with" not in note
