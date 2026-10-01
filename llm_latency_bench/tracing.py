"""Logfire tracing that stays on this machine: spans go to a JSON Lines file, nothing is sent to Logfire's service."""

import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import logfire
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from llm_latency_bench.credentials import Redactor


@contextmanager
def local_tracing(trace_path: Path, redact: Redactor) -> Iterator[None]:
    """Write benchmark spans to `trace_path`, one JSON object per line, until the block exits."""

    def json_line(span: ReadableSpan) -> str:
        # Logfire does not scrub exception messages or span status, so every span is redacted again.
        return redact(span.to_json(indent=None)) + "\n"

    with trace_path.open("w", encoding="utf-8") as handle:
        processor = BatchSpanProcessor(ConsoleSpanExporter(out=handle, formatter=json_line))
        logfire.configure(
            send_to_logfire=False,
            console=False,
            service_name="llm-latency-bench",
            metrics=False,
            inspect_arguments=False,
            # Also replace attribute values that contain a credential.
            scrubbing=logfire.ScrubbingOptions(
                extra_patterns=[re.escape(secret) for secret in redact.secrets if secret]
            ),
            additional_span_processors=[processor],
        )
        # Leave message text out of spans to keep the trace file small.
        logfire.instrument_pydantic_ai(include_content=False)
        try:
            yield
        finally:
            logfire.force_flush()
            processor.shutdown()
