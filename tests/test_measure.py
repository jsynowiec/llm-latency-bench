import asyncio

import httpx2
import openai
import pytest
from pydantic_ai import ModelRequest
from pydantic_ai.exceptions import ContentFilterError
from pydantic_ai.messages import ThinkingPart
from pydantic_ai.models.openrouter import OpenRouterReasoning

from llm_latency_bench.credentials import REDACTED, Redactor
from llm_latency_bench.measure import Attempt, ErrorCategory, classify_error, stream_attempt
from tests.conftest import FAKE_API_KEY
from tests.fake_openrouter import (
    FAKE_PROVIDER,
    FakeOpenRouter,
    Reply,
    Script,
    chunk,
    error_reply,
    generation_id,
    reasoning_chunk,
    usage_chunk,
)

REDACT = Redactor((FAKE_API_KEY,))
REASONING: OpenRouterReasoning = {"effort": "xhigh", "exclude": True}


def measure(openrouter_models, reply: Reply | Script) -> tuple[Attempt, FakeOpenRouter]:
    if isinstance(reply, Reply):
        fixed_reply = reply
        server = FakeOpenRouter(lambda body, number: fixed_reply)
    else:
        server = FakeOpenRouter(reply)

    async def go() -> Attempt:
        async with openrouter_models(server) as models:
            return await stream_attempt(
                models("openai/gpt-6-sol"),
                [ModelRequest.user_text_prompt("hello")],
                {"openrouter_reasoning": REASONING, "max_tokens": 1000},
                request_timeout_s=5.0,
                redact=REDACT,
            )

    return asyncio.run(go()), server


def assert_took(measured: float | None, scripted: float) -> None:
    # Sleeps never end early, so a measured span is at least its scripted delay plus a little local overhead.
    assert measured is not None
    assert scripted - 0.005 <= measured <= scripted + 0.05


def test_ttft_starts_at_first_visible_text_not_at_metadata_reasoning_or_whitespace(openrouter_models):
    reply = Reply(
        chunks=[
            # Pydantic AI opens the stream only after this first chunk, so its delay is part of TTFT.
            (0.12, chunk("", role="assistant")),
            (0.05, reasoning_chunk("Let me think")),
            (0.04, reasoning_chunk(" about it.")),
            (0.06, chunk("  \n")),
            (0.09, chunk("Hello")),
            (0.03, chunk(" world")),
            (0.06, chunk(finish="stop")),
            (0.15, usage_chunk(prompt_tokens=50, completion_tokens=12, reasoning_tokens=10)),
        ]
    )
    attempt, _ = measure(openrouter_models, reply)

    assert attempt.ok, attempt.error_message
    assert_took(attempt.ttft_s, 0.12 + 0.05 + 0.04 + 0.06 + 0.09)
    # The window after TTFT runs to the end of the stream, including the late usage chunk.
    assert_took(attempt.post_ttft_s, 0.03 + 0.06 + 0.15)
    assert_took(attempt.request_latency_s, 0.60)
    assert attempt.visible_chars == len("  \nHello world")
    assert attempt.usage.visible_output_tokens == 2
    assert attempt.tps is not None and attempt.post_ttft_s is not None
    assert attempt.tps == pytest.approx(2 / attempt.post_ttft_s)
    assert attempt.upstream_provider == FAKE_PROVIDER
    assert attempt.generation_id == generation_id(1)


def test_request_goes_to_openrouter_with_the_reasoning_object_and_bearer_key(openrouter_models):
    _, server = measure(openrouter_models, Reply(chunks=[(0, chunk("Hi", finish="stop"))]))

    body = server.bodies[0]
    assert body["model"] == "openai/gpt-6-sol"
    assert body["reasoning"] == REASONING
    assert "reasoning_effort" not in body
    assert body["stream"] is True
    assert body["max_tokens"] == 1000
    request = server.requests[0]
    assert request.headers["authorization"] == f"Bearer {FAKE_API_KEY}"
    assert str(request.url) == "https://openrouter.ai/api/v1/chat/completions"


def test_missing_usage_leaves_tokens_and_tps_empty_instead_of_zero(openrouter_models):
    attempt, _ = measure(openrouter_models, Reply(chunks=[(0, chunk("Hi")), (0.01, chunk(finish="stop"))]))

    assert attempt.ok
    assert attempt.usage.input_tokens is None
    assert attempt.usage.output_tokens is None
    assert attempt.tps is None
    assert attempt.ttft_s is not None


def test_length_finish_is_a_truncated_failure_without_tps_even_when_usage_arrives(openrouter_models):
    reply = Reply(
        chunks=[
            (0, chunk("Partial")),
            (0.02, chunk(finish="length")),
            (0, usage_chunk(prompt_tokens=50, completion_tokens=12, reasoning_tokens=10)),
        ]
    )
    attempt, _ = measure(openrouter_models, reply)

    assert attempt.error_category is ErrorCategory.TRUNCATED
    assert not attempt.retryable
    # Everything needed for a TPS value is present. Only the failure keeps it out of the results.
    assert attempt.usage.visible_output_tokens == 2
    assert attempt.post_ttft_s
    assert attempt.tps is None


def test_content_filter_finish_is_not_retried(openrouter_models):
    reply = Reply(chunks=[(0, chunk("I can")), (0, chunk(finish="content_filter"))])
    attempt, _ = measure(openrouter_models, reply)

    assert attempt.error_category is ErrorCategory.CONTENT_FILTER
    assert not attempt.retryable


def test_tool_call_finish_is_not_a_sample(openrouter_models):
    attempt, _ = measure(openrouter_models, Reply(chunks=[(0, chunk("Hi")), (0, chunk(finish="tool_calls"))]))

    assert attempt.error_category is ErrorCategory.UNEXPECTED_FINISH
    assert not attempt.retryable
    assert "tool_call" in (attempt.error_message or "")


def test_error_finish_is_a_retryable_stream_error(openrouter_models):
    attempt, _ = measure(openrouter_models, Reply(chunks=[(0, chunk("Hi")), (0, chunk(finish="error"))]))

    assert attempt.error_category is ErrorCategory.STREAM_ERROR
    assert attempt.retryable


def test_stream_with_only_hidden_reasoning_is_no_visible_output(openrouter_models):
    reply = Reply(
        chunks=[
            (0, reasoning_chunk("Let me think")),
            (0, reasoning_chunk(" about it.")),
            (0, chunk(finish="stop")),
            (0, usage_chunk(prompt_tokens=50, completion_tokens=12, reasoning_tokens=12)),
        ]
    )
    attempt, _ = measure(openrouter_models, reply)

    assert attempt.error_category is ErrorCategory.NO_VISIBLE_OUTPUT
    assert attempt.ttft_s is None
    assert attempt.visible_chars == 0
    # The reasoning did reach Pydantic AI, so this is not just an empty stream.
    assert attempt.response is not None
    assert [part.content for part in attempt.response.parts if isinstance(part, ThinkingPart)] == [
        "Let me think about it."
    ]


def test_stream_that_ends_without_finish_reason_is_a_retryable_stream_error(openrouter_models):
    reply = Reply(chunks=[(0, chunk("Half an answer"))], send_done=False)
    attempt, _ = measure(openrouter_models, reply)

    assert attempt.error_category is ErrorCategory.STREAM_ERROR
    assert attempt.retryable
    assert attempt.ttft_s is not None


def test_rate_limit_is_retryable(openrouter_models):
    attempt, _ = measure(openrouter_models, error_reply(429, "slow down"))

    assert attempt.error_category is ErrorCategory.RATE_LIMITED
    assert attempt.http_status == 429
    assert attempt.retryable


@pytest.mark.parametrize(
    ("status", "category", "retryable"),
    [
        (400, ErrorCategory.BAD_REQUEST, False),
        (422, ErrorCategory.BAD_REQUEST, False),
        (409, ErrorCategory.HTTP_ERROR, False),
        (499, ErrorCategory.HTTP_ERROR, False),
        (401, ErrorCategory.AUTH, False),
        (402, ErrorCategory.INSUFFICIENT_CREDITS, False),
        (403, ErrorCategory.FORBIDDEN, False),
        (404, ErrorCategory.NOT_FOUND, False),
        (408, ErrorCategory.TIMEOUT, True),
        (500, ErrorCategory.SERVER_ERROR, True),
        (502, ErrorCategory.SERVER_ERROR, True),
        (503, ErrorCategory.SERVER_ERROR, True),
    ],
)
def test_http_errors_are_categorised(openrouter_models, status, category, retryable):
    attempt, _ = measure(openrouter_models, error_reply(status, "nope"))

    assert attempt.error_category is category
    assert attempt.retryable is retryable


def test_error_message_echoing_credentials_is_redacted(openrouter_models):
    attempt, _ = measure(openrouter_models, error_reply(401, f"invalid key {FAKE_API_KEY}"))

    assert attempt.error_message is not None
    assert FAKE_API_KEY not in attempt.error_message
    assert REDACTED in attempt.error_message


def test_long_error_message_is_redacted_and_cut_to_500_characters(openrouter_models):
    attempt, _ = measure(openrouter_models, error_reply(401, f"invalid key {FAKE_API_KEY} " + "detail " * 200))

    assert attempt.error_message is not None
    assert len(attempt.error_message) == 500
    assert FAKE_API_KEY not in attempt.error_message
    assert REDACTED in attempt.error_message


def test_refused_connection_is_a_retryable_connection_error(openrouter_models):
    def refuse(body, number):
        raise httpx2.ConnectError("connection refused")

    attempt, _ = measure(openrouter_models, refuse)

    assert attempt.error_category is ErrorCategory.CONNECTION
    assert attempt.http_status is None
    assert attempt.retryable


def test_connection_reset_mid_stream_keeps_the_ttft_already_seen(openrouter_models):
    reply = Reply(chunks=[(0.05, chunk("Hello")), (0.1, chunk(" wor"))], raise_after_chunks=httpx2.ReadError("reset"))
    attempt, _ = measure(openrouter_models, reply)

    assert attempt.error_category is ErrorCategory.CONNECTION
    assert attempt.retryable
    assert_took(attempt.ttft_s, 0.05)
    assert_took(attempt.request_latency_s, 0.15)
    assert attempt.visible_chars == len("Hello wor")
    assert attempt.post_ttft_s is None


@pytest.mark.parametrize(
    ("error", "category"),
    [
        # A direct model stream never raises these: Pydantic AI wraps connection errors in ModelAPIError,
        # and only agent runs raise ContentFilterError. So the fake cannot produce them.
        (ContentFilterError("blocked"), ErrorCategory.CONTENT_FILTER),
        (openai.APIConnectionError(request=httpx2.Request("POST", "https://openrouter.ai")), ErrorCategory.CONNECTION),
    ],
)
def test_unwrapped_errors_are_categorised(error, category):
    assert classify_error(error) == (category, None)


def test_request_timeout_is_recorded_as_timeout(openrouter_models):
    server = FakeOpenRouter(lambda body, number: Reply(chunks=[(0, chunk("Hi")), (1.0, chunk(finish="stop"))]))

    async def go() -> Attempt:
        async with openrouter_models(server) as models:
            return await stream_attempt(
                models("openai/gpt-6-sol"),
                [ModelRequest.user_text_prompt("hi")],
                {},
                request_timeout_s=0.2,
                redact=REDACT,
            )

    attempt = asyncio.run(go())
    assert attempt.error_category is ErrorCategory.TIMEOUT
    assert attempt.retryable
    assert attempt.request_latency_s == pytest.approx(0.2, abs=0.1)
