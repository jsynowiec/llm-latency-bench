"""Time one streaming request and classify how it ended."""

import asyncio
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

import openai
from pydantic_ai import ModelMessage, ModelResponse
from pydantic_ai.direct import model_request_stream
from pydantic_ai.exceptions import ContentFilterError, ModelAPIError, ModelHTTPError
from pydantic_ai.messages import PartDeltaEvent, PartStartEvent, TextPart, TextPartDelta
from pydantic_ai.models import Model
from pydantic_ai.models.openrouter import OpenRouterModelSettings

from llm_latency_bench.credentials import Redactor

MAX_ERROR_MESSAGE_CHARS = 500


class ErrorCategory(StrEnum):
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    SERVER_ERROR = "server_error"
    CONNECTION = "connection"
    STREAM_ERROR = "stream_error"
    AUTH = "auth"
    INSUFFICIENT_CREDITS = "insufficient_credits"
    FORBIDDEN = "forbidden"
    BAD_REQUEST = "bad_request"
    HTTP_ERROR = "http_error"
    NOT_FOUND = "not_found"
    TRUNCATED = "truncated"
    CONTENT_FILTER = "content_filter"
    NO_VISIBLE_OUTPUT = "no_visible_output"
    UNEXPECTED_FINISH = "unexpected_finish"
    OTHER = "other"


RETRYABLE = frozenset(
    {
        ErrorCategory.TIMEOUT,
        ErrorCategory.RATE_LIMITED,
        ErrorCategory.SERVER_ERROR,
        ErrorCategory.CONNECTION,
        ErrorCategory.STREAM_ERROR,
    }
)


@dataclass(frozen=True)
class Usage:
    """Token counts as reported by the provider. None means the provider did not report the value."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None

    @property
    def visible_output_tokens(self) -> int | None:
        if self.output_tokens is None or self.reasoning_tokens is None:
            return None
        return self.output_tokens - self.reasoning_tokens


@dataclass
class Attempt:
    started_at_utc: datetime
    request_latency_s: float
    ttft_s: float | None = None
    post_ttft_s: float | None = None
    usage: Usage = field(default_factory=Usage)
    finish_reason: str | None = None
    response_model: str | None = None
    # The upstream provider OpenRouter used, and OpenRouter's ID for the request.
    upstream_provider: str | None = None
    generation_id: str | None = None
    response: ModelResponse | None = None
    visible_chars: int = 0
    error_category: ErrorCategory | None = None
    http_status: int | None = None
    error_message: str | None = None

    @property
    def ok(self) -> bool:
        return self.error_category is None

    @property
    def retryable(self) -> bool:
        return self.error_category in RETRYABLE

    @property
    def tps(self) -> float | None:
        """Visible output tokens per second after the first visible token. Includes stream completion overhead."""
        visible = self.usage.visible_output_tokens
        if not self.ok or visible is None or not self.post_ttft_s:
            return None
        return visible / self.post_ttft_s


def _visible_text(event: object) -> str:
    """Text the user would see from this stream event. Hidden reasoning (thinking parts) is not visible text."""
    match event:
        case PartStartEvent(part=TextPart(content=content)):
            return content
        case PartDeltaEvent(delta=TextPartDelta(content_delta=content)):
            return content
    return ""


def _usage(response: ModelResponse) -> Usage:
    usage = response.usage
    # Pydantic AI reports zeros when no usage chunk arrives. Real requests always have input tokens,
    # so zero input means "not reported" and every count stays empty.
    if usage.input_tokens == 0:
        return Usage()
    return Usage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        reasoning_tokens=usage.details.get("reasoning_tokens"),
        cache_read_tokens=usage.cache_read_tokens,
        cache_write_tokens=usage.cache_write_tokens,
    )


def _http_category(status: int) -> ErrorCategory:
    match status:
        case 401:
            return ErrorCategory.AUTH
        case 402:
            return ErrorCategory.INSUFFICIENT_CREDITS
        case 403:
            # OpenRouter uses 403 for guardrail blocks and moderation flags, not for a bad key.
            return ErrorCategory.FORBIDDEN
        case 404:
            return ErrorCategory.NOT_FOUND
        case 408:
            return ErrorCategory.TIMEOUT
        case 400 | 422:
            return ErrorCategory.BAD_REQUEST
        case 429:
            return ErrorCategory.RATE_LIMITED
        case _ if status >= 500:
            return ErrorCategory.SERVER_ERROR
        case _:
            # Not retried, and not taken as proof that the pair is unsupported.
            return ErrorCategory.HTTP_ERROR


def classify_error(error: BaseException) -> tuple[ErrorCategory, int | None]:
    match error:
        case TimeoutError() | openai.APITimeoutError():
            return ErrorCategory.TIMEOUT, None
        case ModelHTTPError(status_code=status):
            return _http_category(status), status
        case ContentFilterError():
            return ErrorCategory.CONTENT_FILTER, None
        case ModelAPIError() if isinstance(error.__cause__, openai.APITimeoutError):
            return ErrorCategory.TIMEOUT, None
        case ModelAPIError() if isinstance(error.__cause__, openai.APIConnectionError):
            return ErrorCategory.CONNECTION, None
        case ModelAPIError():
            # Error object inside an already-started stream, or an undecodable chunk.
            return ErrorCategory.STREAM_ERROR, None
        case openai.APIConnectionError():
            return ErrorCategory.CONNECTION, None
    return ErrorCategory.OTHER, None


def _error_text(error: BaseException, redact: Redactor) -> str:
    # For HTTP errors, Pydantic AI's message already includes the response body.
    return redact(f"{type(error).__name__}: {error}")[:MAX_ERROR_MESSAGE_CHARS]


async def stream_attempt(
    model: Model,
    messages: Sequence[ModelMessage],
    model_settings: OpenRouterModelSettings,
    request_timeout_s: float,
    redact: Redactor,
    clock: Callable[[], float] = time.perf_counter,
) -> Attempt:
    """Send one streaming request and time it.

    TTFT runs from just before the request is sent to the first stream event that carries non-whitespace
    visible text. Pydantic AI sends the HTTP request when the stream context is entered, so `start` is taken
    right before that.
    """
    started_at_utc = datetime.now(UTC)
    first_visible: float | None = None
    visible_chars = 0
    start = clock()
    try:
        async with (
            asyncio.timeout(request_timeout_s),
            model_request_stream(model, messages, model_settings=model_settings) as stream,
        ):
            async for event in stream:
                text = _visible_text(event)
                if first_visible is None and text.strip():
                    first_visible = clock()
                visible_chars += len(text)
            end = clock()
            response = stream.get()
    except Exception as error:
        end = clock()
        category, status = classify_error(error)
        return Attempt(
            started_at_utc=started_at_utc,
            request_latency_s=end - start,
            ttft_s=None if first_visible is None else first_visible - start,
            visible_chars=visible_chars,
            error_category=category,
            http_status=status,
            error_message=_error_text(error, redact),
        )

    attempt = Attempt(
        started_at_utc=started_at_utc,
        request_latency_s=end - start,
        ttft_s=None if first_visible is None else first_visible - start,
        post_ttft_s=None if first_visible is None else end - first_visible,
        usage=_usage(response),
        finish_reason=response.finish_reason,
        response_model=response.model_name,
        upstream_provider=(response.provider_details or {}).get("downstream_provider"),
        generation_id=response.provider_response_id,
        response=response,
        visible_chars=visible_chars,
    )
    _flag_incomplete(attempt)
    return attempt


def _flag_incomplete(attempt: Attempt) -> None:
    """Only a stream that finished normally (`finish_reason=stop`) with visible text is a valid latency sample."""
    match attempt.finish_reason:
        case None:
            category, message = ErrorCategory.STREAM_ERROR, "stream ended without a finish_reason"
        case "length":
            category, message = ErrorCategory.TRUNCATED, "stream stopped at max_tokens (finish_reason=length)"
        case "content_filter":
            category, message = ErrorCategory.CONTENT_FILTER, "stream stopped by content filter"
        case "error":
            category, message = ErrorCategory.STREAM_ERROR, "stream ended with finish_reason=error"
        case "stop" if attempt.ttft_s is None:
            category, message = ErrorCategory.NO_VISIBLE_OUTPUT, "stream finished without visible text"
        case "stop":
            return
        case _:
            category = ErrorCategory.UNEXPECTED_FINISH
            message = f"stream finished with finish_reason={attempt.finish_reason}"
    attempt.error_category = category
    attempt.error_message = message
