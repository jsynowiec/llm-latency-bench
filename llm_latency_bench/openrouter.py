"""Pydantic AI chat models that stream from OpenRouter."""

from collections.abc import Callable

import httpx2
from openai import AsyncOpenAI
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

from llm_latency_bench.settings import ClientSettings

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def build_http_client(
    settings: ClientSettings, transport: httpx2.AsyncBaseTransport | None = None
) -> httpx2.AsyncClient:
    return httpx2.AsyncClient(
        # `read` limits silence between chunks. The runner enforces the whole-request limit.
        timeout=httpx2.Timeout(settings.read_timeout_s, connect=settings.connect_timeout_s),
        limits=httpx2.Limits(keepalive_expiry=settings.keepalive_expiry_s),
        transport=transport,
    )


type ModelFactory = Callable[[str], OpenRouterModel]


def model_factory(api_key: str, http_client: httpx2.AsyncClient) -> ModelFactory:
    # The harness retries itself, so every attempt is timed and recorded.
    client = AsyncOpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        max_retries=0,
        timeout=http_client.timeout,
        http_client=http_client,
    )
    provider = OpenRouterProvider(openai_client=client)
    return lambda model_slug: OpenRouterModel(model_slug, provider=provider)
