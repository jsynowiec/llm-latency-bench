import dataclasses
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from llm_latency_bench.openrouter import ModelFactory, build_http_client, model_factory
from llm_latency_bench.settings import Settings, load_settings
from tests.fake_openrouter import FakeOpenRouter

REPO_ROOT = Path(__file__).resolve().parent.parent
FAKE_API_KEY = "sk-or-test-SECRET-api-key-1234567890"


@pytest.fixture
def settings() -> Settings:
    base = load_settings(REPO_ROOT / "settings.toml")
    # No backoff wait and a short request timeout keep tests fast.
    return dataclasses.replace(
        base,
        retries=dataclasses.replace(base.retries, backoff_base_s=0.0),
        client=dataclasses.replace(base.client, request_timeout_s=5.0),
    )


@pytest.fixture
def openrouter_models(settings: Settings):
    """Build a model factory whose HTTP traffic goes to a FakeOpenRouter."""

    @asynccontextmanager
    async def build(server: FakeOpenRouter) -> AsyncIterator[ModelFactory]:
        async with build_http_client(settings.client, transport=server.transport()) as http_client:
            yield model_factory(FAKE_API_KEY, http_client)

    return build
