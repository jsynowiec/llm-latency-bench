"""A local stand-in for the OpenRouter chat completions endpoint, driven through httpx2.MockTransport.
Responses are scripted per request, and each streamed chunk can be delayed so timing code sees real gaps.
"""

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

import httpx2

FAKE_PROVIDER = "FakeProvider"

type Script = Callable[[dict[str, Any], int], "Reply"]


def generation_id(request_number: int) -> str:
    return f"gen-{request_number}"


def chunk(content: str | None = None, *, finish: str | None = None, **delta: Any) -> dict[str, Any]:
    if content is not None:
        delta["content"] = content
    return {
        "id": "chatcmpl-fake",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "fake-model",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def reasoning_chunk(text: str) -> dict[str, Any]:
    # OpenRouterModel builds ThinkingParts only from `reasoning_details`. It ignores `reasoning_content`.
    return chunk(reasoning_details=[{"type": "reasoning.text", "text": text, "index": 0, "format": "unknown"}])


def usage_chunk(prompt_tokens: int, completion_tokens: int, reasoning_tokens: int, cached_tokens: int = 0) -> dict:
    return {
        "id": "chatcmpl-fake",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "fake-model",
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_tokens_details": {"cached_tokens": cached_tokens},
            "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
        },
    }


@dataclass
class Reply:
    status: int = 200
    # (delay before this chunk in seconds, chunk JSON). Ignored for error replies.
    chunks: list[tuple[float, dict[str, Any]]] = field(default_factory=list)
    body: dict[str, Any] | None = None
    send_done: bool = True
    # Raised by the stream after the chunks, like a connection reset in the middle of a response.
    raise_after_chunks: Exception | None = None


def text_reply(text: str, *, prompt_tokens: int = 100) -> Reply:
    return Reply(
        chunks=[
            (0, chunk("", role="assistant")),
            (0, chunk(text)),
            (0, chunk(finish="stop")),
            (0, usage_chunk(prompt_tokens, 30, 20)),
        ],
    )


def error_reply(status: int, message: str) -> Reply:
    return Reply(status=status, body={"error": {"code": status, "message": message}})


async def _sse(reply: Reply, generation_id: str) -> AsyncIterator[bytes]:
    for delay, data in reply.chunks:
        if delay:
            await asyncio.sleep(delay)
        # OpenRouter puts its generation ID in every chunk and names the upstream provider.
        data = data | {"id": generation_id, "provider": FAKE_PROVIDER}
        yield f"data: {json.dumps(data)}\n\n".encode()
    if reply.raise_after_chunks is not None:
        raise reply.raise_after_chunks
    if reply.send_done:
        yield b"data: [DONE]\n\n"


class FakeOpenRouter:
    """Records every request and answers with `script(request_json, request_number)`."""

    def __init__(self, script: Script) -> None:
        self._script = script
        self.requests: list[httpx2.Request] = []

    @property
    def bodies(self) -> list[dict[str, Any]]:
        return [json.loads(request.content) for request in self.requests]

    async def handle(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        number = len(self.requests)
        reply = self._script(json.loads(request.content), number)
        if reply.status != 200:
            return httpx2.Response(reply.status, json=reply.body)
        stream = _sse(reply, generation_id(number))
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=stream)

    def transport(self) -> httpx2.MockTransport:
        return httpx2.MockTransport(self.handle)


def echo_turns(body: dict[str, Any], number: int) -> Reply:
    """Default script: answer every request with a reply naming the request number."""
    return text_reply(f"Answer {number}.")
