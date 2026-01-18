# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Utility functions for the AIPerf Mock Server."""

import asyncio
import logging
import random
import time
import uuid
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from functools import wraps
from time import perf_counter
from typing import TYPE_CHECKING, Any

import orjson
from aiperf_mock_server.config import server_config

if TYPE_CHECKING:
    from aiperf_mock_server.config import MockServerConfig
from aiperf_mock_server.metrics_utils import (
    record_itl,
    record_streamed_token,
    record_ttft,
)
from aiperf_mock_server.models import (
    ChatCompletionRequest,
    CohereRerankRequest,
    CompletionRequest,
    EmbeddingRequest,
    HFTEIRerankRequest,
    ImageGenerationRequest,
    RankingRequest,
    RequestT,
    SolidoRAGRequest,
    TGIGenerateRequest,
)
from aiperf_mock_server.tokens import TokenizedText, tokenize_request
from fastapi import HTTPException

logger = logging.getLogger(__name__)

# ============================================================================
# FastAPI Decorators
# ============================================================================


def with_error_injection(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to inject errors based on config."""

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any):
        if (
            server_config.error_rate > 0
            and random.random() * 100 < server_config.error_rate
        ):
            raise HTTPException(status_code=500, detail="Simulated error")
        return await func(*args, **kwargs)

    return wrapper


# ============================================================================
# Timing & Latency Simulation
# ============================================================================


class LatencySimulator:
    """Simulates API latency with TTFT and ITL."""

    __slots__ = (
        "ttft_sec",
        "itl_sec",
        "start_time",
        "token_index",
        "last_token_time",
        "endpoint",
        "model",
        "measured_ttft",
        "measured_decode",
    )

    def __init__(
        self,
        endpoint: str,
        model: str,
        start_time: float,
        config: "MockServerConfig | None" = None,
    ) -> None:
        cfg = config or server_config
        self.ttft_sec = cfg.ttft * 0.001
        self.itl_sec = cfg.itl * 0.001
        self.start_time = start_time
        self.token_index = 0
        self.last_token_time: float | None = None
        self.endpoint = endpoint
        self.model = model
        self.measured_ttft: float = 0.0
        self.measured_decode: float = 0.0

    async def wait_for_next_token(self) -> None:
        """Wait for TTFT (first token) or ITL (subsequent tokens)."""
        await self._wait_for_token_at_index(self.token_index)

        now = perf_counter()
        if self.token_index == 0:
            ttft = now - self.start_time
            self.measured_ttft = ttft
            record_ttft(self.endpoint, self.model, ttft)
        elif self.last_token_time is not None:
            itl = now - self.last_token_time
            record_itl(self.endpoint, self.model, itl)

        self.last_token_time = now
        self.token_index += 1

    async def _wait_for_token_at_index(self, token_index: int) -> None:
        """Wait until the specified token index should be emitted."""
        target_time = self.start_time + self.ttft_sec + (self.itl_sec * token_index)
        remaining = target_time - perf_counter()
        if remaining > 0:
            await asyncio.sleep(remaining)

    async def wait_for_tokens(self, num_tokens: int) -> None:
        """Wait for entire completion (TTFT + ITL * num_tokens)."""
        # Wait for TTFT first (prefill phase)
        ttft_target = self.start_time + self.ttft_sec
        ttft_remaining = ttft_target - perf_counter()
        if ttft_remaining > 0:
            await asyncio.sleep(ttft_remaining)

        self.measured_ttft = perf_counter() - self.start_time

        # Wait for decode phase (ITL * num_tokens)
        decode_target = ttft_target + (self.itl_sec * num_tokens)
        decode_remaining = decode_target - perf_counter()
        if decode_remaining > 0:
            await asyncio.sleep(decode_remaining)

        self.measured_decode = perf_counter() - self.start_time - self.measured_ttft


# ============================================================================
# Request Context
# ============================================================================


@dataclass(slots=True)
class RequestCtx:
    """Request context - all fields directly accessible."""

    request_id: str
    model: str
    tokenized: TokenizedText
    usage: dict[str, Any]
    latency_sim: LatencySimulator

    @property
    def tokens(self) -> list[str]:
        return self.tokenized.tokens

    @property
    def content(self) -> str:
        return self.tokenized.content

    @property
    def finish_reason(self) -> str:
        return self.tokenized.finish_reason

    @property
    def reasoning_content(self) -> str | None:
        return self.tokenized.reasoning_content

    @property
    def reasoning_content_tokens(self) -> list[str]:
        return self.tokenized.reasoning_content_tokens


def make_ctx(
    request: RequestT,
    endpoint: str,
    start_time: float,
    config: "MockServerConfig | None" = None,
) -> RequestCtx:
    """Create request context with all fields directly accessible.

    Args:
        request: The parsed request object.
        endpoint: The endpoint path string.
        start_time: Request start time from perf_counter().
        config: Optional MockServerConfig for test isolation. Falls back to global config.
    """
    model = getattr(request, "model", "unknown")
    tokenized = tokenize_request(request)

    return RequestCtx(
        request_id=_create_request_id(request),
        model=model,
        tokenized=tokenized,
        usage=tokenized.create_usage(),
        latency_sim=LatencySimulator(endpoint, model, start_time, config),
    )


def _create_request_id(request: RequestT) -> str:
    """Generate request ID based on request type."""
    match request:
        case ChatCompletionRequest():
            return f"chatcmpl-{uuid.uuid4()}"
        case CompletionRequest() | TGIGenerateRequest():
            return f"cmpl-{uuid.uuid4()}"
        case EmbeddingRequest():
            return f"emb-{uuid.uuid4()}"
        case RankingRequest() | HFTEIRerankRequest() | CohereRerankRequest():
            return f"rank-{uuid.uuid4()}"
        case ImageGenerationRequest():
            return f"img-{uuid.uuid4()}"
        case SolidoRAGRequest():
            return f"rag-{uuid.uuid4()}"
        case _:
            raise ValueError(f"Invalid request type: {type(request)}")


# ============================================================================
# Streaming & Response Generation
# ============================================================================

# SSE prefix/suffix as bytes for efficient concatenation
_SSE_DATA_PREFIX = b"data: "
_SSE_NEWLINES = b"\n\n"
_SSE_DONE = b"data: [DONE]\n\n"


def _sse(data: dict[str, Any]) -> bytes:
    """Format data as SSE chunk bytes."""
    return _SSE_DATA_PREFIX + orjson.dumps(data) + _SSE_NEWLINES


async def stream_chat_completion(
    ctx: RequestCtx, endpoint: str, include_usage: bool
) -> AsyncGenerator[bytes, None]:
    """Stream chat completion tokens as SSE chunks."""
    has_reasoning = bool(ctx.reasoning_content_tokens)

    # Stream reasoning tokens first (if any)
    for token in ctx.reasoning_content_tokens:
        await ctx.latency_sim.wait_for_next_token()
        record_streamed_token(endpoint, ctx.model)
        yield _sse(
            {
                "id": ctx.request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": ctx.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "reasoning_content": token},
                    }
                ],
            }
        )

    # Stream output tokens
    num_tokens = len(ctx.tokens)
    for i, token in enumerate(ctx.tokens):
        await ctx.latency_sim.wait_for_next_token()
        record_streamed_token(endpoint, ctx.model)

        delta: dict[str, Any] = {"content": token}
        if i == 0 and not has_reasoning:
            delta["role"] = "assistant"

        choice: dict[str, Any] = {"index": 0, "delta": delta}
        if i == num_tokens - 1:
            choice["finish_reason"] = ctx.finish_reason

        yield _sse(
            {
                "id": ctx.request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": ctx.model,
                "choices": [choice],
            }
        )

    # Final usage chunk (if requested)
    if include_usage:
        yield _sse(
            {
                "id": ctx.request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": ctx.model,
                "choices": [],
                "usage": ctx.usage,
            }
        )

    yield _SSE_DONE


async def stream_text_completion(
    ctx: RequestCtx, endpoint: str, include_usage: bool
) -> AsyncGenerator[bytes, None]:
    """Stream text completion tokens as SSE chunks."""
    num_tokens = len(ctx.tokens)

    for i, token in enumerate(ctx.tokens):
        await ctx.latency_sim.wait_for_next_token()
        record_streamed_token(endpoint, ctx.model)

        choice: dict[str, Any] = {"index": 0, "text": token}
        if i == num_tokens - 1:
            choice["finish_reason"] = ctx.finish_reason

        yield _sse(
            {
                "id": ctx.request_id,
                "object": "text_completion",
                "created": int(time.time()),
                "model": ctx.model,
                "choices": [choice],
            }
        )

    if include_usage:
        yield _sse(
            {
                "id": ctx.request_id,
                "object": "text_completion",
                "created": int(time.time()),
                "model": ctx.model,
                "choices": [],
                "usage": ctx.usage,
            }
        )

    yield _SSE_DONE


async def stream_tgi_completion(
    ctx: RequestCtx, endpoint: str, _include_usage: bool = False
) -> AsyncGenerator[bytes, None]:
    """Stream TGI tokens as SSE chunks (include_usage ignored - TGI doesn't support it)."""
    num_tokens = len(ctx.tokens)

    for i, token_text in enumerate(ctx.tokens):
        await ctx.latency_sim.wait_for_next_token()
        record_streamed_token(endpoint, ctx.model)

        chunk: dict[str, Any] = {
            "index": i,
            "token": {
                "id": i,
                "text": token_text,
                "logprob": -0.1,
                "special": False,
            },
        }
        if i == num_tokens - 1:
            chunk["generated_text"] = ctx.content

        yield _sse(chunk)
