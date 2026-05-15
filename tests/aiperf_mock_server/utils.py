# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Utility functions for the AIPerf Mock Server."""

import asyncio
import logging
import math
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
from aiperf_mock_server.metrics import DYNAMO_FRONTEND_DISCONNECTED_CLIENTS
from aiperf_mock_server.metrics_utils import (
    get_inflight_count,
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


def _lognormal_jitter(cv: float) -> float:
    """Lognormal multiplier with mean ~= 1.0 and CV = `cv`. cv<=0 returns 1.0.

    sigma = sqrt(ln(1 + cv**2)); factor = exp(sigma * Z - sigma**2 / 2),
    Z ~ N(0, 1). The -sigma**2/2 term keeps E[factor] == 1, so callers can
    multiply a base latency without biasing the mean.
    """
    if cv <= 0.0:
        return 1.0
    sigma = math.sqrt(math.log1p(cv * cv))
    z = random.gauss(0.0, 1.0)
    return math.exp(sigma * z - 0.5 * sigma * sigma)


def _positive_jitter_extra_seconds(base_ms: float, cv: float) -> float:
    """Extra (>=0) seconds to add as jitter on top of base_ms.

    Used when a structural floor (e.g. scheduler admit) prevents pulling
    timing earlier than nominal — we can only ever delay, not accelerate.
    Returns 0 when the lognormal sample would have been faster.
    """
    if cv <= 0.0 or base_ms <= 0.0:
        return 0.0
    factor = _lognormal_jitter(cv)
    if factor <= 1.0:
        return 0.0
    return (factor - 1.0) * base_ms * 0.001


class LatencySimulator:
    """Simulates API latency with TTFT and ITL.

    Latency formula (all coefficients default to 0.0 -> constant TTFT/ITL):
        ttft_ms = (cfg.ttft
                  + cfg.ttft_per_isl_token_ms * isl
                  + cfg.ttft_concurrency_quad_ms * active_inflight ** 2)
                  * lognormal_jitter(cfg.ttft_jitter_cv)
        itl_ms  = (cfg.itl
                  + cfg.itl_per_osl_token_ms * osl
                  + cfg.itl_concurrency_lin_ms * active_inflight)
                  * lognormal_jitter(cfg.itl_jitter_cv)   # resampled per token

    `active_inflight` is sampled lazily on first wait so the per-request
    `record_llm_inflight_start` bump is reflected. TTFT jitter is sampled
    once per request; ITL jitter is sampled fresh per token.
    """

    __slots__ = (
        "_cfg",
        "_isl",
        "_osl",
        "_latencies_ready",
        "_itl_base_sec",
        "_finished",
        "_cancelled",
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
        isl: int = 0,
        osl: int = 0,
    ) -> None:
        self._cfg = config or server_config
        self._isl = isl
        self._osl = osl
        self._latencies_ready = False
        # Filled in on first wait via _ensure_latencies(); pre-populated with
        # the static base so callers that skip the wait path (tests) still get
        # a sensible value.
        self.ttft_sec = self._cfg.ttft * 0.001
        self.itl_sec = self._cfg.itl * 0.001
        self._itl_base_sec = self.itl_sec
        self.start_time = start_time
        self.token_index = 0
        self.last_token_time: float | None = None
        self.endpoint = endpoint
        self.model = model
        self.measured_ttft: float = 0.0
        self.measured_decode: float = 0.0
        self._finished = False
        self._cancelled = False

    @property
    def request_key(self) -> str:
        """Stable per-request key used to identify scheduler waiters."""
        return f"{self.endpoint}-{id(self)}"

    def mark_finished(self) -> None:
        """Mark this request as completed normally — disables disconnect handling."""
        self._finished = True

    def cancel(self) -> None:
        """Free any scheduler slots for this request and record the disconnect.

        Idempotent. Safe to call from a generator's `finally` even when the
        request completed normally (no-op if `mark_finished` was called).
        """
        if self._finished or self._cancelled:
            return
        self._cancelled = True
        cfg = self._cfg
        if cfg.scheduler_enabled:
            from aiperf_mock_server.scheduler import get_scheduler

            sched = get_scheduler()
            if sched is not None:
                sched.cancel(self.request_key)
        try:
            DYNAMO_FRONTEND_DISCONNECTED_CLIENTS.labels(model=self.model).inc()
        except Exception:
            logger.debug("disconnect metric inc failed", exc_info=True)

    def _ensure_latencies(self) -> None:
        """Sample active concurrency once and freeze ttft_sec/itl_sec."""
        if self._latencies_ready:
            return
        cfg = self._cfg
        active = get_inflight_count()
        ttft_ms = (
            cfg.ttft
            + cfg.ttft_per_isl_token_ms * self._isl
            + cfg.ttft_concurrency_quad_ms * (active * active)
        )
        itl_ms = (
            cfg.itl
            + cfg.itl_per_osl_token_ms * self._osl
            + cfg.itl_concurrency_lin_ms * active
        )
        ttft_ms *= _lognormal_jitter(cfg.ttft_jitter_cv)
        self.ttft_sec = ttft_ms * 0.001
        self._itl_base_sec = itl_ms * 0.001
        self.itl_sec = self._itl_base_sec
        self._latencies_ready = True

    async def wait_for_next_token(self) -> None:
        """Wait for TTFT (first token) or ITL (subsequent tokens)."""
        cfg = self._cfg
        if cfg.scheduler_enabled:
            from aiperf_mock_server.scheduler import get_scheduler

            sched = get_scheduler()
            if sched is not None:
                await self._wait_via_scheduler(sched)
                return

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

    async def _wait_via_scheduler(self, sched) -> None:
        """Scheduler-driven path: prefill on first call, then per-token decode admits."""
        cfg = self._cfg
        if self.token_index == 0:
            await sched.run_prefill(
                request_id=self.request_key,
                prompt_tokens=max(1, self._isl),
            )
            extra = _positive_jitter_extra_seconds(cfg.ttft, cfg.ttft_jitter_cv)
            if extra > 0:
                await asyncio.sleep(extra)
            now = perf_counter()
            self.measured_ttft = now - self.start_time
            record_ttft(self.endpoint, self.model, self.measured_ttft)
            self.last_token_time = now
            self.token_index += 1
            return
        await sched.next_decode_step(self.request_key)
        extra = _positive_jitter_extra_seconds(cfg.itl, cfg.itl_jitter_cv)
        if extra > 0:
            await asyncio.sleep(extra)
        now = perf_counter()
        if self.last_token_time is not None:
            record_itl(self.endpoint, self.model, now - self.last_token_time)
        self.last_token_time = now
        self.token_index += 1

    async def _wait_for_token_at_index(self, token_index: int) -> None:
        """Wait until the specified token index should be emitted."""
        self._ensure_latencies()
        cfg = self._cfg
        if token_index == 0:
            target_time = self.start_time + self.ttft_sec
        else:
            # Per-token ITL jitter: sample relative to last token emission.
            anchor = self.last_token_time
            jittered_itl = self._itl_base_sec * _lognormal_jitter(cfg.itl_jitter_cv)
            self.itl_sec = jittered_itl
            if anchor is None:
                target_time = (
                    self.start_time + self.ttft_sec + jittered_itl * token_index
                )
            else:
                target_time = anchor + jittered_itl
        remaining = target_time - perf_counter()
        if remaining > 0:
            await asyncio.sleep(remaining)

    async def wait_for_tokens(self, num_tokens: int) -> None:
        """Wait for entire completion (TTFT + ITL * num_tokens)."""
        cfg = self._cfg
        if cfg.scheduler_enabled:
            from aiperf_mock_server.scheduler import get_scheduler

            sched = get_scheduler()
            if sched is not None:
                await sched.run_prefill(
                    request_id=self.request_key,
                    prompt_tokens=max(1, self._isl),
                )
                ttft_extra = _positive_jitter_extra_seconds(
                    cfg.ttft, cfg.ttft_jitter_cv
                )
                if ttft_extra > 0:
                    await asyncio.sleep(ttft_extra)
                self.measured_ttft = perf_counter() - self.start_time
                for _ in range(num_tokens):
                    await sched.next_decode_step(self.request_key)
                    itl_extra = _positive_jitter_extra_seconds(
                        cfg.itl, cfg.itl_jitter_cv
                    )
                    if itl_extra > 0:
                        await asyncio.sleep(itl_extra)
                self.measured_decode = (
                    perf_counter() - self.start_time - self.measured_ttft
                )
                return

        # Open-loop fallback (existing behavior + jitter).
        self._ensure_latencies()
        ttft_target = self.start_time + self.ttft_sec
        ttft_remaining = ttft_target - perf_counter()
        if ttft_remaining > 0:
            await asyncio.sleep(ttft_remaining)
        self.measured_ttft = perf_counter() - self.start_time
        if cfg.itl_jitter_cv > 0.0:
            # Sum of N independent lognormal samples — fall back to per-token loop.
            decode_target = perf_counter()
            for _ in range(num_tokens):
                decode_target += self._itl_base_sec * _lognormal_jitter(
                    cfg.itl_jitter_cv
                )
        else:
            decode_target = ttft_target + (self._itl_base_sec * num_tokens)
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
    """Unique identifier for this request."""

    model: str
    """Model name from the request."""

    tokenized: TokenizedText
    """Tokenized input and generated output."""

    usage: dict[str, Any]
    """Token usage statistics for the response."""

    latency_sim: LatencySimulator
    """Latency simulator for TTFT and ITL timing."""

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
        latency_sim=LatencySimulator(
            endpoint,
            model,
            start_time,
            config,
            isl=tokenized.prompt_token_count,
            osl=len(tokenized.tokens),
        ),
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

    try:
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

        ctx.latency_sim.mark_finished()
        yield _SSE_DONE
    finally:
        ctx.latency_sim.cancel()


async def stream_text_completion(
    ctx: RequestCtx, endpoint: str, include_usage: bool
) -> AsyncGenerator[bytes, None]:
    """Stream text completion tokens as SSE chunks."""
    num_tokens = len(ctx.tokens)

    try:
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

        ctx.latency_sim.mark_finished()
        yield _SSE_DONE
    finally:
        ctx.latency_sim.cancel()


async def stream_tgi_completion(
    ctx: RequestCtx, endpoint: str, _include_usage: bool = False
) -> AsyncGenerator[bytes, None]:
    """Stream TGI tokens as SSE chunks (include_usage ignored - TGI doesn't support it)."""
    num_tokens = len(ctx.tokens)

    try:
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
                ctx.latency_sim.mark_finished()

            yield _sse(chunk)
    finally:
        ctx.latency_sim.cancel()
