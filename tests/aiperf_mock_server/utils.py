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
from tests.aiperf_mock_server.config import server_config

if TYPE_CHECKING:
    from tests.aiperf_mock_server.config import MockServerConfig
from tests.aiperf_mock_server.metrics import DYNAMO_FRONTEND_DISCONNECTED_CLIENTS
from tests.aiperf_mock_server.metrics_utils import (
    get_inflight_count,
    record_itl,
    record_streamed_token,
    record_ttft,
)
from tests.aiperf_mock_server.models import (
    AnthropicMessagesRequest,
    ChatCompletionRequest,
    CohereRerankRequest,
    CompletionRequest,
    EmbeddingRequest,
    HFTEIRerankRequest,
    ImageGenerationRequest,
    RankingRequest,
    RequestT,
    ResponsesRequest,
    SolidoRAGRequest,
    TGIGenerateRequest,
)
from tests.aiperf_mock_server.request_recorder import get_global_recorder
from tests.aiperf_mock_server.tokens import (
    TokenizedText,
    _extract_osl_fingerprint,
    tokenize_request,
)
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
            from tests.aiperf_mock_server.scheduler import get_scheduler

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
            from tests.aiperf_mock_server.scheduler import get_scheduler

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
            from tests.aiperf_mock_server.scheduler import get_scheduler

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

    continuous_usage: bool = False
    """Emit cumulative usage on every streamed chunk (continuous_usage_stats)."""

    first_chunk_tokens: int = 1
    """Number of output tokens to bundle into the first streamed content chunk."""

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
    request_id = _create_request_id(request)
    _maybe_record_request(request, endpoint, request_id, model)

    return RequestCtx(
        request_id=request_id,
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
        continuous_usage=getattr(request, "continuous_usage_stats", False),
        first_chunk_tokens=max(1, getattr(request, "mock_first_chunk_tokens", 1)),
    )


def _maybe_record_request(
    request: RequestT, endpoint: str, request_id: str, model: str
) -> None:
    """Tokenize and persist the request via the in-process recorder, if enabled.

    Runs inline on the event loop. `--record-requests` forces `--workers=1`,
    so there is exactly one producer and no locking is needed.
    """
    recorder = get_global_recorder()
    if recorder is None:
        return
    fingerprint = _extract_osl_fingerprint(request)
    # TGIGenerateRequest has no `stream` field, so `/generate_stream` would
    # otherwise record as `stream: null` and never increment `streamed_count`.
    stream: bool | None = (
        True if endpoint == "/generate_stream" else getattr(request, "stream", None)
    )
    recorder.record_request(
        ts=time.time(),
        endpoint=endpoint,
        request_id=request_id,
        model=model,
        request=request,
        stream=stream,
        osl_fingerprint=fingerprint,
    )


def _create_request_id(request: RequestT) -> str:
    """Generate request ID based on request type."""
    match request:
        case AnthropicMessagesRequest():
            return f"msg_mock_{uuid.uuid4().hex}"
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
        case ResponsesRequest():
            return f"resp-{uuid.uuid4()}"
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


def _bundle_first_chunk(tokens: list[str], first_chunk_tokens: int) -> list[list[str]]:
    """Group output tokens into streamed chunks, bundling the first
    ``first_chunk_tokens`` tokens into the first chunk; the rest stream one per
    chunk. ``first_chunk_tokens=1`` reproduces one-token-per-chunk streaming."""
    if first_chunk_tokens <= 1 or len(tokens) <= 1:
        return [[t] for t in tokens]
    n = min(first_chunk_tokens, len(tokens))
    return [tokens[:n], *([t] for t in tokens[n:])]


def _partial_usage(ctx: "RequestCtx", completion_tokens: int) -> dict[str, Any]:
    """Cumulative per-chunk usage in OpenAI shape (continuous_usage_stats)."""
    prompt = ctx.usage.get("prompt_tokens", 0)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt + completion_tokens,
    }


async def stream_chat_completion(
    ctx: RequestCtx, endpoint: str, include_usage: bool
) -> AsyncGenerator[bytes, None]:
    """Stream chat completion tokens as SSE chunks."""
    has_reasoning = bool(ctx.reasoning_content_tokens)

    try:
        # Stream reasoning tokens first (if any)
        completion_so_far = 0
        for token in ctx.reasoning_content_tokens:
            await ctx.latency_sim.wait_for_next_token()
            record_streamed_token(endpoint, ctx.model)
            completion_so_far += 1
            chunk: dict[str, Any] = {
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
            if ctx.continuous_usage:
                chunk["usage"] = _partial_usage(ctx, completion_so_far)
            yield _sse(chunk)

        # Stream output tokens, bundling the first chunk when requested.
        groups = _bundle_first_chunk(ctx.tokens, ctx.first_chunk_tokens)
        num_groups = len(groups)
        for gi, group in enumerate(groups):
            for _ in group:
                await ctx.latency_sim.wait_for_next_token()
                record_streamed_token(endpoint, ctx.model)
            completion_so_far += len(group)

            delta: dict[str, Any] = {"content": "".join(group)}
            if gi == 0 and not has_reasoning:
                delta["role"] = "assistant"

            choice: dict[str, Any] = {"index": 0, "delta": delta}
            if gi == num_groups - 1:
                choice["finish_reason"] = ctx.finish_reason

            chunk = {
                "id": ctx.request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": ctx.model,
                "choices": [choice],
            }
            if ctx.continuous_usage:
                chunk["usage"] = _partial_usage(ctx, completion_so_far)
            yield _sse(chunk)

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


def _anthropic_sse(event_type: str, data: dict[str, Any]) -> bytes:
    """Format data as Anthropic SSE event bytes with event type."""
    return b"event: " + event_type.encode() + b"\ndata: " + orjson.dumps(data) + b"\n\n"


# Real Anthropic stop_reason vocabulary. The mock's token generator produces
# OpenAI-style finish reasons ("stop"/"length"); map them so the wire shape
# matches the Messages API contract (end_turn / max_tokens / tool_use).
_ANTHROPIC_STOP_REASON_MAP: dict[str, str] = {
    "stop": "end_turn",
    "length": "max_tokens",
}


def anthropic_stop_reason(finish_reason: str) -> str:
    """Map an OpenAI-style finish reason to the Anthropic stop_reason value."""
    return _ANTHROPIC_STOP_REASON_MAP.get(finish_reason, finish_reason)


# Prompt-cache simulation state: (model, prefix_hash) entries seen so far.
# Server-lifetime persistence, like a real prefix cache without TTL expiry.
# Process-local: with --workers > 1 identical requests can land on different
# processes and miss unpredictably — run cache benchmarks with --workers 1
# (the default).
_ANTHROPIC_PREFIX_CACHE: set[tuple[str, str]] = set()


def reset_anthropic_prefix_cache() -> None:
    """Clear the simulated prompt-cache (test isolation)."""
    _ANTHROPIC_PREFIX_CACHE.clear()


def simulate_anthropic_cache(
    model: str,
    system: Any,
    messages: list[dict[str, Any]],
    total_prompt_tokens: int,
) -> tuple[int, int, int]:
    """Simulate Anthropic prompt caching with DISJOINT accounting.

    Returns ``(input_tokens, cache_read_input_tokens,
    cache_creation_input_tokens)`` such that the three always sum to
    ``total_prompt_tokens`` — mirroring the real API contract where
    ``input_tokens`` counts only the uncached remainder.

    Prefix granularity is message boundaries: the longest previously-seen
    serialized ``system + messages[:k]`` prefix is served from cache
    (``cache_read``), everything after it is written (``cache_creation``,
    the request's ``cache_control`` breakpoint covers the full prompt), and
    the uncached remainder is 0. Token attribution across boundaries is
    char-proportional against the mock tokenizer's total, which keeps the
    simulation deterministic without a second tokenization pass.
    """
    import hashlib

    acc = bytearray(orjson.dumps(system) if system is not None else b"")
    boundaries: list[tuple[str, int]] = []
    for message in messages:
        acc += orjson.dumps(message)
        boundaries.append((hashlib.sha256(bytes(acc)).hexdigest(), len(acc)))

    if not boundaries or total_prompt_tokens <= 0:
        return total_prompt_tokens, 0, 0

    total_chars = boundaries[-1][1]
    read_tokens = 0
    for digest, chars in boundaries:
        if (model, digest) in _ANTHROPIC_PREFIX_CACHE:
            read_tokens = max(
                read_tokens, round(total_prompt_tokens * chars / total_chars)
            )
    for digest, _ in boundaries:
        _ANTHROPIC_PREFIX_CACHE.add((model, digest))

    read_tokens = min(read_tokens, total_prompt_tokens)
    creation_tokens = total_prompt_tokens - read_tokens
    return 0, read_tokens, creation_tokens


def _anthropic_usage(
    input_tokens: int, cache_read: int, cache_creation: int, output_tokens: int
) -> dict[str, int]:
    """Anthropic usage dict in the modern full shape (all keys present)."""
    return {
        "input_tokens": input_tokens,
        "cache_creation_input_tokens": cache_creation,
        "cache_read_input_tokens": cache_read,
        "output_tokens": output_tokens,
    }


_INPUT_JSON_FRAGMENT_CHARS = 16
"""Real servers chunk ``input_json_delta`` across many fragments; stream in
small pieces so consumers exercise the reassembly path."""


async def stream_anthropic_messages(
    ctx: RequestCtx,
    endpoint: str,
    tool_use_block: dict[str, Any] | None = None,
    cache_triple: tuple[int, int, int] | None = None,
) -> AsyncGenerator[bytes, None]:
    """Stream Anthropic Messages tokens as SSE events.

    Wire shapes mirror the real API as captured from live traffic:

    - ``message_start`` usage carries the full key set (input + cache
      read/creation + ``output_tokens: 1``).
    - ``message_delta`` usage is CUMULATIVE (all keys, final output count) by
      default — what api.anthropic.com and Dynamo emit today. Setting
      ``server_config.anthropic_split_usage`` reverts to the docs-canonical
      split shape (``output_tokens`` only) so endpoint usage-merge handling
      stays regression-tested.
    - ``cache_triple`` is the disjoint ``(input, read, creation)`` accounting
      from ``simulate_anthropic_cache``; defaults to no-cache identity.

    When ``tool_use_block`` is supplied, append a ``tool_use`` content block
    after the thinking/text blocks, streaming its ``input`` dict as chunked
    ``input_json_delta`` fragments. The final ``message_delta`` then carries
    ``stop_reason="tool_use"``.
    """
    has_thinking = bool(ctx.reasoning_content_tokens)
    total_prompt = ctx.usage["prompt_tokens"]
    input_tokens, cache_read, cache_creation = cache_triple or (total_prompt, 0, 0)

    yield _anthropic_sse(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": ctx.request_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": ctx.model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": _anthropic_usage(input_tokens, cache_read, cache_creation, 1),
            },
        },
    )

    yield _anthropic_sse("ping", {"type": "ping"})

    block_index = 0

    # Thinking blocks (if any)
    if has_thinking:
        yield _anthropic_sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": block_index,
                "content_block": {"type": "thinking", "thinking": ""},
            },
        )

        for token in ctx.reasoning_content_tokens:
            await ctx.latency_sim.wait_for_next_token()
            record_streamed_token(endpoint, ctx.model)
            yield _anthropic_sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": block_index,
                    "delta": {"type": "thinking_delta", "thinking": token},
                },
            )

        # Signature delta
        yield _anthropic_sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": block_index,
                "delta": {"type": "signature_delta", "signature": "mock-signature"},
            },
        )

        yield _anthropic_sse(
            "content_block_stop",
            {"type": "content_block_stop", "index": block_index},
        )
        block_index += 1

    # Text block
    yield _anthropic_sse(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": block_index,
            "content_block": {"type": "text", "text": ""},
        },
    )

    for token in ctx.tokens:
        await ctx.latency_sim.wait_for_next_token()
        record_streamed_token(endpoint, ctx.model)
        yield _anthropic_sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": block_index,
                "delta": {"type": "text_delta", "text": token},
            },
        )

    yield _anthropic_sse(
        "content_block_stop",
        {"type": "content_block_stop", "index": block_index},
    )

    stop_reason = anthropic_stop_reason(ctx.finish_reason)

    # Tool use block (if requested): stream the input as chunked
    # input_json_delta fragments to mirror the real Anthropic wire shape.
    if tool_use_block is not None:
        block_index += 1
        envelope: dict[str, Any] = {
            k: v for k, v in tool_use_block.items() if k != "type"
        }
        # Open with empty input; deltas fill it.
        envelope_with_empty_input = {**envelope, "input": {}}
        yield _anthropic_sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": block_index,
                "content_block": {"type": "tool_use", **envelope_with_empty_input},
            },
        )
        partial_json = orjson.dumps(tool_use_block.get("input") or {}).decode()
        for i in range(0, len(partial_json), _INPUT_JSON_FRAGMENT_CHARS):
            yield _anthropic_sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": block_index,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": partial_json[
                            i : i + _INPUT_JSON_FRAGMENT_CHARS
                        ],
                    },
                },
            )
        yield _anthropic_sse(
            "content_block_stop",
            {"type": "content_block_stop", "index": block_index},
        )
        stop_reason = "tool_use"

    output_tokens = ctx.usage["completion_tokens"]
    if server_config.anthropic_split_usage:
        delta_usage: dict[str, int] = {"output_tokens": output_tokens}
    else:
        delta_usage = _anthropic_usage(
            input_tokens, cache_read, cache_creation, output_tokens
        )
    yield _anthropic_sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": delta_usage,
        },
    )

    yield _anthropic_sse("message_stop", {"type": "message_stop"})
