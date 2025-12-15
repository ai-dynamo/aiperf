# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Metric update utilities to reduce duplication in endpoint handlers."""

import time
from collections import deque
from collections.abc import Callable
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aiperf_mock_server.utils import RequestCtx

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf_mock_server.metrics import (
    COMPLETION_TOKENS_TOTAL,
    # Dynamo decode metrics
    DYNAMO_DECODE_INFLIGHT_REQUESTS,
    DYNAMO_DECODE_KVSTATS_ACTIVE_BLOCKS,
    DYNAMO_DECODE_KVSTATS_GPU_CACHE_USAGE_PERCENT,
    DYNAMO_DECODE_KVSTATS_TOTAL_BLOCKS,
    DYNAMO_DECODE_REQUEST_DURATION_SECONDS,
    DYNAMO_DECODE_REQUESTS,
    # Dynamo frontend metrics
    DYNAMO_FRONTEND_DISCONNECTED_CLIENTS,
    DYNAMO_FRONTEND_INFLIGHT_REQUESTS,
    DYNAMO_FRONTEND_INPUT_SEQUENCE_TOKENS,
    DYNAMO_FRONTEND_INTER_TOKEN_LATENCY_SECONDS,
    DYNAMO_FRONTEND_MODEL_CONTEXT_LENGTH,
    DYNAMO_FRONTEND_MODEL_KV_CACHE_BLOCK_SIZE,
    DYNAMO_FRONTEND_MODEL_TOTAL_KV_BLOCKS,
    DYNAMO_FRONTEND_OUTPUT_SEQUENCE_TOKENS,
    DYNAMO_FRONTEND_OUTPUT_TOKENS,
    DYNAMO_FRONTEND_QUEUED_REQUESTS,
    DYNAMO_FRONTEND_REQUEST_DURATION_SECONDS,
    DYNAMO_FRONTEND_REQUESTS,
    DYNAMO_FRONTEND_TIME_TO_FIRST_TOKEN_SECONDS,
    # Dynamo prefill metrics
    DYNAMO_PREFILL_INFLIGHT_REQUESTS,
    DYNAMO_PREFILL_KVSTATS_ACTIVE_BLOCKS,
    DYNAMO_PREFILL_KVSTATS_GPU_CACHE_USAGE_PERCENT,
    DYNAMO_PREFILL_KVSTATS_TOTAL_BLOCKS,
    DYNAMO_PREFILL_REQUEST_DURATION_SECONDS,
    DYNAMO_PREFILL_REQUESTS,
    # AIPerf mock metrics
    EMBEDDINGS_GENERATED_TOTAL,
    ERRORS_TOTAL,
    INTER_TOKEN_LATENCY_SECONDS,
    PASSAGES_RANKED_TOTAL,
    PROMPT_TOKENS_TOTAL,
    RANKINGS_GENERATED_TOTAL,
    REQUEST_BYTES_TOTAL,
    REQUEST_LATENCY_SECONDS,
    REQUESTS_BY_MODEL,
    REQUESTS_IN_PROGRESS,
    REQUESTS_TOTAL,
    RESPONSE_BYTES_TOTAL,
    # SGLang metrics
    SGLANG_CACHE_HIT_RATE,
    SGLANG_E2E_REQUEST_LATENCY_SECONDS,
    SGLANG_GEN_THROUGHPUT,
    SGLANG_NUM_QUEUE_REQS,
    SGLANG_NUM_RUNNING_REQS,
    SGLANG_NUM_USED_TOKENS,
    SGLANG_QUEUE_TIME_SECONDS,
    SGLANG_TIME_TO_FIRST_TOKEN_SECONDS,
    SGLANG_TOKEN_USAGE,
    TIME_TO_FIRST_TOKEN_SECONDS,
    TOKENS_PER_REQUEST,
    TOKENS_STREAMED_TOTAL,
    # TRT-LLM metrics
    TRTLLM_E2E_REQUEST_LATENCY_SECONDS,
    TRTLLM_REQUEST_QUEUE_TIME_SECONDS,
    TRTLLM_REQUEST_SUCCESS,
    TRTLLM_TIME_PER_OUTPUT_TOKEN_SECONDS,
    TRTLLM_TIME_TO_FIRST_TOKEN_SECONDS,
    # vLLM metrics
    VLLM_E2E_REQUEST_LATENCY_SECONDS,
    VLLM_GENERATION_TOKENS,
    VLLM_INTER_TOKEN_LATENCY_SECONDS,
    VLLM_ITERATION_TOKENS_TOTAL,
    VLLM_KV_CACHE_USAGE,
    VLLM_NUM_PREEMPTIONS,
    VLLM_NUM_REQUESTS_RUNNING,
    VLLM_NUM_REQUESTS_WAITING,
    VLLM_PREFIX_CACHE_HITS,
    VLLM_PREFIX_CACHE_QUERIES,
    VLLM_PROMPT_TOKENS,
    VLLM_REQUEST_QUEUE_TIME_SECONDS,
    VLLM_REQUEST_SUCCESS,
    VLLM_TIME_TO_FIRST_TOKEN_SECONDS,
)

logger = AIPerfLogger(__name__)


@dataclass(slots=True)
class LLMLatencyInfo:
    """Latency measurements for LLM requests."""

    e2e_latency: float
    prefill_duration: float
    decode_duration: float


def record_request_start(endpoint: str, model: str) -> None:
    """Record metrics at request start."""
    REQUESTS_IN_PROGRESS.labels(endpoint=endpoint).inc()
    REQUESTS_BY_MODEL.labels(model=model, endpoint=endpoint).inc()


def record_request_end(endpoint: str) -> None:
    """Decrement in-progress gauge at request end."""
    REQUESTS_IN_PROGRESS.labels(endpoint=endpoint).dec()


# Track inflight count for KV cache simulation
_inflight_count = 0
_total_kv_blocks = 1024  # Simulated total KV cache blocks

# Token throughput tracking for DCGM load
# Uses batched flushing to handle high throughput (500k+ tokens/sec)
_token_buckets: deque[tuple[float, int]] = deque()  # (timestamp, count) buckets
_throughput_window_sec = 1.0  # Configurable sliding window
_flush_interval_sec = 0.01  # Flush every 10ms → max 100 buckets/sec
_token_buffer = 0  # Accumulates tokens between flushes
_last_flush_time = 0.0
_min_throughput_baseline = 100  # Floor for load calculation (configurable)
_max_observed_throughput = 0.0  # Auto-tracked peak throughput

# Callback to update DCGM faker load (set by app.py during startup)
_dcgm_load_callback: Callable[[float], None] | None = None


def register_dcgm_load_callback(
    callback: Callable[[float], None],
    min_throughput: int = 100,
    window_sec: float = 1.0,
) -> None:
    """Register a callback to update DCGM faker load based on token throughput."""
    global _dcgm_load_callback, _min_throughput_baseline, _throughput_window_sec
    _dcgm_load_callback = callback
    _min_throughput_baseline = min_throughput
    _throughput_window_sec = window_sec


def _record_tokens(count: int) -> None:
    """Record token generation for throughput tracking (batched for efficiency)."""
    global _token_buffer, _last_flush_time
    _token_buffer += count

    now = time.monotonic()
    if now - _last_flush_time >= _flush_interval_sec:
        _flush_tokens(now)


def _flush_tokens(now: float) -> None:
    """Flush buffered tokens to bucket and update DCGM load."""
    global _token_buffer, _last_flush_time

    if _token_buffer > 0:
        _token_buckets.append((now, _token_buffer))
        _token_buffer = 0

    _last_flush_time = now

    # Prune old buckets
    cutoff = now - _throughput_window_sec
    while _token_buckets and _token_buckets[0][0] < cutoff:
        _token_buckets.popleft()

    _update_dcgm_load()


def _get_throughput() -> float:
    """Get current token throughput (tokens/sec)."""
    if not _token_buckets:
        return 0.0
    total = sum(count for _, count in _token_buckets)
    return total / _throughput_window_sec


def _update_dcgm_load() -> None:
    """Update DCGM faker load based on token throughput (auto-scaling)."""
    global _max_observed_throughput
    if _dcgm_load_callback is None:
        return

    throughput = _get_throughput()

    # Track peak throughput
    if throughput > _max_observed_throughput:
        _max_observed_throughput = throughput

    # Use max of observed peak and configured minimum baseline
    effective_max = max(_max_observed_throughput, _min_throughput_baseline)
    load = min(1.0, throughput / effective_max)
    _dcgm_load_callback(load)


def _update_kv_cache_gauges(model: str) -> None:
    """Update KV cache gauges based on inflight requests."""
    # Simulate KV cache usage: each inflight request uses ~10 blocks
    active_blocks = min(_inflight_count * 10, _total_kv_blocks)
    usage_percent = active_blocks / _total_kv_blocks if _total_kv_blocks > 0 else 0.0

    # vLLM KV cache
    VLLM_KV_CACHE_USAGE.set(usage_percent)

    # SGLang token usage and cache hit rate
    SGLANG_TOKEN_USAGE.set(usage_percent)
    SGLANG_CACHE_HIT_RATE.set(0.3)  # Simulate 30% cache hit rate

    # Dynamo prefill KV stats
    DYNAMO_PREFILL_KVSTATS_ACTIVE_BLOCKS.set(active_blocks)
    DYNAMO_PREFILL_KVSTATS_TOTAL_BLOCKS.set(_total_kv_blocks)
    DYNAMO_PREFILL_KVSTATS_GPU_CACHE_USAGE_PERCENT.set(usage_percent)

    # Dynamo decode KV stats
    DYNAMO_DECODE_KVSTATS_ACTIVE_BLOCKS.set(active_blocks)
    DYNAMO_DECODE_KVSTATS_TOTAL_BLOCKS.set(_total_kv_blocks)
    DYNAMO_DECODE_KVSTATS_GPU_CACHE_USAGE_PERCENT.set(usage_percent)


def record_llm_inflight_start(model: str) -> None:
    """Increment all LLM-backend inflight/running gauges."""
    global _inflight_count
    _inflight_count += 1

    VLLM_NUM_REQUESTS_RUNNING.inc()
    VLLM_NUM_REQUESTS_WAITING.set(0)  # Mock server has no queue
    SGLANG_NUM_RUNNING_REQS.inc()
    SGLANG_NUM_QUEUE_REQS.set(0)  # Mock server has no queue
    DYNAMO_FRONTEND_INFLIGHT_REQUESTS.labels(model=model).inc()
    DYNAMO_FRONTEND_QUEUED_REQUESTS.labels(model=model).set(0)
    DYNAMO_PREFILL_INFLIGHT_REQUESTS.labels(
        dynamo_endpoint="generate", model=model
    ).inc()
    DYNAMO_DECODE_INFLIGHT_REQUESTS.labels(
        dynamo_endpoint="generate", model=model
    ).inc()

    _update_kv_cache_gauges(model)


def record_llm_inflight_end(model: str) -> None:
    """Decrement all LLM-backend inflight/running gauges."""
    global _inflight_count
    _inflight_count = max(0, _inflight_count - 1)

    VLLM_NUM_REQUESTS_RUNNING.dec()
    SGLANG_NUM_RUNNING_REQS.dec()
    DYNAMO_FRONTEND_INFLIGHT_REQUESTS.labels(model=model).dec()
    DYNAMO_PREFILL_INFLIGHT_REQUESTS.labels(
        dynamo_endpoint="generate", model=model
    ).dec()
    DYNAMO_DECODE_INFLIGHT_REQUESTS.labels(
        dynamo_endpoint="generate", model=model
    ).dec()

    _update_kv_cache_gauges(model)


def record_token_metrics(endpoint: str, model: str, usage: dict[str, Any]) -> None:
    """Record token count metrics."""
    PROMPT_TOKENS_TOTAL.labels(endpoint=endpoint, model=model).inc(
        usage["prompt_tokens"]
    )
    COMPLETION_TOKENS_TOTAL.labels(endpoint=endpoint, model=model).inc(
        usage["completion_tokens"]
    )
    TOKENS_PER_REQUEST.labels(endpoint=endpoint, token_type="prompt").observe(
        usage["prompt_tokens"]
    )
    TOKENS_PER_REQUEST.labels(endpoint=endpoint, token_type="completion").observe(
        usage["completion_tokens"]
    )
    _record_tokens(usage["completion_tokens"])  # Update throughput for DCGM load


def record_basic_success(endpoint: str, latency: float) -> None:
    """Record basic success metrics (latency, request count)."""
    REQUESTS_TOTAL.labels(endpoint=endpoint, method="POST", status="200").inc()
    REQUEST_LATENCY_SECONDS.labels(endpoint=endpoint).observe(latency)


def record_llm_backend_success(latency: float, usage: dict[str, Any]) -> None:
    """Record vLLM/SGLang/TRT-LLM success metrics."""
    # vLLM metrics
    VLLM_E2E_REQUEST_LATENCY_SECONDS.observe(latency)
    VLLM_PROMPT_TOKENS.inc(usage["prompt_tokens"])
    VLLM_GENERATION_TOKENS.inc(usage["completion_tokens"])
    VLLM_REQUEST_SUCCESS.inc()
    VLLM_ITERATION_TOKENS_TOTAL.observe(usage["total_tokens"])
    VLLM_REQUEST_QUEUE_TIME_SECONDS.observe(0.0)  # No queue in mock server
    VLLM_PREFIX_CACHE_QUERIES.inc(usage["prompt_tokens"])
    # Simulate ~30% cache hit rate
    VLLM_PREFIX_CACHE_HITS.inc(int(usage["prompt_tokens"] * 0.3))

    # SGLang metrics
    SGLANG_E2E_REQUEST_LATENCY_SECONDS.observe(latency)
    SGLANG_QUEUE_TIME_SECONDS.observe(0.0)  # No queue in mock server
    # Update throughput gauge (tokens/sec for this request)
    if latency > 0:
        SGLANG_GEN_THROUGHPUT.set(usage["completion_tokens"] / latency)
    SGLANG_NUM_USED_TOKENS.inc(usage["total_tokens"])

    # TRT-LLM metrics
    TRTLLM_E2E_REQUEST_LATENCY_SECONDS.observe(latency)
    TRTLLM_REQUEST_SUCCESS.inc()
    TRTLLM_REQUEST_QUEUE_TIME_SECONDS.observe(0.0)  # No queue in mock server


def record_dynamo_success(
    model: str, latency: float, usage: dict[str, Any], latency_info: LLMLatencyInfo
) -> None:
    """Record Dynamo frontend/prefill/decode success metrics."""
    # Frontend metrics
    DYNAMO_FRONTEND_REQUEST_DURATION_SECONDS.labels(model=model).observe(latency)
    DYNAMO_FRONTEND_REQUESTS.labels(model=model).inc()
    DYNAMO_FRONTEND_INPUT_SEQUENCE_TOKENS.labels(model=model).inc(
        usage["prompt_tokens"]
    )
    DYNAMO_FRONTEND_OUTPUT_TOKENS.labels(model=model).inc(usage["completion_tokens"])
    DYNAMO_FRONTEND_OUTPUT_SEQUENCE_TOKENS.labels(model=model).inc(
        usage["completion_tokens"]
    )

    # Prefill metrics
    DYNAMO_PREFILL_REQUEST_DURATION_SECONDS.labels(
        dynamo_endpoint="generate", model=model
    ).observe(latency_info.prefill_duration)
    DYNAMO_PREFILL_REQUESTS.labels(dynamo_endpoint="generate", model=model).inc()

    # Decode metrics
    DYNAMO_DECODE_REQUEST_DURATION_SECONDS.labels(
        dynamo_endpoint="generate", model=model
    ).observe(latency_info.decode_duration)
    DYNAMO_DECODE_REQUESTS.labels(dynamo_endpoint="generate", model=model).inc()


def record_llm_success(
    endpoint: str,
    model: str,
    latency: float,
    usage: dict[str, Any],
    latency_info: LLMLatencyInfo,
) -> None:
    """Record all success metrics for LLM endpoints (chat/completions)."""
    record_token_metrics(endpoint, model, usage)
    record_basic_success(endpoint, latency)
    record_llm_backend_success(latency, usage)
    record_dynamo_success(model, latency, usage, latency_info)


def record_embedding_success(
    endpoint: str, model: str, prompt_tokens: int, num_embeddings: int, latency: float
) -> None:
    """Record embedding success metrics."""
    PROMPT_TOKENS_TOTAL.labels(endpoint=endpoint, model=model).inc(prompt_tokens)
    EMBEDDINGS_GENERATED_TOTAL.labels(model=model).inc(num_embeddings)
    record_basic_success(endpoint, latency)


def record_ranking_success(
    endpoint: str, model: str, prompt_tokens: int, num_passages: int, latency: float
) -> None:
    """Record ranking success metrics."""
    PROMPT_TOKENS_TOTAL.labels(endpoint=endpoint, model=model).inc(prompt_tokens)
    RANKINGS_GENERATED_TOTAL.labels(endpoint=endpoint).inc()
    PASSAGES_RANKED_TOTAL.labels(endpoint=endpoint).inc(num_passages)
    record_basic_success(endpoint, latency)


def record_tgi_success(endpoint: str, usage: dict[str, Any], latency: float) -> None:
    """Record HuggingFace TGI success metrics."""
    PROMPT_TOKENS_TOTAL.labels(endpoint=endpoint, model="tgi").inc(
        usage["prompt_tokens"]
    )
    COMPLETION_TOKENS_TOTAL.labels(endpoint=endpoint, model="tgi").inc(
        usage["completion_tokens"]
    )
    record_basic_success(endpoint, latency)


def record_error(endpoint: str, error: Exception) -> None:
    """Record error metrics."""
    REQUESTS_TOTAL.labels(endpoint=endpoint, method="POST", status="500").inc()
    ERRORS_TOTAL.labels(endpoint=endpoint, error_type=type(error).__name__).inc()


def record_ttft(endpoint: str, model: str, ttft: float) -> None:
    """Record time-to-first-token metrics across all backends."""
    TIME_TO_FIRST_TOKEN_SECONDS.labels(endpoint=endpoint).observe(ttft)
    VLLM_TIME_TO_FIRST_TOKEN_SECONDS.observe(ttft)
    SGLANG_TIME_TO_FIRST_TOKEN_SECONDS.observe(ttft)
    TRTLLM_TIME_TO_FIRST_TOKEN_SECONDS.observe(ttft)
    DYNAMO_FRONTEND_TIME_TO_FIRST_TOKEN_SECONDS.labels(model=model).observe(ttft)


def record_itl(endpoint: str, model: str, itl: float) -> None:
    """Record inter-token-latency metrics across all backends."""
    INTER_TOKEN_LATENCY_SECONDS.labels(endpoint=endpoint).observe(itl)
    VLLM_INTER_TOKEN_LATENCY_SECONDS.observe(itl)
    TRTLLM_TIME_PER_OUTPUT_TOKEN_SECONDS.observe(itl)
    DYNAMO_FRONTEND_INTER_TOKEN_LATENCY_SECONDS.labels(model=model).observe(itl)


def record_streamed_token(endpoint: str, model: str) -> None:
    """Record a streamed token metric."""
    TOKENS_STREAMED_TOTAL.labels(endpoint=endpoint, model=model).inc()
    _record_tokens(1)  # Update throughput for DCGM load


def record_request_bytes(
    endpoint: str, request_bytes: int, response_bytes: int
) -> None:
    """Record request and response byte counts."""
    REQUEST_BYTES_TOTAL.labels(endpoint=endpoint).inc(request_bytes)
    RESPONSE_BYTES_TOTAL.labels(endpoint=endpoint).inc(response_bytes)


# Track which models have been initialized
_initialized_models: set[str] = set()


def init_model_config(model: str) -> None:
    """Initialize model config gauges (called once per model)."""
    if model in _initialized_models:
        return
    _initialized_models.add(model)

    # Set Dynamo frontend model config gauges
    DYNAMO_FRONTEND_MODEL_CONTEXT_LENGTH.labels(model=model).set(8192)
    DYNAMO_FRONTEND_MODEL_KV_CACHE_BLOCK_SIZE.labels(model=model).set(16)
    DYNAMO_FRONTEND_MODEL_TOTAL_KV_BLOCKS.labels(model=model).set(_total_kv_blocks)

    # Initialize counters that are never incremented (preemptions, disconnects)
    # These exist for completeness but mock server doesn't simulate these events
    _ = VLLM_NUM_PREEMPTIONS  # Referenced to show it's intentionally unused
    _ = DYNAMO_FRONTEND_DISCONNECTED_CLIENTS.labels(model=model)


@contextmanager
def track_request(endpoint: str, model: str):
    """Context manager for tracking request lifecycle.

    Handles: record_request_start, logging, record_error (on exception), record_request_end.
    """
    record_request_start(endpoint, model)
    try:
        yield
    except Exception as e:
        logger.error("Error in %s: %s", endpoint, e, exc_info=True)
        record_error(endpoint, e)
        raise
    finally:
        record_request_end(endpoint)


@contextmanager
def track_llm_request(ctx: "RequestCtx", model: str, endpoint: str):
    """Context manager for tracking LLM request lifecycle with automatic success metrics."""
    success = False

    record_request_start(endpoint, model)
    record_llm_inflight_start(model)
    try:
        yield
        success = True
    except Exception as e:
        logger.error("Error in %s: %s", endpoint, e, exc_info=True)
        record_error(endpoint, e)
        raise
    finally:
        if success:
            latency = time.perf_counter() - ctx.latency_sim.start_time
            latency_info = LLMLatencyInfo(
                e2e_latency=latency,
                prefill_duration=ctx.latency_sim.measured_ttft,
                decode_duration=max(0.0, latency - ctx.latency_sim.measured_ttft),
            )
            record_llm_success(endpoint, model, latency, ctx.usage, latency_info)

        record_request_end(endpoint)
        record_llm_inflight_end(model)


@asynccontextmanager
async def async_track_request(endpoint: str, model: str):
    """Async context manager for streaming request lifecycle."""
    record_request_start(endpoint, model)
    try:
        yield
    except Exception as e:
        logger.error("Error in %s: %s", endpoint, e, exc_info=True)
        record_error(endpoint, e)
        raise
    finally:
        record_request_end(endpoint)


@asynccontextmanager
async def async_track_llm_request(ctx: "RequestCtx", model: str, endpoint: str):
    """Async context manager for streaming LLM request lifecycle with automatic success metrics."""
    success = False

    record_request_start(endpoint, model)
    record_llm_inflight_start(model)
    try:
        yield
        success = True
    except Exception as e:
        logger.error("Error in %s: %s", endpoint, e, exc_info=True)
        record_error(endpoint, e)
        raise
    finally:
        if success:
            latency = time.perf_counter() - ctx.latency_sim.start_time
            latency_info = LLMLatencyInfo(
                e2e_latency=latency,
                prefill_duration=ctx.latency_sim.measured_ttft,
                decode_duration=max(0.0, latency - ctx.latency_sim.measured_ttft),
            )
            record_llm_success(endpoint, model, latency, ctx.usage, latency_info)

        record_request_end(endpoint)
        record_llm_inflight_end(model)
