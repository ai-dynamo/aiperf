# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prometheus metrics for AIPerf Mock Server.

This module provides separate metric registries for different server types:
- AIPERF_REGISTRY: AIPerf mock server metrics (/metrics)
- VLLM_REGISTRY: vLLM-compatible metrics (/vllm/metrics)
- SGLANG_REGISTRY: SGLang-compatible metrics (/sglang/metrics)
- TRTLLM_REGISTRY: TensorRT-LLM-compatible metrics (/trtllm/metrics)

Each registry can be scraped independently, allowing the mock server to
simulate different inference server metric formats.
"""

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Summary,
)

# ============================================================================
# Separate Registries for Each Server Type
# ============================================================================

# Default registry for all AIPerf mock metrics
AIPERF_MOCK_REGISTRY = CollectorRegistry()

# vLLM-compatible metrics registry
VLLM_REGISTRY = CollectorRegistry()

# SGLang-compatible metrics registry
SGLANG_REGISTRY = CollectorRegistry()

# TensorRT-LLM-compatible metrics registry
TRTLLM_REGISTRY = CollectorRegistry()

# Dynamo frontend metrics registry
DYNAMO_FRONTEND_REGISTRY = CollectorRegistry()

# Dynamo component metrics registries (separate for prefill and decode workers)
DYNAMO_PREFILL_REGISTRY = CollectorRegistry()
DYNAMO_DECODE_REGISTRY = CollectorRegistry()


# ============================================================================
# AIPerf Mock Server Metrics (AIPERF_REGISTRY)
# ============================================================================

# Total requests by endpoint and status
REQUESTS_TOTAL = Counter(
    "aiperf_mock_requests_total",
    "Total number of requests",
    ["endpoint", "method", "status"],
    registry=AIPERF_MOCK_REGISTRY,
)

# Currently processing requests (in-flight)
REQUESTS_IN_PROGRESS = Gauge(
    "aiperf_mock_requests_in_progress",
    "Number of requests currently being processed",
    ["endpoint"],
    registry=AIPERF_MOCK_REGISTRY,
)

# Request latency histogram (very granular for fast mock server)
REQUEST_LATENCY_SECONDS = Histogram(
    "aiperf_mock_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=(
        0.0001, 0.00025, 0.0005, 0.00075, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075,  0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0
    ),
    registry=AIPERF_MOCK_REGISTRY,
)  # fmt: skip

# Total prompt/input tokens
PROMPT_TOKENS_TOTAL = Counter(
    "aiperf_mock_prompt_tokens_total",
    "Total number of prompt/input tokens processed",
    ["endpoint", "model"],
    registry=AIPERF_MOCK_REGISTRY,
)

# Total completion/output tokens
COMPLETION_TOKENS_TOTAL = Counter(
    "aiperf_mock_completion_tokens_total",
    "Total number of completion/output tokens generated",
    ["endpoint", "model"],
    registry=AIPERF_MOCK_REGISTRY,
)

# Tokens per request (summary for percentiles)
TOKENS_PER_REQUEST = Summary(
    "aiperf_mock_tokens_per_request",
    "Tokens per request",
    ["endpoint", "token_type"],
    registry=AIPERF_MOCK_REGISTRY,
)

# Streaming requests
STREAMING_REQUESTS_TOTAL = Counter(
    "aiperf_mock_streaming_requests_total",
    "Total number of streaming requests",
    ["endpoint", "model"],
    registry=AIPERF_MOCK_REGISTRY,
)

# Tokens streamed
TOKENS_STREAMED_TOTAL = Counter(
    "aiperf_mock_tokens_streamed_total",
    "Total number of tokens streamed",
    ["endpoint", "model"],
    registry=AIPERF_MOCK_REGISTRY,
)

# Time to first token histogram (very granular for fast mock server)
TIME_TO_FIRST_TOKEN_SECONDS = Histogram(
    "aiperf_mock_time_to_first_token_seconds",
    "Time to first token in seconds",
    ["endpoint"],
    buckets=(
        0.00001, 0.000025, 0.00005, 0.000075, 0.0001, 0.00025, 0.0005, 0.00075, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.5
    ),
    registry=AIPERF_MOCK_REGISTRY,
)  # fmt: skip

# Inter-token latency histogram (very granular for fast mock server)
INTER_TOKEN_LATENCY_SECONDS = Histogram(
    "aiperf_mock_inter_token_latency_seconds",
    "Inter-token latency in seconds",
    ["endpoint"],
    buckets=(
        0.000001, 0.0000025, 0.000005, 0.0000075, 0.00001, 0.000025, 0.00005, 0.000075, 0.0001, 0.00025, 0.0005, 0.00075, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1,
    ),
    registry=AIPERF_MOCK_REGISTRY,
)  # fmt: skip

# Error metrics
ERRORS_TOTAL = Counter(
    "aiperf_mock_errors_total",
    "Total number of errors",
    ["endpoint", "error_type"],
    registry=AIPERF_MOCK_REGISTRY,
)

# Model metrics
REQUESTS_BY_MODEL = Counter(
    "aiperf_mock_requests_by_model_total",
    "Total requests by model",
    ["model", "endpoint"],
    registry=AIPERF_MOCK_REGISTRY,
)

# Embedding metrics
EMBEDDINGS_GENERATED_TOTAL = Counter(
    "aiperf_mock_embeddings_generated_total",
    "Total number of embeddings generated",
    ["model"],
    registry=AIPERF_MOCK_REGISTRY,
)

# Ranking metrics
RANKINGS_GENERATED_TOTAL = Counter(
    "aiperf_mock_rankings_generated_total",
    "Total number of rankings generated",
    ["endpoint"],
    registry=AIPERF_MOCK_REGISTRY,
)

PASSAGES_RANKED_TOTAL = Counter(
    "aiperf_mock_passages_ranked_total",
    "Total number of passages ranked",
    ["endpoint"],
    registry=AIPERF_MOCK_REGISTRY,
)

# Server uptime (updated on /metrics scrape)
SERVER_UPTIME_SECONDS = Gauge(
    "aiperf_mock_uptime_seconds",
    "Server uptime in seconds (updated on metrics scrape)",
    registry=AIPERF_MOCK_REGISTRY,
)

# Request/response size metrics
REQUEST_BYTES_TOTAL = Counter(
    "aiperf_mock_request_bytes_total",
    "Total number of bytes received in requests",
    ["endpoint"],
    registry=AIPERF_MOCK_REGISTRY,
)

RESPONSE_BYTES_TOTAL = Counter(
    "aiperf_mock_response_bytes_total",
    "Total number of bytes sent in responses",
    ["endpoint"],
    registry=AIPERF_MOCK_REGISTRY,
)

# ============================================================================
# vLLM-style Metrics (VLLM_REGISTRY)
# Buckets match real vLLM server exports
# ============================================================================

# vLLM standard latency buckets
_VLLM_LATENCY_BUCKETS = (
    0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0
)  # fmt: skip

# E2E request latency (vLLM-compatible name)
VLLM_E2E_REQUEST_LATENCY_SECONDS = Histogram(
    "vllm:e2e_request_latency_seconds",
    "Histogram of e2e request latency in seconds.",
    buckets=_VLLM_LATENCY_BUCKETS,
    registry=VLLM_REGISTRY,
)

# Time to first token (vLLM-compatible name)
VLLM_TIME_TO_FIRST_TOKEN_SECONDS = Histogram(
    "vllm:time_to_first_token_seconds",
    "Histogram of time to first token in seconds.",
    buckets=_VLLM_LATENCY_BUCKETS,
    registry=VLLM_REGISTRY,
)

# Inter-token latency (vLLM-compatible name)
VLLM_INTER_TOKEN_LATENCY_SECONDS = Histogram(
    "vllm:inter_token_latency_seconds",
    "Histogram of inter-token latency in seconds.",
    buckets=(
        0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0
    ),
    registry=VLLM_REGISTRY,
)  # fmt: skip

# Prompt tokens processed
VLLM_PROMPT_TOKENS = Counter(
    "vllm:prompt_tokens",
    "Number of prefill tokens processed.",
    registry=VLLM_REGISTRY,
)

# Generation tokens processed
VLLM_GENERATION_TOKENS = Counter(
    "vllm:generation_tokens",
    "Number of generation tokens processed.",
    registry=VLLM_REGISTRY,
)

# Request queue time
VLLM_REQUEST_QUEUE_TIME_SECONDS = Histogram(
    "vllm:request_queue_time_seconds",
    "Histogram of time spent in WAITING phase for request.",
    buckets=_VLLM_LATENCY_BUCKETS,
    registry=VLLM_REGISTRY,
)

# Request success count
VLLM_REQUEST_SUCCESS = Counter(
    "vllm:request_success",
    "Count of successfully processed requests.",
    registry=VLLM_REGISTRY,
)

# Number of running requests
VLLM_NUM_REQUESTS_RUNNING = Gauge(
    "vllm:num_requests_running",
    "Number of requests in model execution batches.",
    registry=VLLM_REGISTRY,
)

# Number of waiting requests
VLLM_NUM_REQUESTS_WAITING = Gauge(
    "vllm:num_requests_waiting",
    "Number of requests waiting to be processed.",
    registry=VLLM_REGISTRY,
)

# KV cache usage (ratio 0-1)
VLLM_KV_CACHE_USAGE = Gauge(
    "vllm:kv_cache_usage_perc",
    "KV-cache usage. 1 means 100 percent usage.",
    registry=VLLM_REGISTRY,
)

# Number of preemptions
VLLM_NUM_PREEMPTIONS = Counter(
    "vllm:num_preemptions",
    "Cumulative number of preemption from the engine.",
    registry=VLLM_REGISTRY,
)

# Prefix cache metrics
VLLM_PREFIX_CACHE_HITS = Counter(
    "vllm:prefix_cache_hits",
    "Prefix cache hits, in terms of number of cached tokens.",
    registry=VLLM_REGISTRY,
)

VLLM_PREFIX_CACHE_QUERIES = Counter(
    "vllm:prefix_cache_queries",
    "Prefix cache queries, in terms of number of queried tokens.",
    registry=VLLM_REGISTRY,
)

# Iteration tokens (histogram of batch sizes)
VLLM_ITERATION_TOKENS_TOTAL = Histogram(
    "vllm:iteration_tokens_total",
    "Histogram of number of tokens per engine_step.",
    buckets=(
        1.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0, 8192.0, 16384.0
    ),
    registry=VLLM_REGISTRY,
)  # fmt: skip

# ============================================================================
# SGLang-style Metrics (SGLANG_REGISTRY)
# Buckets match real SGLang server exports (exponential growth factor ~1.62)
# ============================================================================

# SGLang exponential buckets (factor ~1.62, used for latency histograms)
_SGLANG_EXPONENTIAL_BUCKETS = (
    0.001, 0.00162, 0.002624, 0.004252, 0.006887, 0.011158, 0.018075, 0.029282, 0.047437, 0.076848, 0.124494, 0.201681, 0.326723, 0.529292, 0.857453, 1.389073, 2.250299, 3.645484, 5.905685, 9.567209, 15.498879, 25.108184, 40.675258, 65.893919, 106.748148
)  # fmt: skip

# Generation throughput
SGLANG_GEN_THROUGHPUT = Gauge(
    "sglang:gen_throughput",
    "The generation throughput (token/s).",
    registry=SGLANG_REGISTRY,
)

# Queue metrics
SGLANG_NUM_QUEUE_REQS = Gauge(
    "sglang:num_queue_reqs",
    "The number of requests in the waiting queue.",
    registry=SGLANG_REGISTRY,
)

SGLANG_NUM_RUNNING_REQS = Gauge(
    "sglang:num_running_reqs",
    "The number of running requests.",
    registry=SGLANG_REGISTRY,
)

# Cache hit rate (ratio 0-1)
SGLANG_CACHE_HIT_RATE = Gauge(
    "sglang:cache_hit_rate",
    "The prefix cache hit rate.",
    registry=SGLANG_REGISTRY,
)

# Token usage
SGLANG_NUM_USED_TOKENS = Gauge(
    "sglang:num_used_tokens",
    "The number of used tokens.",
    registry=SGLANG_REGISTRY,
)

SGLANG_TOKEN_USAGE = Gauge(
    "sglang:token_usage",
    "The token usage.",
    registry=SGLANG_REGISTRY,
)

# Queue time histogram
SGLANG_QUEUE_TIME_SECONDS = Histogram(
    "sglang:queue_time_seconds",
    "Histogram of queueing time in seconds.",
    buckets=_SGLANG_EXPONENTIAL_BUCKETS,
    registry=SGLANG_REGISTRY,
)

# E2E request latency (SGLang-style)
SGLANG_E2E_REQUEST_LATENCY_SECONDS = Histogram(
    "sglang:e2e_request_latency_seconds",
    "Histogram of end to end request latency in seconds.",
    buckets=_SGLANG_EXPONENTIAL_BUCKETS,
    registry=SGLANG_REGISTRY,
)

# Time to first token (SGLang-style)
SGLANG_TIME_TO_FIRST_TOKEN_SECONDS = Histogram(
    "sglang:time_to_first_token_seconds",
    "Histogram of time to first token in seconds.",
    buckets=(
        0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0
    ),
    registry=SGLANG_REGISTRY,
)  # fmt: skip

# ============================================================================
# TensorRT-LLM-style Metrics (TRTLLM_REGISTRY)
# Buckets match real TensorRT-LLM server exports
# ============================================================================

# TRT-LLM standard latency buckets (used for e2e latency and queue time)
_TRTLLM_LATENCY_BUCKETS = (
    0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0
)  # fmt: skip

# E2E request latency
TRTLLM_E2E_REQUEST_LATENCY_SECONDS = Histogram(
    "trtllm:e2e_request_latency_seconds",
    "Histogram of end to end request latency in seconds.",
    buckets=_TRTLLM_LATENCY_BUCKETS,
    registry=TRTLLM_REGISTRY,
)

# Time to first token
TRTLLM_TIME_TO_FIRST_TOKEN_SECONDS = Histogram(
    "trtllm:time_to_first_token_seconds",
    "Histogram of time to first token in seconds.",
    buckets=(
        0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0, 160.0, 640.0, 2560.0
    ),
    registry=TRTLLM_REGISTRY,
)  # fmt: skip

# Time per output token
TRTLLM_TIME_PER_OUTPUT_TOKEN_SECONDS = Histogram(
    "trtllm:time_per_output_token_seconds",
    "Histogram of time per output token in seconds.",
    buckets=(
        0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0
    ),
    registry=TRTLLM_REGISTRY,
)  # fmt: skip

# Request queue time
TRTLLM_REQUEST_QUEUE_TIME_SECONDS = Histogram(
    "trtllm:request_queue_time_seconds",
    "Histogram of time spent in WAITING phase for request.",
    buckets=_TRTLLM_LATENCY_BUCKETS,
    registry=TRTLLM_REGISTRY,
)

# Request success count
TRTLLM_REQUEST_SUCCESS = Counter(
    "trtllm:request_success",
    "Count of successfully processed requests.",
    registry=TRTLLM_REGISTRY,
)

# ============================================================================
# Dynamo Frontend Metrics (DYNAMO_FRONTEND_REGISTRY)
# Buckets match real Dynamo frontend exports
# ============================================================================

# Dynamo frontend request duration buckets (from real export)
_DYNAMO_FRONTEND_LATENCY_BUCKETS = (
    0.0, 1.9, 3.4, 6.3, 12.0, 22.0, 40.0, 75.0, 140.0, 260.0, 480.0, 900.0
)  # fmt: skip

# Dynamo frontend TTFT buckets (from real export)
_DYNAMO_FRONTEND_TTFT_BUCKETS = (
    0.0, 0.0022, 0.0047, 0.01, 0.022, 0.047, 0.1, 0.22, 0.47, 1.0, 2.2, 4.7, 10.0, 22.0, 48.0, 100.0, 220.0, 480.0
)  # fmt: skip

# Dynamo frontend ITL buckets (from real export)
_DYNAMO_FRONTEND_ITL_BUCKETS = (
    0.0, 0.0019, 0.0035, 0.0067, 0.013, 0.024, 0.045, 0.084, 0.16, 0.3, 0.56, 1.1, 2.0
)  # fmt: skip

# Request duration histogram
DYNAMO_FRONTEND_REQUEST_DURATION_SECONDS = Histogram(
    "dynamo_frontend_request_duration_seconds",
    "Duration of LLM requests",
    ["model"],
    buckets=_DYNAMO_FRONTEND_LATENCY_BUCKETS,
    registry=DYNAMO_FRONTEND_REGISTRY,
)

# Time to first token histogram
DYNAMO_FRONTEND_TIME_TO_FIRST_TOKEN_SECONDS = Histogram(
    "dynamo_frontend_time_to_first_token_seconds",
    "Time to first token in seconds",
    ["model"],
    buckets=_DYNAMO_FRONTEND_TTFT_BUCKETS,
    registry=DYNAMO_FRONTEND_REGISTRY,
)

# Inter-token latency histogram
DYNAMO_FRONTEND_INTER_TOKEN_LATENCY_SECONDS = Histogram(
    "dynamo_frontend_inter_token_latency_seconds",
    "Inter-token latency in seconds",
    ["model"],
    buckets=_DYNAMO_FRONTEND_ITL_BUCKETS,
    registry=DYNAMO_FRONTEND_REGISTRY,
)

# Request counter
DYNAMO_FRONTEND_REQUESTS = Counter(
    "dynamo_frontend_requests",
    "Total number of requests",
    ["model"],
    registry=DYNAMO_FRONTEND_REGISTRY,
)

# Token counters
DYNAMO_FRONTEND_INPUT_SEQUENCE_TOKENS = Counter(
    "dynamo_frontend_input_sequence_tokens",
    "Total input sequence tokens",
    ["model"],
    registry=DYNAMO_FRONTEND_REGISTRY,
)

DYNAMO_FRONTEND_OUTPUT_SEQUENCE_TOKENS = Counter(
    "dynamo_frontend_output_sequence_tokens",
    "Total output sequence tokens",
    ["model"],
    registry=DYNAMO_FRONTEND_REGISTRY,
)

DYNAMO_FRONTEND_OUTPUT_TOKENS = Counter(
    "dynamo_frontend_output_tokens",
    "Total output tokens",
    ["model"],
    registry=DYNAMO_FRONTEND_REGISTRY,
)

# Queue and inflight gauges
DYNAMO_FRONTEND_QUEUED_REQUESTS = Gauge(
    "dynamo_frontend_queued_requests",
    "Number of requests in the queue",
    ["model"],
    registry=DYNAMO_FRONTEND_REGISTRY,
)

DYNAMO_FRONTEND_INFLIGHT_REQUESTS = Gauge(
    "dynamo_frontend_inflight_requests",
    "Number of requests currently being processed",
    ["model"],
    registry=DYNAMO_FRONTEND_REGISTRY,
)

DYNAMO_FRONTEND_DISCONNECTED_CLIENTS = Counter(
    "dynamo_frontend_disconnected_clients",
    "Total number of disconnected clients",
    ["model"],
    registry=DYNAMO_FRONTEND_REGISTRY,
)

# Model config info gauges
DYNAMO_FRONTEND_MODEL_CONTEXT_LENGTH = Gauge(
    "dynamo_frontend_model_context_length",
    "Maximum context length in tokens for a worker serving the model",
    ["model"],
    registry=DYNAMO_FRONTEND_REGISTRY,
)

DYNAMO_FRONTEND_MODEL_KV_CACHE_BLOCK_SIZE = Gauge(
    "dynamo_frontend_model_kv_cache_block_size",
    "KV cache block size",
    ["model"],
    registry=DYNAMO_FRONTEND_REGISTRY,
)

DYNAMO_FRONTEND_MODEL_TOTAL_KV_BLOCKS = Gauge(
    "dynamo_frontend_model_total_kv_blocks",
    "Total number of KV cache blocks",
    ["model"],
    registry=DYNAMO_FRONTEND_REGISTRY,
)

# ============================================================================
# Dynamo Prefill Worker Metrics (DYNAMO_PREFILL_REGISTRY)
# Buckets match real Dynamo component exports
# ============================================================================

# Dynamo component request duration buckets (from real export)
_DYNAMO_COMPONENT_LATENCY_BUCKETS = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0
)  # fmt: skip

# Request duration histogram
DYNAMO_PREFILL_REQUEST_DURATION_SECONDS = Histogram(
    "dynamo_component_request_duration_seconds",
    "Time spent processing requests by work handler",
    ["dynamo_endpoint", "model"],
    buckets=_DYNAMO_COMPONENT_LATENCY_BUCKETS,
    registry=DYNAMO_PREFILL_REGISTRY,
)

# Request counter
DYNAMO_PREFILL_REQUESTS = Counter(
    "dynamo_component_requests",
    "Total number of requests processed by work handler",
    ["dynamo_endpoint", "model"],
    registry=DYNAMO_PREFILL_REGISTRY,
)

# Inflight requests gauge
DYNAMO_PREFILL_INFLIGHT_REQUESTS = Gauge(
    "dynamo_component_inflight_requests",
    "Number of requests currently being processed by work handler",
    ["dynamo_endpoint", "model"],
    registry=DYNAMO_PREFILL_REGISTRY,
)

# KV stats gauges
DYNAMO_PREFILL_KVSTATS_ACTIVE_BLOCKS = Gauge(
    "dynamo_component_kvstats_active_blocks",
    "Number of active KV cache blocks currently in use",
    registry=DYNAMO_PREFILL_REGISTRY,
)

DYNAMO_PREFILL_KVSTATS_TOTAL_BLOCKS = Gauge(
    "dynamo_component_kvstats_total_blocks",
    "Total number of KV cache blocks available",
    registry=DYNAMO_PREFILL_REGISTRY,
)

DYNAMO_PREFILL_KVSTATS_GPU_CACHE_USAGE_PERCENT = Gauge(
    "dynamo_component_kvstats_gpu_cache_usage_percent",
    "GPU cache usage as a percentage (0.0-1.0)",
    registry=DYNAMO_PREFILL_REGISTRY,
)

# ============================================================================
# Dynamo Decode Worker Metrics (DYNAMO_DECODE_REGISTRY)
# Buckets match real Dynamo component exports
# ============================================================================

# Request duration histogram
DYNAMO_DECODE_REQUEST_DURATION_SECONDS = Histogram(
    "dynamo_component_request_duration_seconds",
    "Time spent processing requests by work handler",
    ["dynamo_endpoint", "model"],
    buckets=_DYNAMO_COMPONENT_LATENCY_BUCKETS,
    registry=DYNAMO_DECODE_REGISTRY,
)

# Request counter
DYNAMO_DECODE_REQUESTS = Counter(
    "dynamo_component_requests",
    "Total number of requests processed by work handler",
    ["dynamo_endpoint", "model"],
    registry=DYNAMO_DECODE_REGISTRY,
)

# Inflight requests gauge
DYNAMO_DECODE_INFLIGHT_REQUESTS = Gauge(
    "dynamo_component_inflight_requests",
    "Number of requests currently being processed by work handler",
    ["dynamo_endpoint", "model"],
    registry=DYNAMO_DECODE_REGISTRY,
)

# KV stats gauges
DYNAMO_DECODE_KVSTATS_ACTIVE_BLOCKS = Gauge(
    "dynamo_component_kvstats_active_blocks",
    "Number of active KV cache blocks currently in use",
    registry=DYNAMO_DECODE_REGISTRY,
)

DYNAMO_DECODE_KVSTATS_TOTAL_BLOCKS = Gauge(
    "dynamo_component_kvstats_total_blocks",
    "Total number of KV cache blocks available",
    registry=DYNAMO_DECODE_REGISTRY,
)

DYNAMO_DECODE_KVSTATS_GPU_CACHE_USAGE_PERCENT = Gauge(
    "dynamo_component_kvstats_gpu_cache_usage_percent",
    "GPU cache usage as a percentage (0.0-1.0)",
    registry=DYNAMO_DECODE_REGISTRY,
)
