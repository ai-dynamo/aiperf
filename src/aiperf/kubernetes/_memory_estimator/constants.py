# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Calibration constants for the memory-estimation model.

All values are static (formulas derived from code inspection, not runtime
profiling). Constants can be recalibrated against real RSS measurements via
``tools/calibrate_memory_estimates.py``.
"""

from __future__ import annotations

# Python subprocess working-set baseline. Captures interpreter + core libs +
# GC + every module loaded by an AIPerf service (numpy, pandas, msgspec,
# pydantic, orjson, aiohttp, ZMQ, asyncio). Calibrated from a real-cluster
# ISL/OSL memory sweep, 2026-04-30 (cell 1: 5K conc / ISL=128 /
# OSL=128 sums to 2569 MiB across 12 containers — ~214 MiB / container at the
# minimum-workload floor; subtracting per-service deltas leaves ~150 MiB
# common Python+libs baseline).
#
# COW (copy-on-write) does not materially shrink the working-set we measure
# — ``container_memory_working_set_bytes`` is per-container anonymous + active
# file cache, and Python's heap rapidly diverges across processes once each
# starts allocating per-task state. We model parent and child the same.
_PYTHON_SUBPROCESS_BASE_MIB = 150
_PYTHON_CHILD_SUBPROCESS_BASE_MIB = 150

# Per-service base overhead beyond subprocess (ZMQ sockets, Pydantic models, etc.)
_SERVICE_BASE_MIB: dict[str, int] = {
    "system_controller": 25,
    "worker_manager": 15,
    "timing_manager": 15,
    "dataset_manager": 30,
    "records_manager": 40,
    "api_service": 20,
    "gpu_telemetry_manager": 15,
    "server_metrics_manager": 15,
    "results_sidecar": 10,
    "worker": 12,  # aiohttp client + ZMQ sockets
    "record_processor": 10,  # record parsing + ZMQ sockets
    "worker_group_manager": 10,
}

# ZMQ proxy memory: 3 proxies (event_bus, dataset_manager, raw_inference)
_ZMQ_PROXY_MIB = 5
_NUM_ZMQ_PROXIES = 3

# RecordsManager: per-worker tracking overhead in RecordsTracker
# (WorkerProcessingStats per worker, not per request)
_BYTES_PER_WORKER_TRACKING = 256

# Wrapper-class overhead atop a numpy-backed time-series array
# (``MetricArray``, ``GpuMetricTimeSeries``, ``ScalarTimeSeries``,
# ``HistogramTimeSeries``). Calibrated via
# ``tools/rebaseline_memory_constants.py`` — at fully-filled
# capacity the wrapper class adds ~0-2% above the raw numpy bytes (dict
# of metric names, bucket-le tuple, sum tracker, etc.).
#
# Historical note: this constant used to be 1.3 to model "doubling
# strategy waste". That waste is now captured by ``_ceil_pow2(N)`` in
# the formula's capacity calculation, so the multiplier is only for
# wrapper overhead. 1.05 keeps a small safety margin without
# double-counting capacity slack.
_GROWABLE_ARRAY_OVERHEAD = 1.05

# Numpy element sizes
_FLOAT64_BYTES = 8
_INT64_BYTES = 8

# HuggingFace tokenizer cache per distinct model
_TOKENIZER_CACHE_MIB = 150

# aiohttp connection pool: per-connection kernel + userspace buffers
_BYTES_PER_CONNECTION = 1024

# Per-request base overhead: msgspec.Struct RequestRecord shell + metadata
# fields, with empty turns/responses lists.
# Calibrated via ``tools/rebaseline_memory_constants.py``;
# pympler-measured deep size of an empty RequestRecord is 504 B.
_REQUEST_RECORD_BASE_BYTES = 504

# SSE streaming: per output token, each creates an SSEField dataclass plus
# a JSON chunk string.
# Calibrated via the rebaseline script with **unique** chunk values (so
# Python string interning doesn't deduplicate chunks). Linear fit on
# SSEMessage(OSL=0..4096) yields per-chunk = 152 B (SSEField + small JSON
# value string), base = 136 B (SSEMessage + empty packets list).
_SSE_MESSAGE_BASE_BYTES = 136  # SSEMessage + list overhead
_SSE_BYTES_PER_CHUNK = 152  # SSEField object + short unique JSON string

# Non-streaming: single TextResponse with full JSON body.
# Calibrated by the rebaseline script: empty TextResponse = 152 B,
# plus ~4 B/token for the response body (chars-per-token in synthetic
# OpenAI-shaped JSON).
_TEXT_RESPONSE_BASE_BYTES = 152  # TextResponse msgspec overhead
_TEXT_RESPONSE_BYTES_PER_TOKEN = 4  # ~4 chars per token in response body

# Turn (prompt) storage per in-flight request: Turn msgspec.Struct +
# Text(contents=[...]) + the prompt string itself.
# Calibrated by the rebaseline script: Turn(role="user", texts=[]) = 408 B,
# plus ~4 B/token for the text content.
_TURN_BASE_BYTES = 408  # Turn + Text msgspec overhead
_TURN_BYTES_PER_TOKEN = 4  # ~4 chars per input token

# Multi-turn session state: per-token in conversation history
_BYTES_PER_SESSION_TOKEN = 4

# Mmap index entry per conversation
_MMAP_INDEX_ENTRY_BYTES = 16

# Default DCGM metrics per GPU
_DEFAULT_GPU_METRICS = 12

# Default Prometheus scrape interval (seconds)
_DEFAULT_SCRAPE_INTERVAL_S = 5.0

# Default unique metric series per Prometheus endpoint (scalar + histogram)
_DEFAULT_UNIQUE_METRIC_SERIES = 200
_DEFAULT_HISTOGRAM_METRICS = 20
_DEFAULT_HISTOGRAM_BUCKETS = 10

# Safety margin multipliers
_STEADY_STATE_MARGIN = 1.2  # 20% headroom for request recommendation
_PEAK_MARGIN = 1.3  # 30% headroom for limit recommendation
_HEADROOM_WARNING_PCT = 15.0  # warn below 15% headroom
_RECORDS_MANAGER_WARN_PCT = 50.0  # warn when RM uses >50% of limit

# "Adequate headroom" threshold for the recommendation that the current
# limits comfortably hold this workload. Tuned to 30% (vs the old 50%)
# because realistic per-process Python baselines (~150 MiB) consume a
# meaningful fraction of even small workloads' limits — at default
# benchmarks the typical headroom sits in the 30-50% band, which is
# adequate, not borderline.
_ADEQUATE_HEADROOM_PCT = 30.0

# Standard metrics computed per record (TTFT, TPOT, ITL, E2E, throughput, etc.)
_DEFAULT_NUM_STANDARD_METRICS = 25

# Conservative default request count when a phase declares neither
# ``requests`` nor ``duration`` nor ``sessions`` (e.g. open-loop trace replay
# without an explicit cap). Used by ``_estimate_phase_requests`` /
# ``_estimate_phase_duration`` to bound memory math; underestimating here
# under-allocates request-shaped buffers, so we err on the high side.
_DEFAULT_PHASE_REQUEST_COUNT = 1000

# Assumed average per-request latency (seconds) when estimating duration of a
# concurrency-driven phase (no explicit ``duration`` and no ``rate``). Multiplied
# against ``requests / max(concurrency, 1)`` to get phase wall time. Calibrated
# against streaming LLM workloads where an end-to-end request typically takes
# 1-3s; 2.0 sits in the middle.
_PHASE_AVG_SEC_PER_REQUEST = 2.0
