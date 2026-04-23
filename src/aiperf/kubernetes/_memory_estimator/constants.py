# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Calibration constants for the memory-estimation model.

All values are static (formulas derived from code inspection, not runtime
profiling). Constants can be recalibrated against real RSS measurements via
``scripts/calibrate_memory_estimates.py``.
"""

from __future__ import annotations

# Python subprocess overhead: interpreter + core libs + GC + imports
# Control-plane subprocesses fork from SystemController and share loaded modules
# via copy-on-write, so they cost less than a fresh interpreter.
_PYTHON_SUBPROCESS_BASE_MIB = 35
# Worker/RP subprocesses share the WPM parent's module pages via COW.
# Effective private RSS is lower (~18 MiB) than a fresh process.
_PYTHON_CHILD_SUBPROCESS_BASE_MIB = 18

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

# GrowableArray overhead factor (doubling strategy).
# Calibrated: measured 1.05x-1.64x across scales, average ~1.3x.
_GROWABLE_ARRAY_OVERHEAD = 1.3

# Numpy element sizes
_FLOAT64_BYTES = 8
_INT64_BYTES = 8

# HuggingFace tokenizer cache per distinct model
_TOKENIZER_CACHE_MIB = 150

# aiohttp connection pool: per-connection kernel + userspace buffers
_BYTES_PER_CONNECTION = 1024

# Per-request base overhead: Pydantic RequestRecord shell + metadata fields.
# Calibrated: empty RequestRecord = 1.6 KiB.
_REQUEST_RECORD_BASE_BYTES = 1600

# SSE streaming: per output token, each creates an SSEField dataclass (~150 bytes
# object overhead) plus the JSON chunk string (~70 bytes).
# Calibrated: SSEField = 856 B deep including the ~70-char value string.
# We model as: base_overhead(SSEMessage shell) + OSL × per_chunk.
_SSE_MESSAGE_BASE_BYTES = 200  # SSEMessage + list overhead
_SSE_BYTES_PER_CHUNK = 200  # SSEField object + short JSON string (calibrated ~150-200B)

# Non-streaming: single TextResponse with full JSON body.
# Calibrated: ISL=2048 OSL=512 text response = 5.7 KiB total record.
# Response body ~ OSL * 4 chars + JSON wrapper.
_TEXT_RESPONSE_BASE_BYTES = 400  # TextResponse Pydantic overhead
_TEXT_RESPONSE_BYTES_PER_TOKEN = 4  # ~4 chars per token in response body

# Turn (prompt) storage per in-flight request: Turn Pydantic model + text content.
# Calibrated: Turn with ISL=512 adds ~2 KiB. Text is ISL * 4 chars.
_TURN_BASE_BYTES = 400  # Turn Pydantic overhead
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

# Standard metrics computed per record (TTFT, TPOT, ITL, E2E, throughput, etc.)
_DEFAULT_NUM_STANDARD_METRICS = 25
