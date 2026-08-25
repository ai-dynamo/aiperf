# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Calibration constants for the memory-estimation model.

Two distinct classes of constant live here, with different provenance and
different rules for changing them:

- **Aggregate MiB baselines** (``_PYTHON_SUBPROCESS_BASE_MIB``,
  ``_SERVICE_BASE_MIB``, ``_TOKENIZER_CACHE_MIB``, the margin multipliers)
  are calibrated against real-cluster ``container_memory_working_set_bytes``
  sweeps. Re-derive them only from a new cluster sweep.
- **Per-object BYTE constants** (``_REQUEST_RECORD_BASE_BYTES``,
  ``_TURN_BASE_BYTES``, the SSE / TextResponse terms) are measured in-process
  against the real model classes. ``tests/unit/kubernetes/test_memory_estimator.py``
  (``TestPerRequestBytesAgainstMeasuredHeap``) re-measures them with
  ``tracemalloc`` and fails if the model drifts from the objects it claims to
  describe, so no external script is required.
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
# (``ColumnStore`` columns, ``GpuMetricTimeSeries``, ``ScalarTimeSeries``,
# ``HistogramTimeSeries``). At fully-filled capacity the wrapper class adds
# ~0-2% above the raw numpy bytes (dict of metric names, bucket-le tuple,
# sum tracker, etc.).
#
# Historical note: this constant used to be 1.3 to model "doubling
# strategy waste". That waste is now captured by ``_ceil_pow2(N)`` in
# the formula's capacity calculation, so the multiplier is only for
# wrapper overhead. 1.05 keeps a small safety margin without
# double-counting capacity slack.
_GROWABLE_ARRAY_OVERHEAD = 1.05

# Numpy element sizes
_FLOAT64_BYTES = 8
_INT32_BYTES = 4
_INT64_BYTES = 8

# HuggingFace tokenizer cache per distinct model
_TOKENIZER_CACHE_MIB = 150

# aiohttp connection pool: per-connection kernel + userspace buffers
_BYTES_PER_CONNECTION = 1024

# ---------------------------------------------------------------------------
# Per-in-flight-request byte constants.
#
# Measurement method for every constant in this block: amortized marginal heap
# bytes per instance, captured with ``tracemalloc`` snapshot diffs over 1000-3000
# live instances after a warmup pass, minus the holder list's own pointer
# storage. Amortizing over many instances is required — the one-off cost of
# shared field-name strings, Pydantic validator/schema objects and interned
# literals is paid on the first instance and must not be attributed to the
# marginal one. ``sys.getsizeof`` is not usable here: it reports only the
# top-level object and misses ``__dict__`` / ``__pydantic_extra__`` and every
# referenced string.
#
# Re-measure by running ``TestPerRequestBytesAgainstMeasuredHeap`` in
# ``tests/unit/kubernetes/test_memory_estimator.py``, which drives the real
# model classes and asserts ``_per_request_bytes`` stays on the conservative
# side of the measurement.
#
# Type note (the source of the pre-2026-08 error): ``RequestRecord`` and
# ``Turn`` are Pydantic ``AIPerfBaseModel`` subclasses with ``extra="allow"``,
# so every instance carries ``__dict__`` + ``__pydantic_extra__`` +
# ``__pydantic_fields_set__`` on top of its slots. The values below were
# originally derived assuming ``msgspec.Struct`` layout and were 2.2-5.4x low.
# ``TextResponse`` / ``SSEMessage`` / ``SSEField`` are genuinely
# ``@dataclass(slots=True)`` and are much cheaper.
# ---------------------------------------------------------------------------

# Per-request base overhead: the Pydantic ``RequestRecord`` shell plus the
# ``RecordContext`` it carries, with empty turns/responses lists. Both are
# counted here because ``_per_request_bytes`` has no separate context term.
# Measured 2026-08-24: 1125 B bare record + 1896 B RecordContext, 3596 B for
# the populated pair (model_name, status, timestamps set).
_REQUEST_RECORD_BASE_BYTES = 3600

# SSE streaming. The transport appends one ``SSEMessage`` per wire chunk to
# ``RequestRecord.responses`` (``aiohttp_client``), each holding a single
# ``SSEField`` — so an OSL-token response retains OSL separate messages, not
# one message with OSL packets. The old 152 B/chunk was fitted against that
# wrong shape (one message, many packets), which measures ~112 B/chunk and is
# why the streaming path under-predicted by ~2x.
#
# Measured 2026-08-24 with **unique** chunk values (so string interning cannot
# deduplicate) over the OpenAI-compatible chunk envelope this repo's own mock
# server emits: 418.7 B/chunk slope with a ~0 B intercept. A minimal
# ``{"c":"<n>"}`` envelope measures 286 B/chunk, so the per-chunk cost is
# dominated by the provider's chunk envelope rather than by the token.
_SSE_MESSAGE_BASE_BYTES = 136  # SSEMessage + list overhead; measured intercept ~0
_SSE_BYTES_PER_CHUNK = 420  # SSEMessage + packets list + SSEField + JSON string

# Non-streaming: a single ``TextResponse`` dataclass holding the full JSON body.
# Measured 2026-08-24: 86 B empty, 4.00 B/token of body. The base is left at
# 152 B — it errs high by ~66 B, which is a negligible and safe margin, and
# lowering it would only shave accuracy off the conservative side.
_TEXT_RESPONSE_BASE_BYTES = 152  # dataclass(slots=True) shell; measured 86 B
_TEXT_RESPONSE_BYTES_PER_TOKEN = 4  # ~4 chars per token in response body

# Turn (prompt) storage per in-flight request: Pydantic ``Turn`` +
# ``Text(contents=[...])`` + the prompt string itself.
# Measured 2026-08-24: Turn+Text with an empty content string = 2235 B
# (Turn alone 1576 B, Text alone 592 B), plus 4.01 B/token of prompt text.
_TURN_BASE_BYTES = 2240  # Turn + Text Pydantic overhead
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

# RecordsManager ColumnStore layout. Keep these synchronized with
# ``MetricsAccumulator.process_record`` and ``ColumnStore``. The standard
# metric count includes the sole list-valued metric, inter_chunk_latency; the
# remaining metrics use one float64 column each.
#
# ``_COLUMN_STORE_METADATA_NUMERIC_COLUMNS`` is an ALLOCATED-column count, not
# a field count. ``process_record`` offers five numeric metadata fields
# (session_num, credit_issued_ns, request_ack_ns, cancellation_time_ns,
# turn_index) but ``ColumnStore.ingest_metadata`` allocates lazily, skipping
# any field whose value is None — so the resident column count is
# workload-dependent: 4 for the default streaming single-turn shape, 3 when
# responses are buffered (no request_ack_ns), 5 once any request is cancelled.
# 4 models the default. ``TestColumnStoreMetadataColumnDrift`` drives records
# through the real accumulator and counts allocated columns, so adding a
# metadata field that is populated by default fails there rather than silently
# under-estimating here.
_COLUMN_STORE_LIST_METRIC_COLUMNS = 1
_COLUMN_STORE_TIMESTAMP_COLUMNS = 3
_COLUMN_STORE_METADATA_NUMERIC_COLUMNS = 4
_COLUMN_STORE_METADATA_CATEGORICAL_COLUMNS = 6
_COLUMN_STORE_METADATA_BOOL_COLUMNS = 2
_COLUMN_STORE_INITIAL_CAPACITY = 1024
_RAGGED_OFFSETS_INITIAL_CAPACITY = 256

# A full-cardinality x_correlation_id intern entry includes the UUID string,
# dict slot, and integer code. Calibrated at ~6.5 MiB for 50K unique values.
_CATEGORICAL_INTERN_BYTES_PER_REQUEST = 136

# The alternate list backend keeps a single t-digest plus exact running
# scalar statistics. It is bounded independently of request and token count.
_TDIGEST_LIST_BACKEND_BYTES = 4 * 1024

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
