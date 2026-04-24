# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure functions that estimate memory for individual services."""

from __future__ import annotations

from aiperf.kubernetes._memory_estimator.constants import (
    _BYTES_PER_CONNECTION,
    _FLOAT64_BYTES,
    _GROWABLE_ARRAY_OVERHEAD,
    _INT64_BYTES,
    _MMAP_INDEX_ENTRY_BYTES,
    _PYTHON_CHILD_SUBPROCESS_BASE_MIB,
    _PYTHON_SUBPROCESS_BASE_MIB,
    _REQUEST_RECORD_BASE_BYTES,
    _SERVICE_BASE_MIB,
    _SSE_BYTES_PER_CHUNK,
    _SSE_MESSAGE_BASE_BYTES,
    _TEXT_RESPONSE_BASE_BYTES,
    _TEXT_RESPONSE_BYTES_PER_TOKEN,
    _TOKENIZER_CACHE_MIB,
    _TURN_BASE_BYTES,
    _TURN_BYTES_PER_TOKEN,
)
from aiperf.kubernetes._memory_estimator.estimates import ComponentEstimate
from aiperf.kubernetes._memory_estimator.utils import _ceil_pow2, _mib


def _estimate_records_manager(
    total_requests: int, num_metrics: int
) -> ComponentEstimate:
    """RecordsManager: accumulates all metric records + GrowableArrays."""
    base = _SERVICE_BASE_MIB["records_manager"] + _PYTHON_SUBPROCESS_BASE_MIB

    # GrowableArray per metric: ceil_pow2(N) × float64 × overhead
    # This is the dominant cost: one GrowableArray per metric tag, each holding
    # one float64 per request. Calibrated overhead: 1.3x (doubling waste).
    capacity = _ceil_pow2(total_requests)
    arrays_bytes = num_metrics * capacity * _FLOAT64_BYTES * _GROWABLE_ARRAY_OVERHEAD
    arrays_mib = _mib(arrays_bytes)

    # Per-worker tracking overhead (small, scales with workers not requests)
    # RecordsTracker stores WorkerProcessingStats per worker — negligible
    tracker_mib = 1.0  # ~1 MiB for all worker tracking structures

    variable = arrays_mib + tracker_mib
    peak = base + variable * 1.1  # 10% finalization overhead

    warning = None
    if variable > 500:
        warning = (
            f"RecordsManager variable memory is {variable:.0f} MiB. "
            f"At {total_requests:,} requests with {num_metrics} metrics, "
            "consider reducing request count or enabling streaming export."
        )

    return ComponentEstimate(
        name="RecordsManager",
        base_mib=base,
        variable_mib=variable,
        peak_mib=peak,
        formula=f"base({base}) + "
        f"{num_metrics} metrics x ceil_pow2({total_requests:,}) x 8B x {_GROWABLE_ARRAY_OVERHEAD}",
        dominant_factor=f"{total_requests:,} total requests",
        warning=warning,
    )


def _estimate_dataset_manager(
    dataset_count: int, avg_isl: int, avg_osl: int, max_turns: int
) -> ComponentEstimate:
    """DatasetManager: generates dataset, then steady-state is mmap metadata only.

    During generation, the full dataset is held as Pydantic Conversation objects.
    Each conversation includes Turn models, Text models, and the prompt string.
    The per-token cost is higher than raw chars due to Pydantic model overhead
    and Python string interning behavior.

    Calibrated: ISL=100K OSL=73K with 100 entries → ~297 MiB PSS.
    That's ~(100K + 73K) × 100 × ~17 bytes/token effective cost.
    The overhead comes from: Pydantic model wrappers (~1 KiB per Turn),
    Python string object headers (~50 bytes per string), and the
    tokenizer's token-to-text expansion being > 4 chars for synthetic data.
    """
    base = _SERVICE_BASE_MIB["dataset_manager"] + _PYTHON_SUBPROCESS_BASE_MIB

    # Generation peak: full dataset as Pydantic Conversation objects.
    # Per-turn: Turn model (~1 KiB) + prompt string (ISL * ~8 bytes effective).
    # The 8 bytes/token accounts for: 4 chars/token average + Python string overhead
    # + Pydantic model wrapper amortized across the token count.
    # Calibrated against ISL=100K: predicted 100K*8 = 800KB/turn, measured ~2.7 MiB/turn
    # (Pydantic deep copy + conversation wrapper adds ~3x). Use 16 bytes/token.
    _BYTES_PER_TOKEN_IN_DATASET = 16
    _TURN_OVERHEAD_BYTES = 1500  # Turn + Text + Conversation Pydantic overhead
    bytes_per_turn = (
        _TURN_OVERHEAD_BYTES + (avg_isl + avg_osl) * _BYTES_PER_TOKEN_IN_DATASET
    )
    bytes_per_conversation = max_turns * bytes_per_turn
    gen_peak_bytes = dataset_count * bytes_per_conversation
    gen_peak_mib = _mib(gen_peak_bytes)

    # Steady state: just mmap index metadata
    index_bytes = dataset_count * _MMAP_INDEX_ENTRY_BYTES
    steady_variable = _mib(index_bytes)

    peak = base + gen_peak_mib

    return ComponentEstimate(
        name="DatasetManager",
        base_mib=base,
        variable_mib=steady_variable,
        peak_mib=peak,
        formula=f"steady: base({base}) + {dataset_count:,} x {_MMAP_INDEX_ENTRY_BYTES}B index | "
        f"peak: {dataset_count:,} conv x {max_turns} turns x "
        f"({avg_isl}+{avg_osl}) tok x 16B",
        dominant_factor=f"{dataset_count:,} conversations (peak during generation)",
    )


def _per_request_bytes(avg_isl: int, avg_osl: int, *, streaming: bool) -> int:
    """Bytes held in memory for a single in-flight RequestRecord.

    Shared by worker and record-processor estimators so the two stay in sync.
    """
    turn_bytes = _TURN_BASE_BYTES + avg_isl * _TURN_BYTES_PER_TOKEN
    if streaming:
        response_bytes = _SSE_MESSAGE_BASE_BYTES + avg_osl * _SSE_BYTES_PER_CHUNK
    else:
        response_bytes = (
            _TEXT_RESPONSE_BASE_BYTES + avg_osl * _TEXT_RESPONSE_BYTES_PER_TOKEN
        )
    return _REQUEST_RECORD_BASE_BYTES + turn_bytes + response_bytes


def _estimate_worker(
    concurrency_per_worker: int,
    avg_osl: int,
    *,
    streaming: bool,
    max_turns: int,
    avg_isl: int,
    connections_per_worker: int,
) -> ComponentEstimate:
    """Single worker process: connection pool + in-flight requests + session cache.

    Each in-flight request holds a RequestRecord with:
    - Base record overhead (Pydantic model shell + metadata)
    - Turn(s) with prompt text (ISL × 4 chars per turn)
    - Response: SSEMessage with SSEField per token (streaming) or TextResponse (non-streaming)
    """
    base = _SERVICE_BASE_MIB["worker"] + _PYTHON_CHILD_SUBPROCESS_BASE_MIB

    # Connection pool
    pool_mib = _mib(connections_per_worker * _BYTES_PER_CONNECTION)

    per_request = _per_request_bytes(avg_isl, avg_osl, streaming=streaming)
    inflight_mib = _mib(concurrency_per_worker * per_request)

    # Session cache (multi-turn only)
    session_mib = 0.0
    if max_turns > 1:
        turn_bytes = _TURN_BASE_BYTES + avg_isl * _TURN_BYTES_PER_TOKEN
        extra_turn_bytes = (max_turns - 1) * turn_bytes
        session_mib = _mib(concurrency_per_worker * extra_turn_bytes)

    variable = pool_mib + inflight_mib + session_mib
    peak = base + variable * 1.1

    mode = "SSE" if streaming else "text"
    return ComponentEstimate(
        name="Worker",
        base_mib=base,
        variable_mib=variable,
        peak_mib=peak,
        formula=f"base({base}) + {concurrency_per_worker} inflight x "
        f"(record:{_REQUEST_RECORD_BASE_BYTES}B + turn:ISL={avg_isl}x4B + "
        f"resp:{mode}:OSL={avg_osl}x{'200' if streaming else '4'}B) + "
        f"pool({connections_per_worker}x1KB)",
        dominant_factor=f"{concurrency_per_worker} inflight x {mode} ISL={avg_isl} OSL={avg_osl}",
    )


# RP buffer constants (module-local: calibration intents, not user-tunable)
_RAW_BATCH_SIZE = 10
_EXPORT_BATCH_SIZE = 100
_EXPORT_BYTES_PER_RECORD = 1100


def _rp_buffer_mib(avg_osl: int, *, streaming: bool) -> tuple[float, float]:
    """(raw_buffer_mib, export_buffer_mib) for a single record processor."""
    per_raw_record_bytes = 1500 + avg_osl * (80 if streaming else 4)
    raw_buffer_mib = _mib(_RAW_BATCH_SIZE * per_raw_record_bytes)
    export_buffer_mib = _mib(_EXPORT_BATCH_SIZE * _EXPORT_BYTES_PER_RECORD)
    return raw_buffer_mib, export_buffer_mib


def _rp_warning(
    *,
    inflight_mib: float,
    tokenizer_mib: int,
    per_record_bytes: int,
    concurrency_per_rp: int,
    streaming: bool,
    avg_isl: int,
    avg_osl: int,
    num_models: int,
    model_word: str,
) -> str | None:
    """Build the single highest-severity warning string for an RP estimate."""
    mode = "SSE" if streaming else "text"
    if inflight_mib > 50:
        return (
            f"RP in-flight records use {inflight_mib:.0f} MiB "
            f"({concurrency_per_rp} records x {_mib(per_record_bytes):.1f} MiB each). "
            f"Driven by {mode} ISL={avg_isl} OSL={avg_osl} at concurrency {concurrency_per_rp}."
        )
    if tokenizer_mib > 450:
        return (
            f"Tokenizer cache is {tokenizer_mib} MiB ({num_models} {model_word} x "
            f"{_TOKENIZER_CACHE_MIB} MiB each). Consider reducing model count."
        )
    return None


def _estimate_record_processor(
    num_models: int,
    *,
    avg_isl: int = 512,
    avg_osl: int = 128,
    streaming: bool = True,
    concurrency_per_rp: int = 10,
) -> ComponentEstimate:
    """Single record processor: tokenizer + in-flight records + write buffers.

    The RP pulls records from ZMQ concurrently (PULL_MAX_CONCURRENCY=100K,
    effectively unbounded). Each record being processed holds the full
    RequestRecord — SSE chunks, turns, parsed response — until
    _free_record_data() runs after metrics are computed.

    The practical in-flight count is bounded by the benchmark's concurrency
    distributed across RPs: ``max_concurrency / total_rps``.
    """
    base = _SERVICE_BASE_MIB["record_processor"] + _PYTHON_CHILD_SUBPROCESS_BASE_MIB

    # Tokenizer cache (RSS-measured: GPT-2 ~73 MiB, Llama-3 ~50-100 MiB, large SP ~150 MiB)
    tokenizer_mib = num_models * _TOKENIZER_CACHE_MIB

    per_record_bytes = _per_request_bytes(avg_isl, avg_osl, streaming=streaming)
    inflight_mib = _mib(concurrency_per_rp * per_record_bytes)
    raw_buffer_mib, export_buffer_mib = _rp_buffer_mib(avg_osl, streaming=streaming)

    variable = tokenizer_mib + inflight_mib + raw_buffer_mib + export_buffer_mib
    # Peak includes burst where all concurrent records + write buffer are live.
    peak = base + variable * 1.1

    model_word = "model" if num_models == 1 else "models"
    mode = "SSE" if streaming else "text"
    warning = _rp_warning(
        inflight_mib=inflight_mib,
        tokenizer_mib=tokenizer_mib,
        per_record_bytes=per_record_bytes,
        concurrency_per_rp=concurrency_per_rp,
        streaming=streaming,
        avg_isl=avg_isl,
        avg_osl=avg_osl,
        num_models=num_models,
        model_word=model_word,
    )

    return ComponentEstimate(
        name="RecordProcessor",
        base_mib=base,
        variable_mib=variable,
        peak_mib=peak,
        formula=f"base({base}) + {num_models}{model_word[0]} x {_TOKENIZER_CACHE_MIB}M tok + "
        f"{concurrency_per_rp} inflight x {mode}(ISL={avg_isl},OSL={avg_osl}) + "
        f"buffers(raw={_RAW_BATCH_SIZE}+export={_EXPORT_BATCH_SIZE})",
        dominant_factor=f"{num_models} {model_word} tokenizer + {concurrency_per_rp} inflight {mode}",
        warning=warning,
    )


def _estimate_gpu_telemetry(
    num_gpus: int, duration_s: float, sample_interval_s: float, num_metrics: int
) -> ComponentEstimate:
    """GPU telemetry: columnar numpy arrays per GPU per metric."""
    base = _SERVICE_BASE_MIB["gpu_telemetry_manager"] + _PYTHON_SUBPROCESS_BASE_MIB

    if num_gpus == 0:
        return ComponentEstimate(
            name="GPU Telemetry",
            base_mib=base,
            variable_mib=0,
            peak_mib=base,
            formula="disabled (no DCGM URLs)",
            dominant_factor="N/A",
        )

    n_samples = int(duration_s / max(sample_interval_s, 0.1))
    capacity = _ceil_pow2(n_samples)

    # Per GPU: timestamps + metric arrays
    per_gpu_bytes = (capacity * _INT64_BYTES) + (
        num_metrics * capacity * _FLOAT64_BYTES
    )
    total_bytes = num_gpus * per_gpu_bytes * _GROWABLE_ARRAY_OVERHEAD
    variable = _mib(total_bytes)
    peak = base + variable

    return ComponentEstimate(
        name="GPU Telemetry",
        base_mib=base,
        variable_mib=variable,
        peak_mib=peak,
        formula=f"{num_gpus} GPUs x ({num_metrics} metrics x ceil_pow2({n_samples}) x 8B + timestamps) x 1.5",
        dominant_factor=f"{num_gpus} GPUs x {duration_s:.0f}s duration",
    )


def _estimate_server_metrics(
    num_endpoints: int,
    duration_s: float,
    scrape_interval_s: float,
    *,
    unique_series: int,
    histogram_count: int,
    histogram_buckets: int,
) -> ComponentEstimate:
    """Server metrics: scalar + histogram time series per endpoint."""
    base = _SERVICE_BASE_MIB["server_metrics_manager"] + _PYTHON_SUBPROCESS_BASE_MIB

    if num_endpoints == 0:
        return ComponentEstimate(
            name="Server Metrics",
            base_mib=base,
            variable_mib=0,
            peak_mib=base,
            formula="disabled (no Prometheus URLs)",
            dominant_factor="N/A",
        )

    n_scrapes = int(duration_s / max(scrape_interval_s, 0.1))
    capacity = _ceil_pow2(n_scrapes)

    scalar_count = max(0, unique_series - histogram_count)
    # Scalar: timestamp + value per scrape
    scalar_bytes = scalar_count * capacity * (_INT64_BYTES + _FLOAT64_BYTES)
    # Histogram: timestamp + sum + count + buckets per scrape
    hist_bytes = (
        histogram_count
        * capacity
        * (_INT64_BYTES + 2 * _FLOAT64_BYTES + histogram_buckets * _FLOAT64_BYTES)
    )
    # Fetch tracking: timestamps + latencies
    fetch_bytes = n_scrapes * _INT64_BYTES * 2

    per_endpoint = (scalar_bytes + hist_bytes + fetch_bytes) * _GROWABLE_ARRAY_OVERHEAD
    total_bytes = num_endpoints * per_endpoint
    variable = _mib(total_bytes)
    peak = base + variable

    return ComponentEstimate(
        name="Server Metrics",
        base_mib=base,
        variable_mib=variable,
        peak_mib=peak,
        formula=f"{num_endpoints} endpoints x ({scalar_count} scalar + {histogram_count} histogram) "
        f"x ceil_pow2({n_scrapes}) scrapes",
        dominant_factor=f"{num_endpoints} endpoints x {duration_s:.0f}s duration",
    )


def _estimate_fixed_service(
    name: str, display_name: str | None = None
) -> ComponentEstimate:
    """Fixed-overhead services (SystemController, WorkerManager, TimingManager, API, WPM)."""
    base = _SERVICE_BASE_MIB.get(name, 20) + _PYTHON_SUBPROCESS_BASE_MIB
    return ComponentEstimate(
        name=display_name or name.replace("_", " ").title(),
        base_mib=base,
        variable_mib=0,
        peak_mib=base,
        formula=f"fixed: subprocess({_PYTHON_SUBPROCESS_BASE_MIB}) + service({_SERVICE_BASE_MIB.get(name, 20)})",
        dominant_factor="fixed overhead",
    )
