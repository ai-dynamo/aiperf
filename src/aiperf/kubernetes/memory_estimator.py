# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Memory estimation framework for AIPerf Kubernetes deployments.

Computes per-pod and cluster-wide memory estimates from an AIPerfConfig
and deployment parameters. Used by ``aiperf kube generate``, ``aiperf kube profile``,
and the operator preflight to detect OOM risk before deployment.

The model is purely static (formulas derived from code inspection, not runtime
profiling). Constants can be calibrated against real RSS measurements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from aiperf.kubernetes.environment import CONTROLLER_RESOURCE_KEYS, K8sEnvironment
from aiperf.kubernetes.utils import parse_memory_mib

if TYPE_CHECKING:
    from aiperf.config.config import AIPerfConfig
    from aiperf.config.phases import BasePhaseConfig

__all__ = [
    "ClusterMemoryEstimate",
    "ComponentEstimate",
    "MemoryEstimationParams",
    "MemoryEstimator",
    "PodEstimate",
    "estimate_memory",
]


# =============================================================================
# Constants (calibrate against real RSS measurements)
# =============================================================================

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


# =============================================================================
# Data structures
# =============================================================================


@dataclass(slots=True)
class ComponentEstimate:
    """Memory estimate for one logical component."""

    name: str
    """Display name of the component."""

    base_mib: float
    """Fixed baseline memory in MiB (subprocess + service overhead)."""

    variable_mib: float
    """Workload-dependent memory in MiB (scales with requests, tokens, etc.)."""

    peak_mib: float
    """Worst-case memory including transient spikes."""

    formula: str
    """Human-readable formula explaining how the estimate was computed."""

    dominant_factor: str
    """Primary driver of memory usage for this component."""

    warning: str | None = None
    """Optional warning when estimated usage is unusually high."""

    @property
    def steady_state_mib(self) -> float:
        return self.base_mib + self.variable_mib


@dataclass(slots=True)
class PodEstimate:
    """Aggregated memory estimate for a pod."""

    pod_type: str
    """Pod role: 'controller', 'worker', or 'operator'."""

    components: list[ComponentEstimate]
    """Per-component estimates that sum to the pod total."""

    current_limit_mib: float
    """Configured K8s memory limit for this pod type."""

    replicas: int = 1
    """Number of pod replicas of this type in the cluster."""

    @property
    def total_steady_state_mib(self) -> float:
        return sum(c.steady_state_mib for c in self.components)

    @property
    def total_peak_mib(self) -> float:
        return sum(c.peak_mib for c in self.components)

    @property
    def recommended_request_mib(self) -> int:
        return int(math.ceil(self.total_steady_state_mib * _STEADY_STATE_MARGIN))

    @property
    def recommended_limit_mib(self) -> int:
        return int(math.ceil(self.total_peak_mib * _PEAK_MARGIN))

    @property
    def headroom_pct(self) -> float:
        if self.current_limit_mib <= 0:
            return 0.0
        return (
            (self.current_limit_mib - self.total_peak_mib)
            / self.current_limit_mib
            * 100
        )

    @property
    def at_risk(self) -> bool:
        return self.headroom_pct < _HEADROOM_WARNING_PCT


@dataclass(slots=True)
class ClusterMemoryEstimate:
    """Full cluster memory estimate."""

    params: MemoryEstimationParams
    """Input parameters used to produce this estimate."""

    controller: PodEstimate
    """Memory estimate for the controller pod."""

    worker_pod: PodEstimate
    """Memory estimate for a single worker pod."""

    operator: PodEstimate
    """Memory estimate for the operator pod."""

    warnings: list[str] = field(default_factory=list)
    """Generated warnings about memory risk."""

    recommendations: list[str] = field(default_factory=list)
    """Actionable recommendations for resource tuning."""

    @property
    def total_cluster_mib(self) -> float:
        return (
            self.controller.total_steady_state_mib
            + self.worker_pod.total_steady_state_mib * self.worker_pod.replicas
            + self.operator.total_steady_state_mib
        )


@dataclass(slots=True)
class MemoryEstimationParams:
    """All parameters that influence memory usage, derived from config."""

    total_workers: int
    """Total number of worker processes across all pods."""

    workers_per_pod: int
    """Number of worker processes in each worker pod."""

    num_worker_pods: int
    """Number of worker pod replicas."""

    record_processors_per_pod: int
    """Number of record processor processes per worker pod."""

    max_concurrency: int
    """Peak in-flight request concurrency across all workers."""

    total_requests: int
    """Estimated total requests for the entire benchmark."""

    total_benchmark_duration_s: float
    """Estimated total benchmark duration in seconds."""

    dataset_count: int
    """Number of conversations in the dataset."""

    avg_isl_tokens: int
    """Average input sequence length in tokens."""

    avg_osl_tokens: int
    """Average output sequence length in tokens."""

    max_turns: int
    """Maximum conversation turns (1 for single-turn)."""

    streaming: bool
    """Whether the endpoint uses SSE streaming responses."""

    num_endpoints: int
    """Number of target inference endpoint URLs."""

    connections_per_worker: int
    """HTTP connection pool size per worker."""

    num_gpus: int
    """Estimated total GPUs across DCGM endpoints."""

    gpu_sample_interval_s: float
    """GPU telemetry sampling interval in seconds."""

    num_gpu_metrics: int
    """Number of DCGM metrics collected per GPU."""

    num_server_metrics_endpoints: int
    """Number of Prometheus server metrics endpoints."""

    server_metrics_scrape_interval_s: float
    """Prometheus scrape interval in seconds."""

    est_unique_metric_series: int
    """Estimated unique metric series per Prometheus endpoint."""

    est_histogram_metrics: int
    """Estimated histogram-type metrics per endpoint."""

    est_histogram_buckets: int
    """Number of histogram buckets per histogram metric."""

    num_models: int
    """Number of distinct model names requiring tokenizer loading."""

    num_standard_metrics: int
    """Number of standard metrics computed per record."""

    export_http_trace: bool
    """Whether HTTP trace export is enabled."""

    @classmethod
    def from_config(
        cls,
        config: AIPerfConfig,
        total_workers: int = 10,
        workers_per_pod: int | None = None,
        connections_per_worker: int = 200,
    ) -> MemoryEstimationParams:
        """Derive estimation parameters from an AIPerfConfig.

        Args:
            config: The benchmark configuration.
            total_workers: Total desired workers (from KubeOptions.workers).
            workers_per_pod: Workers per pod (None = use default).
            connections_per_worker: Connections per worker.
        """
        from aiperf.common.environment import Environment
        from aiperf.kubernetes.environment import K8sEnvironment

        wpp = (
            workers_per_pod
            or config.runtime.workers_per_pod
            or Environment.WORKER.DEFAULT_WORKERS_PER_POD
        )
        num_pods = max(1, math.ceil(total_workers / wpp))
        actual_wpp = min(total_workers, wpp)
        rp_per_pod = max(1, actual_wpp // K8sEnvironment.RECORD_PROCESSOR_SCALE_FACTOR)

        # Derive load profile from phases
        max_conc = 1
        total_req = 0
        total_dur = 0.0
        for phase in config.phases.values():
            conc = getattr(phase, "concurrency", None) or 1
            max_conc = max(max_conc, conc)
            total_req += _estimate_phase_requests(phase, conc)
            total_dur += _estimate_phase_duration(phase, conc)

        # Derive dataset params from first dataset
        ds = config.get_default_dataset()
        isl, osl, turns, count = _extract_dataset_params(ds)

        # Models
        num_models = len(config.get_model_names())

        # GPU telemetry
        num_gpu_urls = (
            len(config.gpu_telemetry.urls) if config.gpu_telemetry.enabled else 0
        )
        # Rough estimate: 1-8 GPUs per DCGM endpoint
        est_gpus = num_gpu_urls * 4 if num_gpu_urls else 0

        # Server metrics
        num_sm_urls = (
            len(config.server_metrics.urls) if config.server_metrics.enabled else 0
        )

        # HTTP trace
        export_trace = (
            "http_trace"
            in {
                fmt.value if hasattr(fmt, "value") else str(fmt)
                for fmt in config.artifacts.formats
            }
            if hasattr(config.artifacts, "formats")
            else False
        )

        return cls(
            total_workers=total_workers,
            workers_per_pod=actual_wpp,
            num_worker_pods=num_pods,
            record_processors_per_pod=rp_per_pod,
            max_concurrency=max_conc,
            total_requests=max(total_req, 1),
            total_benchmark_duration_s=max(total_dur, 60.0),
            dataset_count=count,
            avg_isl_tokens=isl,
            avg_osl_tokens=osl,
            max_turns=turns,
            streaming=config.endpoint.streaming,
            num_endpoints=len(config.endpoint.urls),
            connections_per_worker=connections_per_worker,
            num_gpus=est_gpus,
            gpu_sample_interval_s=1.0,
            num_gpu_metrics=_DEFAULT_GPU_METRICS,
            num_server_metrics_endpoints=num_sm_urls,
            server_metrics_scrape_interval_s=_DEFAULT_SCRAPE_INTERVAL_S,
            est_unique_metric_series=_DEFAULT_UNIQUE_METRIC_SERIES,
            est_histogram_metrics=_DEFAULT_HISTOGRAM_METRICS,
            est_histogram_buckets=_DEFAULT_HISTOGRAM_BUCKETS,
            num_models=num_models,
            num_standard_metrics=_DEFAULT_NUM_STANDARD_METRICS,
            export_http_trace=export_trace,
        )


# =============================================================================
# Utility
# =============================================================================


def _ceil_pow2(n: int) -> int:
    """Next power of 2 >= n (matches GrowableArray doubling behavior)."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


def _mib(byte_count: float) -> float:
    """Convert bytes to MiB."""
    return byte_count / (1024 * 1024)


def _estimate_phase_requests(phase: BasePhaseConfig, concurrency: int) -> int:
    """Estimate total requests for a phase."""
    if phase.requests is not None:
        return phase.requests
    if phase.duration is not None:
        rate = getattr(phase, "rate", None)
        if rate is not None:
            return int(phase.duration * rate)
        # For concurrency-driven phases, estimate ~10 req/sec per concurrent slot
        return int(phase.duration * concurrency * 0.5)
    if phase.sessions is not None:
        return phase.sessions * 3  # assume ~3 turns average
    return 1000  # conservative default


def _estimate_phase_duration(phase: BasePhaseConfig, concurrency: int) -> float:
    """Estimate phase duration in seconds."""
    if phase.duration is not None:
        return phase.duration
    requests = phase.requests or 1000
    rate = getattr(phase, "rate", None)
    if rate is not None:
        return requests / rate
    # Concurrency-driven: estimate ~2 sec per request / concurrency
    return requests * 2.0 / max(concurrency, 1)


def _extract_dataset_params(ds: object) -> tuple[int, int, int, int]:
    """Extract ISL, OSL, turns, count from a dataset config.

    Returns:
        (avg_isl, avg_osl, max_turns, count)
    """
    isl = 512  # default
    osl = 128
    turns = 1
    count = 100

    # SyntheticDataset has .prompts with .isl/.osl and .entries
    if hasattr(ds, "prompts") and ds.prompts is not None:
        prompts = ds.prompts
        if hasattr(prompts, "isl") and prompts.isl is not None:
            isl = (
                int(prompts.isl.mean)
                if hasattr(prompts.isl, "mean")
                else int(prompts.isl)
            )
        if hasattr(prompts, "osl") and prompts.osl is not None:
            osl = (
                int(prompts.osl.mean)
                if hasattr(prompts.osl, "mean")
                else int(prompts.osl)
            )
        if hasattr(prompts, "sequence_distribution") and prompts.sequence_distribution:
            # Weighted average
            total_prob = sum(e.probability for e in prompts.sequence_distribution)
            if total_prob > 0:
                isl = int(
                    sum(
                        e.isl.mean * e.probability
                        for e in prompts.sequence_distribution
                    )
                    / total_prob
                )
                osl = int(
                    sum(
                        e.osl.mean * e.probability
                        for e in prompts.sequence_distribution
                    )
                    / total_prob
                )

    if hasattr(ds, "entries"):
        count = ds.entries or 100
    elif hasattr(ds, "count"):
        count = ds.count or 100

    # Multi-turn detection
    if hasattr(ds, "format"):
        fmt = ds.format
        fmt_str = fmt.value if hasattr(fmt, "value") else str(fmt)
        if "multi_turn" in fmt_str:
            turns = 5  # reasonable default for multi-turn

    return isl, osl, turns, count


# =============================================================================
# Component estimators (pure functions)
# =============================================================================


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


def _estimate_worker(
    concurrency_per_worker: int,
    avg_osl: int,
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
    pool_bytes = connections_per_worker * _BYTES_PER_CONNECTION
    pool_mib = _mib(pool_bytes)

    # Per in-flight request memory: record shell + turns + response
    turn_bytes = _TURN_BASE_BYTES + avg_isl * _TURN_BYTES_PER_TOKEN
    if streaming:
        response_bytes = _SSE_MESSAGE_BASE_BYTES + avg_osl * _SSE_BYTES_PER_CHUNK
    else:
        response_bytes = (
            _TEXT_RESPONSE_BASE_BYTES + avg_osl * _TEXT_RESPONSE_BYTES_PER_TOKEN
        )
    per_request = _REQUEST_RECORD_BASE_BYTES + turn_bytes + response_bytes

    inflight_bytes = concurrency_per_worker * per_request
    inflight_mib = _mib(inflight_bytes)

    # Session cache (multi-turn only)
    session_mib = 0.0
    if max_turns > 1:
        extra_turn_bytes = (max_turns - 1) * turn_bytes
        session_bytes = concurrency_per_worker * extra_turn_bytes
        session_mib = _mib(session_bytes)

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


def _estimate_record_processor(
    num_models: int,
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

    Memory components:
    - Tokenizer cache per model (dominant: 50-150 MiB per model via RSS)
    - In-flight records being processed: concurrency_per_rp × per-record size
    - Raw record JSONL buffer (batch_size=10): ISL/OSL dependent when streaming
    - Record export JSONL buffer (batch_size=100): ~100 KiB (metrics only)
    """
    base = _SERVICE_BASE_MIB["record_processor"] + _PYTHON_CHILD_SUBPROCESS_BASE_MIB

    # Tokenizer cache (RSS-measured: GPT-2 ~73 MiB, Llama-3 ~50-100 MiB, large SP ~150 MiB)
    tokenizer_mib = num_models * _TOKENIZER_CACHE_MIB

    # In-flight records being processed concurrently.
    # Each holds full RequestRecord: record shell + turn (ISL) + response (OSL).
    # Same per-request model as worker estimate.
    turn_bytes = _TURN_BASE_BYTES + avg_isl * _TURN_BYTES_PER_TOKEN
    if streaming:
        response_bytes = _SSE_MESSAGE_BASE_BYTES + avg_osl * _SSE_BYTES_PER_CHUNK
    else:
        response_bytes = (
            _TEXT_RESPONSE_BASE_BYTES + avg_osl * _TEXT_RESPONSE_BYTES_PER_TOKEN
        )
    per_record_bytes = _REQUEST_RECORD_BASE_BYTES + turn_bytes + response_bytes
    inflight_mib = _mib(concurrency_per_rp * per_record_bytes)

    # Raw record JSONL write buffer: 10 serialized records before flush.
    # SSE records serialize all chunks (~80B/chunk), text records compact (~4B/token).
    _RAW_BATCH_SIZE = 10
    per_raw_record_bytes = 1500 + avg_osl * (80 if streaming else 4)
    raw_buffer_mib = _mib(_RAW_BATCH_SIZE * per_raw_record_bytes)

    # Record export JSONL buffer: 100 records × ~1.1 KiB each (metrics + metadata).
    _EXPORT_BATCH_SIZE = 100
    export_buffer_mib = _mib(_EXPORT_BATCH_SIZE * 1100)

    variable = tokenizer_mib + inflight_mib + raw_buffer_mib + export_buffer_mib
    # Peak includes burst where all concurrent records + write buffer are live.
    peak = base + variable * 1.1

    model_word = "model" if num_models == 1 else "models"
    mode = "SSE" if streaming else "text"
    warning = None
    if inflight_mib > 50:
        warning = (
            f"RP in-flight records use {inflight_mib:.0f} MiB "
            f"({concurrency_per_rp} records x {_mib(per_record_bytes):.1f} MiB each). "
            f"Driven by {mode} ISL={avg_isl} OSL={avg_osl} at concurrency {concurrency_per_rp}."
        )
    elif tokenizer_mib > 450:
        warning = (
            f"Tokenizer cache is {tokenizer_mib} MiB ({num_models} {model_word} x "
            f"{_TOKENIZER_CACHE_MIB} MiB each). Consider reducing model count."
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


# =============================================================================
# Pod-level estimators
# =============================================================================


def _memory_from_resources(resources: dict[str, dict[str, str]]) -> str:
    """Return the memory string from limits if present, otherwise requests."""
    return resources.get("limits", resources["requests"])["memory"]


def _get_controller_limit_mib() -> float:
    """Get controller pod memory limit from K8sEnvironment."""
    total = 0
    for key in CONTROLLER_RESOURCE_KEYS:
        total += parse_memory_mib(
            _memory_from_resources(getattr(K8sEnvironment, key).to_k8s_resources())
        )
    return float(total)


def _get_worker_pod_limit_mib(workers_per_pod: int, rp_per_pod: int) -> float:
    """Get worker pod memory limit from K8sEnvironment."""
    return float(
        parse_memory_mib(
            _memory_from_resources(K8sEnvironment.WORKER_POD.to_k8s_resources())
        )
    )


class MemoryEstimator:
    """Orchestrates memory estimation across all pod types."""

    def __init__(self, params: MemoryEstimationParams) -> None:
        self.params = params

    def estimate(self) -> ClusterMemoryEstimate:
        """Run the full estimation and generate warnings."""
        p = self.params
        controller = self._estimate_controller()
        worker_pod = self._estimate_worker_pod()
        operator = self._estimate_operator()

        estimate = ClusterMemoryEstimate(
            params=p,
            controller=controller,
            worker_pod=worker_pod,
            operator=operator,
        )
        self._generate_warnings(estimate)
        return estimate

    def _estimate_controller(self) -> PodEstimate:
        p = self.params
        components = [
            _estimate_fixed_service("system_controller"),
            _estimate_fixed_service("timing_manager"),
            _estimate_dataset_manager(
                p.dataset_count, p.avg_isl_tokens, p.avg_osl_tokens, p.max_turns
            ),
            _estimate_records_manager(p.total_requests, p.num_standard_metrics),
            _estimate_fixed_service("api_service", "API Service"),
            _estimate_gpu_telemetry(
                p.num_gpus,
                p.total_benchmark_duration_s,
                p.gpu_sample_interval_s,
                p.num_gpu_metrics,
            ),
            _estimate_server_metrics(
                p.num_server_metrics_endpoints,
                p.total_benchmark_duration_s,
                p.server_metrics_scrape_interval_s,
                p.est_unique_metric_series,
                p.est_histogram_metrics,
                p.est_histogram_buckets,
            ),
            _estimate_fixed_service("results_sidecar", "Results Sidecar"),
        ]
        # ZMQ proxies
        components.append(
            ComponentEstimate(
                name="ZMQ Proxies",
                base_mib=_NUM_ZMQ_PROXIES * _ZMQ_PROXY_MIB,
                variable_mib=0,
                peak_mib=_NUM_ZMQ_PROXIES * _ZMQ_PROXY_MIB,
                formula=f"{_NUM_ZMQ_PROXIES} proxies x {_ZMQ_PROXY_MIB} MiB",
                dominant_factor="fixed",
            )
        )

        return PodEstimate(
            pod_type="controller",
            components=components,
            current_limit_mib=_get_controller_limit_mib(),
            replicas=1,
        )

    def _estimate_worker_pod(self) -> PodEstimate:
        p = self.params
        conc_per_worker = max(1, p.max_concurrency // max(p.total_workers, 1))

        # RP queue depth: the ZMQ pull client has PULL_MAX_CONCURRENCY=100K
        # (effectively unbounded). All completed records are deserialized into
        # Python objects and held in memory until the RP finishes processing.
        #
        # At steady state with fast processing (low ISL/OSL), the queue depth
        # is ~pod_concurrency / rps_per_pod. But at high ISL/OSL, tokenization
        # becomes the bottleneck and records accumulate. The queue can grow to
        # the full pod concurrency (all records queued, none processed yet).
        #
        # Model: use pod_concurrency as the queue depth. This is conservative
        # at low ISL/OSL but accurate at high ISL/OSL where it matters most
        # (because per-record memory is large).
        pod_concurrency = conc_per_worker * p.workers_per_pod
        conc_per_rp = max(1, pod_concurrency // max(p.record_processors_per_pod, 1))

        # At high token counts, tokenization becomes the bottleneck. Each record
        # with ISL+OSL > 10K takes 50-500ms to tokenize. During that time,
        # workers keep completing and pushing records into the RP's ZMQ pull
        # queue (PULL_MAX_CONCURRENCY=100K, effectively unbounded). Records
        # pile up as deserialized Python objects faster than they can be processed.
        #
        # The queue depth scales with token pressure: more tokens per record
        # means slower processing means deeper queue. Calibrated against actual
        # PSS: at ISL+OSL=173K, queue reaches ~150 records per RP (10x base).
        avg_tokens = p.avg_isl_tokens + p.avg_osl_tokens
        if avg_tokens > 10_000:
            token_pressure = min(avg_tokens / 10_000, 10.0)
            rp_queue_depth = int(conc_per_rp * token_pressure)
        else:
            rp_queue_depth = conc_per_rp

        wpm = _estimate_fixed_service("worker_group_manager", "WorkerGroupManager")
        worker = _estimate_worker(
            conc_per_worker,
            p.avg_osl_tokens,
            p.streaming,
            p.max_turns,
            p.avg_isl_tokens,
            p.connections_per_worker,
        )
        rp = _estimate_record_processor(
            p.num_models,
            p.avg_isl_tokens,
            p.avg_osl_tokens,
            p.streaming,
            rp_queue_depth,
        )

        # Scale by count
        workers_total = ComponentEstimate(
            name=f"Workers (x{p.workers_per_pod})",
            base_mib=worker.base_mib * p.workers_per_pod,
            variable_mib=worker.variable_mib * p.workers_per_pod,
            peak_mib=worker.peak_mib * p.workers_per_pod,
            formula=f"{p.workers_per_pod} x [{worker.formula}]",
            dominant_factor=worker.dominant_factor,
            warning=worker.warning,
        )
        rp_total = ComponentEstimate(
            name=f"RecordProcessors (x{p.record_processors_per_pod})",
            base_mib=rp.base_mib * p.record_processors_per_pod,
            variable_mib=rp.variable_mib * p.record_processors_per_pod,
            peak_mib=rp.peak_mib * p.record_processors_per_pod,
            formula=f"{p.record_processors_per_pod} x [{rp.formula}]",
            dominant_factor=rp.dominant_factor,
            warning=rp.warning,
        )

        return PodEstimate(
            pod_type="worker",
            components=[wpm, workers_total, rp_total],
            current_limit_mib=_get_worker_pod_limit_mib(
                p.workers_per_pod, p.record_processors_per_pod
            ),
            replicas=p.num_worker_pods,
        )

    def _estimate_operator(self) -> PodEstimate:
        return PodEstimate(
            pod_type="operator",
            components=[
                ComponentEstimate(
                    name="Operator",
                    base_mib=256,
                    variable_mib=0,
                    peak_mib=256,
                    formula="fixed 256 MiB",
                    dominant_factor="fixed",
                )
            ],
            current_limit_mib=512,
            replicas=1,
        )

    def _generate_warnings(self, est: ClusterMemoryEstimate) -> None:
        p = self.params
        warnings = []
        recommendations = []

        # Controller headroom
        if est.controller.at_risk:
            warnings.append(
                f"Controller pod peak ({est.controller.total_peak_mib:.0f} MiB) is within "
                f"{est.controller.headroom_pct:.1f}% of limit ({est.controller.current_limit_mib:.0f} MiB). "
                "Risk of OOM kill."
            )

        # RecordsManager dominance
        rm = next(
            (c for c in est.controller.components if c.name == "RecordsManager"), None
        )
        if rm and est.controller.current_limit_mib > 0:
            rm_pct = rm.steady_state_mib / est.controller.current_limit_mib * 100
            if rm_pct > _RECORDS_MANAGER_WARN_PCT:
                warnings.append(
                    f"RecordsManager uses {rm_pct:.0f}% of controller limit "
                    f"({rm.steady_state_mib:.0f}/{est.controller.current_limit_mib:.0f} MiB). "
                    f"Driven by {p.total_requests:,} total requests."
                )

        # Worker pod headroom
        if est.worker_pod.at_risk:
            warnings.append(
                f"Worker pod peak ({est.worker_pod.total_peak_mib:.0f} MiB) is within "
                f"{est.worker_pod.headroom_pct:.1f}% of limit ({est.worker_pod.current_limit_mib:.0f} MiB)."
            )

        # High request count
        if p.total_requests > 500_000:
            warnings.append(
                f"Total requests ({p.total_requests:,}) will create significant metric array storage. "
                f"~{p.num_standard_metrics} metrics x {_ceil_pow2(p.total_requests):,} capacity x 8B each."
            )

        # Tokenizer memory (per-processor, not aggregate across all RPs)
        per_rp_tokenizer_mib = p.num_models * _TOKENIZER_CACHE_MIB
        if per_rp_tokenizer_mib > 450:
            model_word = "model" if p.num_models == 1 else "models"
            warnings.append(
                f"Each RecordProcessor loads {per_rp_tokenizer_mib} MiB in tokenizer cache "
                f"({p.num_models} {model_word} x {_TOKENIZER_CACHE_MIB} MiB). "
                f"With {p.record_processors_per_pod} RP(s)/pod, that is "
                f"{per_rp_tokenizer_mib * p.record_processors_per_pod} MiB per worker pod."
            )

        # HTTP trace
        if p.export_http_trace and p.total_requests > 10_000:
            warnings.append(
                f"HTTP trace export with {p.total_requests:,} requests will accumulate "
                "per-chunk timing data in memory. Consider disabling for large runs."
            )

        # Multi-turn session warning
        if p.max_turns > 1:
            sessions_per_worker = max(1, p.max_concurrency // max(p.total_workers, 1))
            if sessions_per_worker > 100:
                warnings.append(
                    f"Multi-turn with {sessions_per_worker} concurrent sessions per worker. "
                    "Session cache may consume significant memory."
                )

        # Recommendations
        if est.controller.headroom_pct > 50 and est.worker_pod.headroom_pct > 50:
            recommendations.append(
                "Current resource limits have adequate headroom for this workload."
            )

        if est.controller.at_risk:
            recommendations.append(
                f"Increase controller memory limit to at least "
                f"{est.controller.recommended_limit_mib} MiB "
                f"(currently {est.controller.current_limit_mib:.0f} MiB)."
            )
        if est.worker_pod.at_risk:
            recommendations.append(
                f"Increase worker pod memory limit to at least "
                f"{est.worker_pod.recommended_limit_mib} MiB "
                f"(currently {est.worker_pod.current_limit_mib:.0f} MiB)."
            )

        est.warnings = warnings
        est.recommendations = recommendations


# =============================================================================
# Formatting
# =============================================================================


def format_estimate(est: ClusterMemoryEstimate) -> str:
    """Format a ClusterMemoryEstimate as a human-readable string."""
    p = est.params
    lines: list[str] = []

    lines.append("Memory Estimation for AIPerf Kubernetes Deployment")
    lines.append("=" * 68)
    lines.append("")
    lines.append(
        f"Topology: 1 controller + {p.num_worker_pods} worker pod(s) "
        f"({p.workers_per_pod} workers/pod, {p.record_processors_per_pod} RP/pod)"
    )
    lines.append(
        f"Total requests: ~{p.total_requests:,} | "
        f"Max concurrency: {p.max_concurrency:,} | "
        f"Duration: {p.total_benchmark_duration_s:.0f}s"
    )
    lines.append(
        f"Dataset: {p.dataset_count:,} conversations | "
        f"ISL: {p.avg_isl_tokens} | OSL: {p.avg_osl_tokens} | "
        f"Turns: {p.max_turns}"
    )
    lines.append("")

    # Controller pod
    _format_pod(lines, "Controller Pod", est.controller)

    # Worker pod
    wp_label = f"Worker Pod (x{p.num_worker_pods})"
    _format_pod(lines, wp_label, est.worker_pod)

    # Cluster total
    lines.append("Cluster Total")
    lines.append("-" * 68)
    lines.append(
        f"  {'Controller':<42} {est.controller.total_steady_state_mib:>7.0f} MiB"
    )
    worker_total = est.worker_pod.total_steady_state_mib * p.num_worker_pods
    lines.append(
        f"  {'Workers (' + str(p.num_worker_pods) + ' pods)':<42} {worker_total:>7.0f} MiB"
    )
    lines.append(f"  {'Operator':<42} {est.operator.total_steady_state_mib:>7.0f} MiB")
    lines.append("-" * 68)
    lines.append(f"  {'TOTAL':<42} {est.total_cluster_mib:>7.0f} MiB")
    lines.append("")

    # Warnings
    if est.warnings:
        lines.append("Warnings:")
        for w in est.warnings:
            lines.append(f"  [!] {w}")
        lines.append("")

    # Recommendations
    if est.recommendations:
        lines.append("Recommendations:")
        for r in est.recommendations:
            lines.append(f"  - {r}")
        lines.append("")

    return "\n".join(lines)


def _format_pod(lines: list[str], title: str, pod: PodEstimate) -> None:
    """Format a single pod estimate table."""
    risk = " [!]" if pod.at_risk else ""
    lines.append(f"{title}{risk}")
    lines.append(f"  {'Component':<42} {'Steady':>7}  {'Peak':>7}")
    lines.append("-" * 68)
    for c in pod.components:
        warn = " [!]" if c.warning else ""
        lines.append(
            f"  {c.name:<42} {c.steady_state_mib:>6.0f}  {c.peak_mib:>6.0f}{warn}"
        )
    lines.append("-" * 68)
    lines.append(
        f"  {'TOTAL':<42} {pod.total_steady_state_mib:>6.0f}  {pod.total_peak_mib:>6.0f}  "
        f"(limit: {pod.current_limit_mib:.0f})"
    )
    headroom_str = f"{pod.headroom_pct:.1f}%"
    lines.append(f"  {'Headroom':<42} {headroom_str:>14}")
    lines.append("")


# =============================================================================
# Public API
# =============================================================================


def estimate_memory(
    config: AIPerfConfig,
    total_workers: int = 10,
    workers_per_pod: int | None = None,
    connections_per_worker: int = 200,
) -> ClusterMemoryEstimate:
    """Estimate memory usage for an AIPerf Kubernetes deployment.

    This is the primary entry point. Derives all estimation parameters
    from the config and returns a full cluster estimate with warnings.

    Args:
        config: The benchmark configuration.
        total_workers: Total desired workers.
        workers_per_pod: Workers per pod (None = default).
        connections_per_worker: Connections per worker.

    Returns:
        ClusterMemoryEstimate with per-pod and cluster-wide estimates.
    """
    params = MemoryEstimationParams.from_config(
        config, total_workers, workers_per_pod, connections_per_worker
    )
    estimator = MemoryEstimator(params)
    return estimator.estimate()
