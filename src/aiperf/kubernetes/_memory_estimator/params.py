# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``MemoryEstimationParams`` and helpers that derive it from ``AIPerfConfig``."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Literal, cast

from aiperf.common.enums import ServerMetricsDiscoveryMode
from aiperf.common.metric_utils import normalize_metrics_endpoint_url
from aiperf.kubernetes._memory_estimator.constants import (
    _DEFAULT_GPU_METRICS,
    _DEFAULT_HISTOGRAM_BUCKETS,
    _DEFAULT_HISTOGRAM_METRICS,
    _DEFAULT_NUM_STANDARD_METRICS,
    _DEFAULT_PHASE_REQUEST_COUNT,
    _DEFAULT_UNIQUE_METRIC_SERIES,
    _PHASE_AVG_SEC_PER_REQUEST,
)
from aiperf.kubernetes.spec_converter import (
    DEFAULT_CONNECTIONS_PER_WORKER,
    apply_worker_config,
)

if TYPE_CHECKING:
    from aiperf.config.config import AIPerfConfig, BenchmarkConfig
    from aiperf.config.phases import BasePhaseConfig


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

    list_metric_backend: Literal["ragged", "tdigest"]
    """Storage backend used for list-valued record metrics."""

    num_endpoints: int
    """Number of target inference endpoint URLs."""

    connections_per_worker: int
    """HTTP connection pool size per worker."""

    gpu_telemetry_enabled: bool
    """Whether the GPU telemetry manager container is deployed."""

    server_metrics_enabled: bool
    """Whether the server metrics manager container is deployed."""

    server_metrics_discovery_enabled: bool
    """Whether Kubernetes discovery can add server metrics endpoints."""

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
        connections_per_worker: int = DEFAULT_CONNECTIONS_PER_WORKER,
    ) -> MemoryEstimationParams:
        """Derive estimation parameters from an AIPerfConfig.

        Args:
            config: The benchmark configuration.
            total_workers: Total desired workers (from KubeOptions.total_workers).
            workers_per_pod: Workers per pod (None = use default).
            connections_per_worker: Connections per worker.
        """
        bench = config.benchmark
        topology = _derive_topology(bench, total_workers, workers_per_pod)
        max_conc, total_req, total_dur = _derive_load_profile(bench)
        ds = bench.get_default_dataset()
        isl, osl, turns, count = _extract_dataset_params(ds)
        est_gpus = _derive_gpu_telemetry(bench)
        num_sm_urls = _derive_server_metrics(bench)
        export_trace = _derive_http_trace(bench)
        from aiperf.common.environment import Environment

        return cls(
            total_workers=total_workers,
            workers_per_pod=topology.actual_wpp,
            num_worker_pods=topology.num_pods,
            record_processors_per_pod=topology.rp_per_pod,
            max_concurrency=max_conc,
            total_requests=max(total_req, 1),
            total_benchmark_duration_s=max(total_dur, 60.0),
            dataset_count=count,
            avg_isl_tokens=isl,
            avg_osl_tokens=osl,
            max_turns=turns,
            streaming=bench.endpoint.streaming,
            list_metric_backend=Environment.METRICS.LIST_BACKEND,
            num_endpoints=len(bench.endpoint.urls),
            connections_per_worker=connections_per_worker,
            gpu_telemetry_enabled=bench.gpu_telemetry.enabled,
            server_metrics_enabled=bench.server_metrics.enabled,
            server_metrics_discovery_enabled=(
                bench.server_metrics.enabled
                and bench.server_metrics.discovery.mode
                != ServerMetricsDiscoveryMode.DISABLED
            ),
            num_gpus=est_gpus,
            gpu_sample_interval_s=Environment.GPU.COLLECTION_INTERVAL,
            num_gpu_metrics=_DEFAULT_GPU_METRICS,
            num_server_metrics_endpoints=num_sm_urls,
            server_metrics_scrape_interval_s=Environment.SERVER_METRICS.COLLECTION_INTERVAL,
            est_unique_metric_series=_DEFAULT_UNIQUE_METRIC_SERIES,
            est_histogram_metrics=_DEFAULT_HISTOGRAM_METRICS,
            est_histogram_buckets=_DEFAULT_HISTOGRAM_BUCKETS,
            num_models=len(bench.get_model_names()),
            num_standard_metrics=_DEFAULT_NUM_STANDARD_METRICS,
            export_http_trace=export_trace,
        )


@dataclass(slots=True)
class _Topology:
    """Internal: pod/worker/RP layout derived from config + CLI flags."""

    actual_wpp: int
    num_pods: int
    rp_per_pod: int


def _derive_topology(
    config: BenchmarkConfig, total_workers: int, workers_per_pod: int | None
) -> _Topology:
    """Derive the pod layout by running the deployment path's own normalizer.

    ``spec_converter.apply_worker_config`` is the single source of truth for the
    JobSet topology: the single-pod collapse for worker totals that are not a
    multiple of ``workers_per_pod``, the split of an authored
    ``runtime.record_processors`` total across pods, and the scale-factor
    fallback all live there. A re-derivation here drifted from it once already,
    so the estimator calls it instead of restating the rules.

    It runs against a throwaway copy of ``benchmark.runtime`` rather than the
    caller's config: ``kube profile`` and ``kube generate --operator`` estimate
    *before* the config is normalized and serialized, and the estimate must not
    be the thing that mutates the spec they are about to submit.
    """
    runtime = config.runtime.model_copy(deep=True)
    if workers_per_pod is not None:
        runtime.workers_per_pod = workers_per_pod

    # apply_worker_config reads and writes only ``benchmark.runtime``.
    probe = SimpleNamespace(benchmark=SimpleNamespace(runtime=runtime))
    # Unlike the deployment path, the estimator is reachable from unvalidated
    # CR specs, where a non-positive total would divide by zero downstream.
    num_pods = apply_worker_config(
        cast("AIPerfConfig", probe), max(1, int(total_workers))
    )
    return _Topology(
        actual_wpp=runtime.workers_per_pod,
        num_pods=num_pods,
        rp_per_pod=runtime.record_processors_per_pod,
    )


def _derive_load_profile(config: BenchmarkConfig) -> tuple[int, int, float]:
    """Return (max_concurrency, total_requests, total_duration_s) across phases."""
    max_conc = 1
    total_req = 0
    total_dur = 0.0
    for phase in config.phases:
        conc = getattr(phase, "concurrency", None) or 1
        max_conc = max(max_conc, conc)
        total_req += _estimate_phase_requests(phase, conc)
        total_dur += _estimate_phase_duration(phase, conc)
    return max_conc, total_req, total_dur


def _derive_gpu_telemetry(config: BenchmarkConfig) -> int:
    """Estimate GPUs from configured DCGM endpoints when telemetry is enabled."""
    if not config.gpu_telemetry.enabled:
        return 0
    from aiperf.common.environment import Environment

    collector = config.gpu_telemetry.collector
    from aiperf.plugin import plugins

    if plugins.get_gpu_telemetry_collector_metadata(collector).is_local:
        return 4

    endpoints = [
        *Environment.GPU.DEFAULT_DCGM_ENDPOINTS,
        *(
            f"{url.rstrip('/')}/metrics"
            if not url.rstrip("/").endswith("/metrics")
            else url.rstrip("/")
            for url in (config.gpu_telemetry.urls or [])
        ),
    ]
    return len(dict.fromkeys(endpoints)) * 4


def _derive_server_metrics(config: BenchmarkConfig) -> int:
    """Count the normalized endpoint union used by the metrics manager."""
    if not config.server_metrics.enabled:
        return 0
    endpoints = {
        normalize_metrics_endpoint_url(url)
        for url in [*config.endpoint.urls, *(config.server_metrics.urls or [])]
    }
    return len(endpoints)


def _derive_http_trace(config: BenchmarkConfig) -> bool:
    if not hasattr(config.artifacts, "formats"):
        return False
    return "http_trace" in {
        fmt.value if hasattr(fmt, "value") else str(fmt)
        for fmt in config.artifacts.formats
    }


def _estimate_phase_requests(phase: BasePhaseConfig, concurrency: int) -> int:
    """Estimate total requests for a phase."""
    if phase.requests is not None:
        return phase.requests
    if phase.duration is not None:
        rate = getattr(phase, "rate", None)
        if rate is not None:
            return int(phase.duration * rate)
        # Inverse of the concurrency-driven branch of _estimate_phase_duration:
        # each slot completes one request every _PHASE_AVG_SEC_PER_REQUEST.
        return int(phase.duration * concurrency / _PHASE_AVG_SEC_PER_REQUEST)
    if phase.sessions is not None:
        return phase.sessions * 3  # assume ~3 turns average
    return _DEFAULT_PHASE_REQUEST_COUNT


def _estimate_phase_duration(phase: BasePhaseConfig, concurrency: int) -> float:
    """Estimate phase duration in seconds."""
    if phase.duration is not None:
        return phase.duration
    requests = phase.requests or _DEFAULT_PHASE_REQUEST_COUNT
    rate = getattr(phase, "rate", None)
    if rate is not None:
        return requests / rate
    # Concurrency-driven: estimate avg-latency-sec per request / concurrency.
    return requests * _PHASE_AVG_SEC_PER_REQUEST / max(concurrency, 1)


def _extract_dataset_params(ds: object) -> tuple[int, int, int, int]:
    """Extract ISL, OSL, turns, count from a dataset config.

    Returns:
        (avg_isl, avg_osl, max_turns, count)
    """
    isl, osl = _extract_sequence_lengths(ds)
    count = _extract_entry_count(ds)
    turns = _extract_max_turns(ds)
    return isl, osl, turns, count


def _extract_sequence_lengths(ds: object) -> tuple[int, int]:
    isl = 512  # default
    osl = 128
    if not hasattr(ds, "prompts") or ds.prompts is None:
        return isl, osl
    prompts = ds.prompts
    if hasattr(prompts, "isl") and prompts.isl is not None:
        isl = (
            int(prompts.isl.mean) if hasattr(prompts.isl, "mean") else int(prompts.isl)
        )
    if hasattr(prompts, "osl") and prompts.osl is not None:
        osl = (
            int(prompts.osl.mean) if hasattr(prompts.osl, "mean") else int(prompts.osl)
        )
    if hasattr(prompts, "sequence_distribution") and prompts.sequence_distribution:
        total_prob = sum(e.probability for e in prompts.sequence_distribution)
        if total_prob > 0:
            isl = int(
                sum(e.isl.mean * e.probability for e in prompts.sequence_distribution)
                / total_prob
            )
            osl = int(
                sum(e.osl.mean * e.probability for e in prompts.sequence_distribution)
                / total_prob
            )
    return isl, osl


def _extract_entry_count(ds: object) -> int:
    if hasattr(ds, "entries"):
        return ds.entries or 100
    if hasattr(ds, "count"):
        return ds.count or 100
    return 100


def _extract_max_turns(ds: object) -> int:
    if not hasattr(ds, "format"):
        return 1
    fmt = ds.format
    fmt_str = fmt.value if hasattr(fmt, "value") else str(fmt)
    return 5 if "multi_turn" in fmt_str else 1
