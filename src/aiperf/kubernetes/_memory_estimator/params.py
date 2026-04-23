# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``MemoryEstimationParams`` and helpers that derive it from ``AIPerfConfig``."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from aiperf.kubernetes._memory_estimator.constants import (
    _DEFAULT_GPU_METRICS,
    _DEFAULT_HISTOGRAM_BUCKETS,
    _DEFAULT_HISTOGRAM_METRICS,
    _DEFAULT_NUM_STANDARD_METRICS,
    _DEFAULT_SCRAPE_INTERVAL_S,
    _DEFAULT_UNIQUE_METRIC_SERIES,
)

if TYPE_CHECKING:
    from aiperf.config.config import AIPerfConfig
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
        topology = _derive_topology(config, total_workers, workers_per_pod)
        max_conc, total_req, total_dur = _derive_load_profile(config)
        ds = config.get_default_dataset()
        isl, osl, turns, count = _extract_dataset_params(ds)
        num_gpu_urls, est_gpus = _derive_gpu_telemetry(config)
        num_sm_urls = _derive_server_metrics(config)
        export_trace = _derive_http_trace(config)

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
            num_models=len(config.get_model_names()),
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
    config: AIPerfConfig, total_workers: int, workers_per_pod: int | None
) -> _Topology:
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
    return _Topology(actual_wpp=actual_wpp, num_pods=num_pods, rp_per_pod=rp_per_pod)


def _derive_load_profile(config: AIPerfConfig) -> tuple[int, int, float]:
    """Return (max_concurrency, total_requests, total_duration_s) across phases."""
    max_conc = 1
    total_req = 0
    total_dur = 0.0
    for phase in config.phases.values():
        conc = getattr(phase, "concurrency", None) or 1
        max_conc = max(max_conc, conc)
        total_req += _estimate_phase_requests(phase, conc)
        total_dur += _estimate_phase_duration(phase, conc)
    return max_conc, total_req, total_dur


def _derive_gpu_telemetry(config: AIPerfConfig) -> tuple[int, int]:
    """Return (num_gpu_urls, estimated_total_gpus). Rough: 1-8 GPUs per endpoint."""
    num_gpu_urls = len(config.gpu_telemetry.urls) if config.gpu_telemetry.enabled else 0
    return num_gpu_urls, num_gpu_urls * 4 if num_gpu_urls else 0


def _derive_server_metrics(config: AIPerfConfig) -> int:
    return len(config.server_metrics.urls) if config.server_metrics.enabled else 0


def _derive_http_trace(config: AIPerfConfig) -> bool:
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
