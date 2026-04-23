# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``MemoryEstimator`` orchestrator + pod-level estimators."""

from __future__ import annotations

from aiperf.kubernetes._memory_estimator.components import (
    _estimate_dataset_manager,
    _estimate_fixed_service,
    _estimate_gpu_telemetry,
    _estimate_record_processor,
    _estimate_records_manager,
    _estimate_server_metrics,
    _estimate_worker,
)
from aiperf.kubernetes._memory_estimator.constants import (
    _NUM_ZMQ_PROXIES,
    _RECORDS_MANAGER_WARN_PCT,
    _TOKENIZER_CACHE_MIB,
    _ZMQ_PROXY_MIB,
)
from aiperf.kubernetes._memory_estimator.estimates import (
    ClusterMemoryEstimate,
    ComponentEstimate,
    PodEstimate,
)
from aiperf.kubernetes._memory_estimator.params import MemoryEstimationParams
from aiperf.kubernetes._memory_estimator.utils import _ceil_pow2
from aiperf.kubernetes.environment import CONTROLLER_RESOURCE_KEYS, K8sEnvironment
from aiperf.kubernetes.utils import parse_memory_mib


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


def _rp_queue_depth(conc_per_rp: int, avg_isl_tokens: int, avg_osl_tokens: int) -> int:
    """Model RP pull-queue depth under token pressure.

    At high token counts, tokenization becomes the bottleneck. Each record
    with ISL+OSL > 10K takes 50-500ms to tokenize. During that time, workers
    keep completing and pushing records into the RP's ZMQ pull queue
    (PULL_MAX_CONCURRENCY=100K, effectively unbounded). Records pile up as
    deserialized Python objects faster than they can be processed.

    Calibrated against actual PSS: at ISL+OSL=173K, queue reaches ~150
    records per RP (10x base).
    """
    avg_tokens = avg_isl_tokens + avg_osl_tokens
    if avg_tokens > 10_000:
        token_pressure = min(avg_tokens / 10_000, 10.0)
        return int(conc_per_rp * token_pressure)
    return conc_per_rp


def _aggregate_pod_component(
    name: str, per_unit: ComponentEstimate, count: int
) -> ComponentEstimate:
    """Scale a per-process estimate to a pod total of ``count`` copies."""
    return ComponentEstimate(
        name=name,
        base_mib=per_unit.base_mib * count,
        variable_mib=per_unit.variable_mib * count,
        peak_mib=per_unit.peak_mib * count,
        formula=f"{count} x [{per_unit.formula}]",
        dominant_factor=per_unit.dominant_factor,
        warning=per_unit.warning,
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
                unique_series=p.est_unique_metric_series,
                histogram_count=p.est_histogram_metrics,
                histogram_buckets=p.est_histogram_buckets,
            ),
            _estimate_fixed_service("results_sidecar", "Results Sidecar"),
            ComponentEstimate(
                name="ZMQ Proxies",
                base_mib=_NUM_ZMQ_PROXIES * _ZMQ_PROXY_MIB,
                variable_mib=0,
                peak_mib=_NUM_ZMQ_PROXIES * _ZMQ_PROXY_MIB,
                formula=f"{_NUM_ZMQ_PROXIES} proxies x {_ZMQ_PROXY_MIB} MiB",
                dominant_factor="fixed",
            ),
        ]

        return PodEstimate(
            pod_type="controller",
            components=components,
            current_limit_mib=_get_controller_limit_mib(),
            replicas=1,
        )

    def _estimate_worker_pod(self) -> PodEstimate:
        p = self.params
        conc_per_worker = max(1, p.max_concurrency // max(p.total_workers, 1))
        pod_concurrency = conc_per_worker * p.workers_per_pod
        conc_per_rp = max(1, pod_concurrency // max(p.record_processors_per_pod, 1))
        rp_queue_depth = _rp_queue_depth(
            conc_per_rp, p.avg_isl_tokens, p.avg_osl_tokens
        )

        wpm = _estimate_fixed_service("worker_group_manager", "WorkerGroupManager")
        worker = _estimate_worker(
            conc_per_worker,
            p.avg_osl_tokens,
            streaming=p.streaming,
            max_turns=p.max_turns,
            avg_isl=p.avg_isl_tokens,
            connections_per_worker=p.connections_per_worker,
        )
        rp = _estimate_record_processor(
            p.num_models,
            avg_isl=p.avg_isl_tokens,
            avg_osl=p.avg_osl_tokens,
            streaming=p.streaming,
            concurrency_per_rp=rp_queue_depth,
        )

        workers_total = _aggregate_pod_component(
            f"Workers (x{p.workers_per_pod})", worker, p.workers_per_pod
        )
        rp_total = _aggregate_pod_component(
            f"RecordProcessors (x{p.record_processors_per_pod})",
            rp,
            p.record_processors_per_pod,
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
        warnings: list[str] = []
        warnings.extend(_warn_controller_pod(est))
        warnings.extend(_warn_records_manager(est, self.params))
        warnings.extend(_warn_worker_pod(est))
        warnings.extend(_warn_request_volume(self.params))
        warnings.extend(_warn_tokenizer(self.params))
        warnings.extend(_warn_http_trace(self.params))
        warnings.extend(_warn_multi_turn(self.params))
        est.warnings = warnings
        est.recommendations = _build_recommendations(est)


# =============================================================================
# Warning helpers — each returns 0 or 1 message so the caller stays flat.
# =============================================================================


def _warn_controller_pod(est: ClusterMemoryEstimate) -> list[str]:
    if not est.controller.at_risk:
        return []
    return [
        f"Controller pod peak ({est.controller.total_peak_mib:.0f} MiB) is within "
        f"{est.controller.headroom_pct:.1f}% of limit ({est.controller.current_limit_mib:.0f} MiB). "
        "Risk of OOM kill."
    ]


def _warn_records_manager(
    est: ClusterMemoryEstimate, p: MemoryEstimationParams
) -> list[str]:
    rm = next(
        (c for c in est.controller.components if c.name == "RecordsManager"), None
    )
    if rm is None or est.controller.current_limit_mib <= 0:
        return []
    rm_pct = rm.steady_state_mib / est.controller.current_limit_mib * 100
    if rm_pct <= _RECORDS_MANAGER_WARN_PCT:
        return []
    return [
        f"RecordsManager uses {rm_pct:.0f}% of controller limit "
        f"({rm.steady_state_mib:.0f}/{est.controller.current_limit_mib:.0f} MiB). "
        f"Driven by {p.total_requests:,} total requests."
    ]


def _warn_worker_pod(est: ClusterMemoryEstimate) -> list[str]:
    if not est.worker_pod.at_risk:
        return []
    return [
        f"Worker pod peak ({est.worker_pod.total_peak_mib:.0f} MiB) is within "
        f"{est.worker_pod.headroom_pct:.1f}% of limit ({est.worker_pod.current_limit_mib:.0f} MiB)."
    ]


def _warn_request_volume(p: MemoryEstimationParams) -> list[str]:
    if p.total_requests <= 500_000:
        return []
    return [
        f"Total requests ({p.total_requests:,}) will create significant metric array storage. "
        f"~{p.num_standard_metrics} metrics x {_ceil_pow2(p.total_requests):,} capacity x 8B each."
    ]


def _warn_tokenizer(p: MemoryEstimationParams) -> list[str]:
    per_rp_tokenizer_mib = p.num_models * _TOKENIZER_CACHE_MIB
    if per_rp_tokenizer_mib <= 450:
        return []
    model_word = "model" if p.num_models == 1 else "models"
    return [
        f"Each RecordProcessor loads {per_rp_tokenizer_mib} MiB in tokenizer cache "
        f"({p.num_models} {model_word} x {_TOKENIZER_CACHE_MIB} MiB). "
        f"With {p.record_processors_per_pod} RP(s)/pod, that is "
        f"{per_rp_tokenizer_mib * p.record_processors_per_pod} MiB per worker pod."
    ]


def _warn_http_trace(p: MemoryEstimationParams) -> list[str]:
    if not (p.export_http_trace and p.total_requests > 10_000):
        return []
    return [
        f"HTTP trace export with {p.total_requests:,} requests will accumulate "
        "per-chunk timing data in memory. Consider disabling for large runs."
    ]


def _warn_multi_turn(p: MemoryEstimationParams) -> list[str]:
    if p.max_turns <= 1:
        return []
    sessions_per_worker = max(1, p.max_concurrency // max(p.total_workers, 1))
    if sessions_per_worker <= 100:
        return []
    return [
        f"Multi-turn with {sessions_per_worker} concurrent sessions per worker. "
        "Session cache may consume significant memory."
    ]


def _build_recommendations(est: ClusterMemoryEstimate) -> list[str]:
    recommendations: list[str] = []
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
    return recommendations
