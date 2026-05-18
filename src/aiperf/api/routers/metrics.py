# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metrics router component -- owns real-time metrics state and Prometheus/JSON endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse
from pydantic import Field

from aiperf import __version__ as aiperf_version
from aiperf.api.routers.base_router import BaseRouter, component_dependency
from aiperf.common.mixins.realtime_metrics_mixin import RealtimeMetricsMixin
from aiperf.common.models import MetricResult
from aiperf.common.models.base_models import AIPerfBaseModel
from aiperf.config.loader.parsing import coerce_value
from aiperf.config.phases import RatePhaseConfig
from aiperf.metrics.prometheus_formatter import InfoLabels, format_as_prometheus

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun

MetricsDep = Annotated["MetricsRouter", component_dependency("metrics")]


class MetricsResponse(AIPerfBaseModel):
    """JSON metrics endpoint response."""

    aiperf_version: str = Field(description="AIPerf version string")
    benchmark_id: str | None = Field(default=None, description="Benchmark identifier")
    model: str | None = Field(default=None, description="Model name(s)")
    endpoint_type: str | None = Field(default=None, description="Endpoint type")
    streaming: bool | None = Field(
        default=None, description="Whether streaming is enabled"
    )
    concurrency: int | None = Field(
        default=None, ge=1, description="Concurrency setting"
    )
    request_rate: float | None = Field(
        default=None, gt=0, description="Request rate setting"
    )
    metrics: dict[str, Any] = Field(
        default_factory=dict, description="Metrics keyed by tag"
    )


metrics_router = APIRouter()


class MetricsRouter(RealtimeMetricsMixin, BaseRouter):
    """Owns real-time metrics state and exposes /metrics and /api/metrics."""

    def __init__(
        self,
        *,
        run: BenchmarkRun,
        **kwargs,
    ) -> None:
        super().__init__(run=run, **kwargs)
        self._info_labels: InfoLabels | None = None

    def get_info_labels(self) -> InfoLabels:
        """Get cached info labels for metrics."""
        if self._info_labels is None:
            self._info_labels = build_info_labels(self.run)
        return self._info_labels

    def get_router(self) -> APIRouter:
        return metrics_router


@metrics_router.get("/metrics", response_class=PlainTextResponse, tags=["Metrics"])
async def prometheus_metrics(component: MetricsDep) -> PlainTextResponse:
    """Get metrics in Prometheus exposition format."""
    return PlainTextResponse(
        format_as_prometheus(
            metrics=list(component._metrics),
            info_labels=component.get_info_labels(),
        )
    )


@metrics_router.get("/api/metrics", response_model=MetricsResponse, tags=["Metrics"])
async def json_metrics(component: MetricsDep) -> MetricsResponse:
    """Get metrics in JSON format."""
    return format_metrics_json(
        metrics=list(component._metrics),
        info_labels=component.get_info_labels(),
        benchmark_id=component.run.benchmark_id,
    )


def build_info_labels(run: BenchmarkRun) -> InfoLabels:
    """Build info labels for metrics from a BenchmarkRun.

    These labels identify the benchmark and are included in Prometheus metrics.
    Concurrency and request_rate come from the first profiling phase, which
    represents the active variant for this run.

    Args:
        run: The BenchmarkRun for the active iteration.

    Returns:
        Dictionary of label names to values for the info metric.
    """
    cfg = run.cfg
    labels: InfoLabels = {}

    if run.benchmark_id:
        labels["benchmark_id"] = run.benchmark_id

    labels["model"] = ",".join(sorted(cfg.get_model_names()))
    labels["endpoint_type"] = cfg.endpoint.type
    labels["streaming"] = str(cfg.endpoint.streaming).lower()

    profiling_phases = cfg.get_profiling_phases()
    head_phase = profiling_phases[0] if profiling_phases else None
    if head_phase is not None:
        if head_phase.concurrency is not None:
            labels["concurrency"] = str(head_phase.concurrency)
        if isinstance(head_phase, RatePhaseConfig):
            labels["request_rate"] = str(head_phase.rate)

    return labels


def format_metrics_json(
    metrics: list[MetricResult],
    info_labels: InfoLabels | None = None,
    benchmark_id: str | None = None,
) -> MetricsResponse:
    """Format metrics as a structured response.

    Args:
        metrics: List of MetricResult objects from realtime metrics.
        info_labels: Optional dict of labels for additional metadata.
        benchmark_id: Optional benchmark ID to include.

    Returns:
        Structured MetricsResponse.
    """
    labels: dict[str, Any] = {}
    if info_labels:
        labels = {
            key: coerce_value(value)
            for key, value in info_labels.items()
            if key not in ("config", "version", "benchmark_id")
        }

    metrics_dict = {}
    for metric in metrics:
        metrics_dict[metric.tag] = metric.model_dump(
            mode="json", exclude_none=True, exclude={"tag"}
        )

    return MetricsResponse(
        aiperf_version=aiperf_version,
        benchmark_id=benchmark_id,
        metrics=metrics_dict,
        **labels,
    )
