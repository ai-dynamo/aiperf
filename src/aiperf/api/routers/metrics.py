# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metrics router component -- owns real-time metrics state and Prometheus/JSON endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

import msgspec
from fastapi import APIRouter
from fastapi.responses import PlainTextResponse
from pydantic import Field

from aiperf import __version__ as aiperf_version
from aiperf.api.routers.base_router import BaseRouter, component_dependency
from aiperf.common.mixins.realtime_metrics_mixin import RealtimeMetricsMixin
from aiperf.common.models import MetricResult
from aiperf.common.models.base_models import AIPerfBaseModel, _msgspec_enc_hook
from aiperf.config.parsing import coerce_value
from aiperf.metrics.prometheus_formatter import InfoLabels, format_as_prometheus

if TYPE_CHECKING:
    from aiperf.config import BenchmarkConfig

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
    concurrency: int | None = Field(default=None, description="Concurrency setting")
    request_rate: float | None = Field(default=None, description="Request rate setting")
    metrics: dict[str, Any] = Field(
        default_factory=dict, description="Metrics keyed by tag"
    )


metrics_router = APIRouter()


class MetricsRouter(RealtimeMetricsMixin, BaseRouter):
    """Owns real-time metrics state and exposes /metrics and /api/metrics."""

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._info_labels: InfoLabels | None = None

    def get_info_labels(self) -> InfoLabels:
        """Get cached info labels for metrics."""
        if self._info_labels is None:
            self._info_labels = build_info_labels(self.run.cfg)
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
        benchmark_id=component.run.cfg.benchmark_id,
    )


def build_info_labels(config: BenchmarkConfig) -> InfoLabels:
    """Build info labels for metrics from BenchmarkConfig.

    These labels identify the benchmark and are included in Prometheus metrics.

    Args:
        config: The AIPerf configuration for the benchmark.

    Returns:
        Dictionary of label names to values for the info metric.
    """
    labels: InfoLabels = {}

    if config.benchmark_id:
        labels["benchmark_id"] = config.benchmark_id

    labels["model"] = ",".join(sorted(config.get_model_names()))
    labels["endpoint_type"] = config.endpoint.type
    labels["streaming"] = str(config.endpoint.streaming).lower()

    # Get concurrency/rate from the first profiling phase (if any)
    profiling_phases = config.get_profiling_phases()
    if profiling_phases:
        first_phase = next(iter(profiling_phases.values()))
        if first_phase.concurrency is not None:
            labels["concurrency"] = str(first_phase.concurrency)
        rate = getattr(first_phase, "rate", None)
        if rate is not None:
            labels["request_rate"] = str(rate)

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
        payload = msgspec.to_builtins(metric, enc_hook=_msgspec_enc_hook)
        # Mirror the previous Pydantic ``exclude_none=True, exclude={"tag"}``:
        # drop tag (used as the dict key) and any None values.
        metrics_dict[metric.tag] = {
            k: v for k, v in payload.items() if v is not None and k != "tag"
        }

    return MetricsResponse(
        aiperf_version=aiperf_version,
        benchmark_id=benchmark_id,
        metrics=metrics_dict,
        **labels,
    )
