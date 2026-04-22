# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pydantic models for AIPerfJob operator.

This module provides validated models for:
- AIPerfJob spec validation
- Metrics summary extraction
- Results TTL configuration
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass
from typing import Any

from pydantic import Field, field_validator, model_validator

from aiperf.common.models import AIPerfBaseModel
from aiperf.kubernetes.enums import ImagePullPolicy
from aiperf.kubernetes.k8s_models import K8sCamelModel


class OwnerReference(K8sCamelModel):
    """Kubernetes owner reference for cascade deletion."""

    api_version: str = Field(
        description="The API group and version (e.g. 'aiperf.nvidia.com/v1alpha1')"
    )
    kind: str = Field(description="The kind of the owner resource (e.g. 'AIPerfJob')")
    name: str = Field(description="The name of the owner resource")
    uid: str = Field(description="The UID of the owner resource")
    controller: bool = Field(
        default=True,
        description="Whether this reference points to the managing controller",
    )
    block_owner_deletion: bool = Field(
        default=True,
        description="Whether the owner's deletion is blocked until this resource is removed",
    )

    @classmethod
    def for_aiperf_job(cls, name: str, uid: str) -> OwnerReference:
        """Create an owner reference for an AIPerfJob CR."""
        from aiperf.kubernetes.cr_refs import AIPERF_JOB_API_VERSION

        return cls(
            api_version=AIPERF_JOB_API_VERSION,
            kind="AIPerfJob",
            name=name,
            uid=uid,
        )


@dataclass(slots=True)
class HealthCheckResult:
    """Result of an endpoint health check."""

    reachable: bool
    """Whether the endpoint responded to at least one health probe."""

    error: str
    """Error message if unreachable, empty string otherwise."""


@dataclass(slots=True)
class FetchResult:
    """Result of fetching metrics and files from controller pod."""

    metrics: dict[str, Any] | None
    """Full metrics dict from the controller's /api/metrics endpoint."""

    downloaded: list[str]
    """List of file paths successfully downloaded to the results directory."""

    checkpoints: list[str] = dataclasses.field(default_factory=list)
    """Checkpoint artifact paths downloaded for partial recovery."""

    error: str = ""
    """Error message if fetch failed or returned partial results."""


class PhaseProgress(K8sCamelModel):
    """Progress data for a single benchmark phase."""

    requests_completed: int = Field(
        description="Number of requests that received a complete response"
    )
    requests_sent: int = Field(
        description="Number of requests dispatched to the endpoint"
    )
    requests_total: int = Field(
        description="Total number of requests expected for this phase"
    )
    requests_cancelled: int = Field(
        description="Number of requests cancelled before completion"
    )
    requests_errors: int = Field(
        description="Number of requests that failed with an error"
    )
    requests_in_flight: int = Field(
        description="Number of requests currently awaiting a response"
    )
    requests_per_second: float = Field(description="Current request throughput")
    requests_progress_percent: float = Field(
        description="Percentage of total requests completed (0.0 to 100.0)"
    )
    sessions_sent: int = Field(description="Number of multi-turn sessions dispatched")
    sessions_completed: int = Field(
        description="Number of sessions that finished all turns"
    )
    sessions_cancelled: int = Field(
        description="Number of sessions cancelled before completion"
    )
    sessions_in_flight: int = Field(
        description="Number of sessions currently in progress"
    )
    records_success: int = Field(
        description="Number of individual records completed successfully"
    )
    records_error: int = Field(description="Number of individual records that failed")
    records_per_second: float = Field(description="Current record throughput")
    records_progress_percent: float = Field(
        description="Percentage of total records completed (0.0 to 100.0)"
    )
    sending_complete: bool = Field(
        description="Whether all requests have been dispatched"
    )
    timeout_triggered: bool = Field(
        description="Whether the phase ended due to a timeout"
    )
    was_cancelled: bool = Field(
        description="Whether the phase was cancelled by the user"
    )
    requests_eta_seconds: int | None = Field(
        default=None, description="Estimated seconds until all requests complete"
    )
    records_eta_seconds: int | None = Field(
        default=None, description="Estimated seconds until all records complete"
    )
    expected_duration_seconds: float | None = Field(
        default=None, description="Expected total duration of the phase in seconds"
    )
    elapsed_time_seconds: float | None = Field(
        default=None, description="Wall-clock time elapsed since the phase started"
    )


@dataclass(slots=True)
class MetricsSummary:
    """Extracted key metrics for CRD status.

    These are the most important metrics that users want to see at a glance
    via `kubectl get aiperfjob -o wide` or in dashboards.
    """

    throughput_rps: float | None = None
    """Request throughput (requests/second)."""

    throughput_tps: float | None = None
    """Token throughput (tokens/second)."""

    latency_avg_ms: float | None = None
    """Average request latency (milliseconds)."""

    latency_p50_ms: float | None = None
    """P50 request latency (milliseconds)."""

    latency_p99_ms: float | None = None
    """P99 request latency (milliseconds)."""

    ttft_avg_ms: float | None = None
    """Average time to first token (milliseconds)."""

    ttft_p50_ms: float | None = None
    """P50 time to first token (milliseconds)."""

    ttft_p99_ms: float | None = None
    """P99 time to first token (milliseconds)."""

    total_requests: int | None = None
    """Total requests completed."""

    error_rate: float | None = None
    """Error rate (0.0 to 1.0)."""

    @classmethod
    def from_metrics(cls, metrics: dict[str, Any]) -> MetricsSummary:
        """Extract summary from full metrics response."""
        if not metrics:
            return cls()

        def get_metric(tag_pattern: str, stat: str = "avg") -> float | None:
            raw = metrics.get("metrics", [])
            # Live metrics API returns a dict keyed by metric name;
            # results API returns a list of dicts with "tag" fields.
            items: list[tuple[str, dict]] = []
            if isinstance(raw, dict):
                items = [(k, v) for k, v in raw.items() if isinstance(v, dict)]
            elif isinstance(raw, list):
                items = [(m.get("tag", ""), m) for m in raw if isinstance(m, dict)]

            for tag, m in items:
                if re.search(tag_pattern, tag, re.IGNORECASE):
                    if stat in m:
                        return m[stat]
                    if "value" in m:
                        return m["value"]
            return None

        return cls(
            throughput_rps=get_metric(r"request.*throughput|throughput.*request"),
            throughput_tps=get_metric(r"output.*token.*throughput|token.*throughput"),
            latency_avg_ms=get_metric(r"request.*latency", "avg"),
            latency_p50_ms=get_metric(r"request.*latency", "p50"),
            latency_p99_ms=get_metric(r"request.*latency", "p99"),
            ttft_avg_ms=get_metric(r"time.*first.*token|ttft", "avg"),
            ttft_p50_ms=get_metric(r"time.*first.*token|ttft", "p50"),
            ttft_p99_ms=get_metric(r"time.*first.*token|ttft", "p99"),
            total_requests=int(get_metric(r"total.*requests") or 0) or None,
            error_rate=get_metric(r"error.*rate"),
        )

    def to_status_dict(self) -> dict[str, Any]:
        """Convert to dict for CRD status, excluding None values."""
        return {k: v for k, v in dataclasses.asdict(self).items() if v is not None}


class EndpointConfig(AIPerfBaseModel):
    """Validated endpoint configuration."""

    __slots__ = ()

    url: str = Field(description="LLM endpoint URL")
    model: str | None = Field(default=None, description="Model name")
    api_type: str = Field(default="openai", description="API type (openai, triton)")

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate URL format."""
        if not v:
            raise ValueError("Endpoint URL is required")
        if not v.startswith(("http://", "https://")):
            raise ValueError("Endpoint URL must start with http:// or https://")
        return v


class AIPerfJobSpec(AIPerfBaseModel):
    """Validated AIPerfJob spec from nested CRD.

    Validates required fields and value ranges before creating resources.
    The CRD spec is nested — AIPerfConfig fields live under spec.benchmark
    and deployment fields live at the spec level.
    """

    __slots__ = ()

    image: str = Field(
        default="nvcr.io/nvidia/aiperf:latest",
        description="Container image for AIPerf",
    )
    image_pull_policy: ImagePullPolicy | None = Field(
        default=None, description="Image pull policy"
    )
    endpoint: dict[str, Any] = Field(description="Endpoint configuration")
    timeout_seconds: float = Field(
        default=0, ge=0, description="Job timeout in seconds (0 = no timeout)"
    )
    ttl_seconds_after_finished: int | None = Field(
        default=None, ge=0, description="TTL for completed jobs"
    )
    results_ttl_days: int | None = Field(
        default=None, ge=1, le=365, description="TTL for results in PVC (days)"
    )
    cancel: bool = Field(default=False, description="Set to true to cancel the job")

    @field_validator("image")
    @classmethod
    def validate_image(cls, v: str) -> str:
        """Validate image is not empty."""
        if not v or not v.strip():
            raise ValueError("Image is required")
        return v

    @model_validator(mode="after")
    def validate_endpoint_config(self) -> AIPerfJobSpec:
        """Validate endpoint has required fields."""
        endpoint = self.endpoint
        if not endpoint:
            raise ValueError("endpoint is required")

        if not endpoint.get("url") and not endpoint.get("urls"):
            raise ValueError("endpoint.url or endpoint.urls is required")

        return self

    @classmethod
    def from_crd_spec(cls, spec: dict[str, Any]) -> AIPerfJobSpec:
        """Create from nested CRD spec dict.

        Reads deployment fields from the top-level spec and endpoint
        from spec.benchmark.

        Args:
            spec: Raw spec from Kubernetes CRD (deployment fields at top level,
                AIPerfConfig fields under spec.benchmark).

        Returns:
            Validated AIPerfJobSpec.

        Raises:
            ValueError: If validation fails.
        """
        benchmark = spec.get("benchmark") or {}
        return cls(
            image=spec.get("image", "nvcr.io/nvidia/aiperf:latest"),
            image_pull_policy=spec.get("imagePullPolicy"),
            endpoint=benchmark.get("endpoint", {}),
            timeout_seconds=spec.get("timeoutSeconds", 0),
            ttl_seconds_after_finished=spec.get("ttlSecondsAfterFinished"),
            results_ttl_days=spec.get("resultsTtlDays"),
            cancel=spec.get("cancel", False),
        )

    def get_endpoint_url(self) -> str | None:
        """Extract primary endpoint URL from flat spec."""
        return self.endpoint.get("url") or (self.endpoint.get("urls") or [None])[0]
