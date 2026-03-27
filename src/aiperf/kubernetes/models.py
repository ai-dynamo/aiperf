# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Data models for Kubernetes operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import Field, field_validator

from aiperf.kubernetes.constants import Annotations, Labels, ProgressAnnotations
from aiperf.kubernetes.enums import JobSetStatus
from aiperf.kubernetes.k8s_models import K8sCamelModel


@dataclass
class JobSetInfo:
    """Information about a found JobSet.

    Use ``JobSetInfo.from_raw(raw_dict)`` to create from a Kubernetes API
    response dict.  All field extraction and status parsing is handled here.
    """

    name: str
    """Kubernetes JobSet resource name."""

    namespace: str
    """Kubernetes namespace containing the JobSet."""

    jobset: dict[str, Any]
    """Raw JobSet dict from the Kubernetes API."""

    status: str
    """Current status: "Running", "Completed", or "Failed"."""

    custom_name: str | None = None
    """User-provided benchmark name, if set."""

    model: str | None = None
    """Target model name from the endpoint, if set."""

    endpoint: str | None = None
    """Target LLM endpoint URL, if set."""

    # -- Factory ----------------------------------------------------------

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> JobSetInfo:
        """Create a JobSetInfo from a raw Kubernetes JobSet dict."""
        metadata = raw.get("metadata", {})
        labels = metadata.get("labels", {})
        annotations = metadata.get("annotations", {})
        return cls(
            name=metadata["name"],
            namespace=metadata["namespace"],
            jobset=raw,
            status=cls._parse_status(raw),
            custom_name=labels.get(Labels.NAME),
            model=annotations.get(Annotations.MODEL),
            endpoint=annotations.get(Annotations.ENDPOINT),
        )

    # -- Derived properties -----------------------------------------------

    @property
    def job_id(self) -> str:
        """AIPerf job ID (falls back to the JobSet name)."""
        labels = self.jobset.get("metadata", {}).get("labels", {})
        return labels.get(Labels.JOB_ID, self.name)

    @property
    def created(self) -> str:
        """Creation timestamp from the JobSet metadata."""
        return self.jobset.get("metadata", {}).get("creationTimestamp", "")

    @property
    def progress(self) -> str | None:
        """Human-readable progress string, or None if unavailable."""
        annotations = self.jobset.get("metadata", {}).get("annotations", {})
        if not annotations.get(ProgressAnnotations.STATUS):
            return None

        parts: list[str] = []
        phase = annotations.get(ProgressAnnotations.PHASE, "")
        if phase:
            parts.append(phase)
        requests = annotations.get(ProgressAnnotations.REQUESTS)
        if requests:
            parts.append(requests)
        percent = annotations.get(ProgressAnnotations.PERCENT)
        if percent:
            parts.append(f"({percent}%)")
        return " ".join(parts) if parts else annotations.get(ProgressAnnotations.STATUS)

    # -- Private helpers --------------------------------------------------

    @staticmethod
    def _parse_status(raw: dict[str, Any]) -> str:
        """Extract status string from a raw JobSet dict."""
        status = raw.get("status", {})
        conditions = status.get("conditions", [])
        condition_status = {c.get("type"): c.get("status") for c in conditions}
        if condition_status.get("Completed") == "True":
            return JobSetStatus.COMPLETED
        if condition_status.get("Failed") == "True":
            replicated = {
                rj.get("name"): rj for rj in status.get("replicatedJobsStatus", [])
            }
            if replicated.get("controller", {}).get("failed", 0) > 0:
                return JobSetStatus.FAILED
        return JobSetStatus.RUNNING


# =============================================================================
# AIPerfJob CR structure — parsed via AIPerfJobCR.model_validate(raw_dict)
#
# Reuses operator models where they exist:
#   - PhaseProgress (operator/models.py) for status.phases values
#   - MetricsSummary (operator/models.py) for summary extraction
# Defines only what doesn't exist: metadata, spec subset, status envelope.
# =============================================================================


class CRMetadata(K8sCamelModel):
    """Kubernetes object metadata (subset relevant to AIPerfJob)."""

    name: str = Field(default="", description="Resource name.")
    namespace: str = Field(default="", description="Resource namespace.")
    creation_timestamp: str = Field(default="", description="Creation timestamp.")


class CREndpoint(K8sCamelModel):
    """Endpoint section from AIPerfJob spec."""

    url: str | None = Field(default=None, description="Single endpoint URL.")
    urls: list[str] = Field(default_factory=list, description="List of endpoint URLs.")


class CRBenchmark(K8sCamelModel):
    """Benchmark section from AIPerfJob spec (nested under spec.benchmark)."""

    models: str | list | dict[str, Any] = Field(
        default_factory=list, description="Model name(s) to benchmark."
    )
    endpoint: CREndpoint | dict[str, Any] = Field(
        default_factory=CREndpoint, description="Endpoint configuration."
    )


class CRSpec(K8sCamelModel):
    """AIPerfJob spec (subset relevant for display).

    AIPerfConfig fields are nested under spec.benchmark. Deployment fields
    (image, podTemplate, etc.) live at the spec level.
    """

    benchmark: CRBenchmark = Field(
        default_factory=CRBenchmark, description="Benchmark configuration."
    )


class CRWorkerStatus(K8sCamelModel):
    """Worker readiness counts from status.workers."""

    ready: int = Field(default=0, description="Number of ready workers.")
    total: int = Field(default=0, description="Total number of workers.")


class CRJobStatus(K8sCamelModel):
    """AIPerfJob status subresource.

    Phase progress dicts (status.phases) are written by the operator via
    PhaseProgress.to_k8s_dict() (camelCase keys including
    ``requestsProgressPercent``). Summary dicts are written via
    MetricsSummary.to_status_dict() (snake_case keys including
    ``throughput_rps``, ``latency_p99_ms``). Both are kept as raw dicts
    to avoid a circular import with the operator package.
    """

    phase: str = Field(default="Pending", description="Current lifecycle phase.")
    job_id: str = Field(default="", description="Operator-assigned job ID.")
    job_set_name: str | None = Field(
        default=None, description="Name of the managed JobSet."
    )
    workers: CRWorkerStatus = Field(
        default_factory=CRWorkerStatus, description="Worker readiness."
    )
    current_phase: str | None = Field(
        default=None, description="Current benchmark phase name."
    )
    error: str | None = Field(default=None, description="Error message if failed.")
    start_time: str | None = Field(default=None, description="Job start timestamp.")
    completion_time: str | None = Field(
        default=None, description="Job completion timestamp."
    )
    live_summary: dict[str, Any] | None = Field(
        default=None,
        description="Live metrics (MetricsSummary.to_status_dict() format).",
    )
    summary: dict[str, Any] | None = Field(
        default=None,
        description="Final metrics (MetricsSummary.to_status_dict() format).",
    )
    phases: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-phase progress (PhaseProgress.to_k8s_dict() format).",
    )

    @field_validator("phase", mode="before")
    @classmethod
    def coerce_none_phase(cls, v: str | None) -> str:
        """Coerce None or empty phase to 'Pending'."""
        return v or "Pending"


class AIPerfJobCR(K8sCamelModel):
    """Parsed AIPerfJob custom resource.

    Use ``AIPerfJobCR.model_validate(raw_dict)`` to parse a raw K8s API
    response dict. Then call ``to_info()`` for a flat CLI display model.
    """

    metadata: CRMetadata = Field(
        default_factory=CRMetadata, description="K8s object metadata."
    )
    spec: CRSpec = Field(default_factory=CRSpec, description="Job specification.")
    status: CRJobStatus = Field(
        default_factory=CRJobStatus, description="Job status subresource."
    )

    def to_info(self) -> AIPerfJobInfo:
        """Convert to flat AIPerfJobInfo for CLI display."""
        models = self.spec.benchmark.models
        if isinstance(models, str):
            model = models
        elif isinstance(models, dict):
            items = models.get("items", [])
            model = items[0].get("name") if items else None
        elif models:
            first = models[0]
            model = first if isinstance(first, str) else None
        else:
            model = None

        ep = self.spec.benchmark.endpoint
        if isinstance(ep, dict):
            endpoint_url = ep.get("url") or (ep.get("urls", [None])[0])
        else:
            endpoint_url = ep.url or (ep.urls[0] if ep.urls else None)

        # Progress: read requestsProgressPercent written by PhaseProgress
        progress: float | None = None
        for p in self.status.phases.values():
            pct = p.get("requestsProgressPercent")
            if pct is not None:
                progress = float(pct)

        # Summary: operator writes snake_case via MetricsSummary.to_status_dict()
        s = self.status.live_summary or self.status.summary or {}
        throughput = s.get("throughput_rps")
        latency = s.get("latency_p99_ms")

        return AIPerfJobInfo(
            name=self.metadata.name,
            namespace=self.metadata.namespace,
            phase=self.status.phase,
            job_id=self.status.job_id or self.metadata.name,
            jobset_name=self.status.job_set_name,
            workers_ready=self.status.workers.ready,
            workers_total=self.status.workers.total,
            current_phase=self.status.current_phase,
            error=self.status.error,
            start_time=self.status.start_time,
            completion_time=self.status.completion_time,
            created=self.metadata.creation_timestamp,
            progress_percent=progress,
            throughput_rps=float(throughput) if throughput is not None else None,
            latency_p99_ms=float(latency) if latency is not None else None,
            model=model,
            endpoint=endpoint_url,
        )


# =============================================================================
# AIPerfJobInfo — flat display model for CLI consumption
# =============================================================================


class AIPerfJobInfo(K8sCamelModel):
    """Flat view of an AIPerfJob for CLI display.

    Constructed via ``AIPerfJobCR.model_validate(raw).to_info()`` for
    data from the K8s API, or directly with kwargs for fallback paths.
    """

    name: str = Field(description="AIPerfJob resource name.")
    namespace: str = Field(description="Kubernetes namespace containing the AIPerfJob.")
    phase: str = Field(description="Current lifecycle phase.")
    job_id: str = Field(description="Operator-assigned job ID.")
    jobset_name: str | None = Field(
        default=None, description="Name of the managed JobSet from .status.jobSetName."
    )
    workers_ready: int = Field(default=0, description="Number of ready workers.")
    workers_total: int = Field(default=0, description="Total number of workers.")
    current_phase: str | None = Field(
        default=None,
        description="Current benchmark phase name (e.g. warmup, profiling).",
    )
    error: str | None = Field(
        default=None, description="Error message if the job failed."
    )
    start_time: str | None = Field(
        default=None, description="ISO 8601 timestamp when the job started."
    )
    completion_time: str | None = Field(
        default=None, description="ISO 8601 timestamp when the job completed."
    )
    created: str = Field(default="", description="Creation timestamp from metadata.")
    progress_percent: float | None = Field(
        default=None, description="Overall progress percentage (0-100)."
    )
    throughput_rps: float | None = Field(
        default=None, description="Live or final throughput in requests per second."
    )
    latency_p99_ms: float | None = Field(
        default=None, description="Live or final p99 latency in milliseconds."
    )
    model: str | None = Field(default=None, description="Target model name from spec.")
    endpoint: str | None = Field(
        default=None, description="Target endpoint URL from spec."
    )

    @property
    def workers_str(self) -> str:
        """Format as 'ready/total'."""
        return f"{self.workers_ready}/{self.workers_total}"


@dataclass
class PodSummary:
    """Summary of pod readiness for a JobSet."""

    ready: int
    """Number of pods with all containers ready and phase Running."""

    total: int
    """Total number of pods belonging to the JobSet."""

    restarts: int
    """Sum of container restart counts across all pods."""

    @property
    def ready_str(self) -> str:
        """Format as 'ready/total'."""
        return f"{self.ready}/{self.total}"
