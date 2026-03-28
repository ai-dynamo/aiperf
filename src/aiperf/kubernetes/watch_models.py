# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Data models for the aiperf kube watch command."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class ProgressSnapshot:
    """Benchmark progress at a point in time."""

    percent: float = 0.0
    """Overall completion percentage (0-100)."""

    requests_completed: int = 0
    """Number of requests completed so far."""

    requests_total: int = 0
    """Target total number of requests."""

    eta_seconds: float | None = None
    """Estimated seconds remaining, if available."""

    duration_target_seconds: float | None = None
    """Duration-based phase target in seconds, if applicable."""


@dataclass(frozen=True)
class MetricsSnapshot:
    """Real-time performance metrics from the benchmark."""

    request_throughput_rps: float = 0.0
    """Requests per second throughput."""

    request_latency_avg_ms: float = 0.0
    """Average end-to-end request latency in milliseconds."""

    request_latency_p50_ms: float = 0.0
    """Median (p50) request latency in milliseconds."""

    request_latency_p99_ms: float = 0.0
    """99th percentile request latency in milliseconds."""

    ttft_avg_ms: float = 0.0
    """Average time to first token in milliseconds."""

    ttft_p50_ms: float = 0.0
    """Median (p50) time to first token in milliseconds."""

    ttft_p99_ms: float = 0.0
    """99th percentile time to first token in milliseconds."""

    time_to_second_token_avg_ms: float = 0.0
    """Average time to second token in milliseconds."""

    inter_token_latency_avg_ms: float = 0.0
    """Average inter-token latency in milliseconds."""

    inter_token_latency_p99_ms: float = 0.0
    """99th percentile inter-token latency in milliseconds."""

    output_token_throughput_tps: float = 0.0
    """Aggregate output token throughput in tokens per second."""

    total_token_throughput_tps: float = 0.0
    """Aggregate total (input + output) token throughput in tokens per second."""

    prefill_throughput_per_user_tps: float = 0.0
    """Per-user prefill throughput in tokens per second."""

    output_token_throughput_per_user_tps: float = 0.0
    """Per-user output token throughput in tokens per second."""

    request_count: int = 0
    """Total completed requests."""

    error_count: int = 0
    """Total failed requests."""

    goodput_rps: float = 0.0
    """Successful requests per second (excludes errors)."""

    streaming: bool = False
    """Whether the benchmark is using streaming responses."""


@dataclass(frozen=True)
class WorkersSnapshot:
    """Worker readiness state."""

    ready: int = 0
    """Number of workers reporting ready."""

    total: int = 0
    """Total number of expected workers."""


@dataclass(frozen=True)
class PodSnapshot:
    """Status of a single Kubernetes pod."""

    name: str
    """Pod name from Kubernetes metadata."""

    role: str
    """Pod role: 'controller' or 'worker'."""

    status: str
    """Current pod phase (Pending, Running, Succeeded, Failed)."""

    restarts: int = 0
    """Total container restart count across all containers."""

    ready: bool = False
    """Whether all containers in the pod are ready."""

    oom_killed: bool = False
    """Whether any container was OOM-killed."""

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> PodSnapshot:
        metadata = raw.get("metadata", {})
        status = raw.get("status", {})
        name = metadata.get("name", "")
        role = "controller" if "controller" in name else "worker"
        containers = status.get("containerStatuses", [])
        restarts = sum(c.get("restartCount", 0) for c in containers)
        ready = all(c.get("ready", False) for c in containers)
        oom = any(
            c.get("lastState", {}).get("terminated", {}).get("reason") == "OOMKilled"
            for c in containers
        )
        return cls(
            name=name,
            role=role,
            status=status.get("phase", "Unknown"),
            restarts=restarts,
            ready=ready,
            oom_killed=oom,
        )


@dataclass(frozen=True)
class EventSnapshot:
    """A Kubernetes event relevant to the benchmark."""

    timestamp: str
    """ISO-format timestamp of the event."""

    type: str
    """Event type: 'Normal' or 'Warning'."""

    reason: str
    """Short machine-readable reason string."""

    object: str
    """Name of the involved Kubernetes object."""

    message: str
    """Human-readable event message."""

    count: int = 1
    """Number of times this event has occurred."""


@dataclass(frozen=True)
class DiagnosisIssue:
    """A single diagnosed issue in the benchmark."""

    id: str
    """Unique issue identifier for deduplication."""

    severity: str
    """Severity level: 'info', 'warning', or 'critical'."""

    title: str
    """Short issue title for display."""

    detail: str
    """Detailed description of the issue."""

    impact: str
    """Expected impact on benchmark execution."""

    suggested_fix: str
    """Recommended action to resolve the issue."""

    runbook: str | None = None
    """Optional link to a runbook or documentation."""


@dataclass(frozen=True)
class DiagnosisResult:
    """Aggregated health diagnosis for the benchmark."""

    health: str = "healthy"
    """Overall health: healthy, degraded, stalled, failing, completed, or failed."""

    issues: list[DiagnosisIssue] = field(default_factory=list)
    """Detected issues ordered by severity."""

    stalled: bool = False
    """Whether the benchmark appears stalled."""

    stall_reason: str | None = None
    """Explanation for why the benchmark is stalled."""

    error_rate: float = 0.0
    """Current error rate as a fraction (0.0-1.0)."""


@dataclass(frozen=True)
class WatchSnapshot:
    """Complete point-in-time snapshot of a watched benchmark."""

    timestamp: datetime
    """When this snapshot was captured."""

    job_id: str
    """Unique benchmark job identifier."""

    namespace: str
    """Kubernetes namespace containing the benchmark."""

    phase: str
    """Current JobSet phase."""

    current_phase: str | None = None
    """Active benchmark phase name, if running."""

    elapsed_seconds: float = 0.0
    """Seconds elapsed since the benchmark started."""

    progress: ProgressSnapshot | None = None
    """Request completion progress."""

    metrics: MetricsSnapshot | None = None
    """Real-time performance metrics."""

    workers: WorkersSnapshot = field(default_factory=WorkersSnapshot)
    """Worker readiness summary."""

    pods: list[PodSnapshot] = field(default_factory=list)
    """Status of all pods in the benchmark."""

    events: list[EventSnapshot] = field(default_factory=list)
    """Recent Kubernetes events."""

    conditions: dict[str, bool] = field(default_factory=dict)
    """JobSet condition states."""

    diagnosis: DiagnosisResult = field(default_factory=DiagnosisResult)
    """Automated health diagnosis."""

    raw_metrics: dict[str, Any] = field(default_factory=dict)
    """Unprocessed metric data from the API."""

    server_metrics: dict[str, Any] = field(default_factory=dict)
    """Server-side metrics from Prometheus endpoints."""

    model: str | None = None
    """Target model name."""

    endpoint: str | None = None
    """Target inference endpoint URL."""

    image: str | None = None
    """Container image used for the benchmark."""

    results: dict[str, Any] | None = None
    """Final benchmark results, if completed."""

    error: str | None = None
    """Error message if the benchmark failed."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""

        def _convert_datetimes(obj: Any) -> Any:
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, dict):
                return {k: _convert_datetimes(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_convert_datetimes(i) for i in obj]
            return obj

        return _convert_datetimes(dataclasses.asdict(self))
