# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark-metric heuristics over an AIPerfJob's ``.status``.

These detectors are the salvaged half of the removed ``aiperf kube watch``
diagnosis engine. The pod-level checks it also carried (crash loop, OOM,
pending-too-long, ImagePull) are not reproduced here.

The operator does not run the retired ``watchdog_*`` stack. Its monitor handler
does independently reconcile pod startup blockers: container waiting states
and PodScheduled=False/Unschedulable conditions become durable CR status and
warnings, with conservative terminalization for stable non-recoverable states.
This module stays focused on benchmark metrics and does not duplicate that pod
inspection.

What is *not* duplicated anywhere else is the benchmark-metric view: error
rate, tail-latency skew, and a throughput-aware notion of "running but making
no progress". ``WatchdogDataSource`` only sees pods, events, nodes and
cpu/memory, so it cannot compute any of these.

Pure functions over the raw CR ``status`` dict: no Kubernetes client, no I/O,
no model layer. That keeps them trivially testable and lets any caller that
can fetch an AIPerfJob feed them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aiperf.kubernetes.environment import K8sEnvironment

__all__ = [
    "BenchmarkFinding",
    "diagnose_benchmark",
    "error_rate",
]


@dataclass(slots=True)
class BenchmarkFinding:
    """A single benchmark-metric problem detected on an AIPerfJob."""

    id: str
    """Stable identifier, e.g. ``high_error_rate``."""

    severity: str
    """One of ``info``, ``warning``, ``critical``."""

    title: str
    """Short human-readable headline."""

    detail: str
    """The measurement that triggered the finding."""

    impact: str
    """Why the operator should care."""

    suggested_fix: str
    """The next action to take."""


def _metric_avg(metrics: dict[str, Any], key: str) -> float:
    """Read ``metrics[key]["avg"]``, tolerating absent or non-dict entries."""
    m = metrics.get(key, {})
    return m.get("avg", 0.0) if isinstance(m, dict) else 0.0


def _metric_stat(metrics: dict[str, Any], key: str, stat: str) -> float:
    """Read ``metrics[key][stat]``, tolerating absent or non-dict entries."""
    m = metrics.get(key, {})
    return m.get(stat, 0.0) if isinstance(m, dict) else 0.0


def _live_metrics(status: dict[str, Any]) -> dict[str, Any]:
    live = status.get("liveMetrics", {})
    metrics = live.get("metrics", {}) if isinstance(live, dict) else {}
    return metrics if isinstance(metrics, dict) else {}


def error_rate(status: dict[str, Any]) -> float:
    """Return the observed request error rate, or 0.0 when unavailable.

    ``request_count`` and ``error_count`` are averaged independently from
    staggered liveMetrics windows, so ``error_count > request_count`` is
    observable; the result is clamped to ``[0.0, 1.0]`` rather than reported
    as a nonsense rate above 100%.
    """
    metrics = _live_metrics(status)
    requests = _metric_avg(metrics, "request_count")
    if requests <= 0:
        return 0.0
    return min(1.0, max(0.0, _metric_avg(metrics, "error_count") / requests))


def _check_error_rate(status: dict[str, Any], findings: list[BenchmarkFinding]) -> None:
    rate = error_rate(status)
    if rate <= K8sEnvironment.DIAGNOSIS.HIGH_ERROR_RATE_THRESHOLD:
        return
    metrics = _live_metrics(status)
    errors = int(_metric_avg(metrics, "error_count"))
    requests = int(_metric_avg(metrics, "request_count"))
    findings.append(
        BenchmarkFinding(
            id="high_error_rate",
            severity="warning",
            title="High request error rate",
            detail=f"Error rate: {rate:.1%} ({errors}/{requests})",
            impact="Benchmark results may be unreliable due to errors",
            suggested_fix="Check endpoint capacity and error responses in logs",
        )
    )


def _check_tail_latency(
    status: dict[str, Any], findings: list[BenchmarkFinding]
) -> None:
    metrics = _live_metrics(status)
    avg = _metric_avg(metrics, "request_latency")
    p99 = _metric_stat(metrics, "request_latency", "p99")
    multiplier = K8sEnvironment.DIAGNOSIS.HIGH_LATENCY_P99_MULTIPLIER
    if avg <= 0 or p99 <= 0 or p99 <= multiplier * avg:
        return
    findings.append(
        BenchmarkFinding(
            id="high_latency",
            severity="warning",
            title="High tail latency",
            detail=f"p99 ({p99:.0f}ms) is >{multiplier:.0f}x avg ({avg:.0f}ms)",
            impact="Latency outliers may indicate endpoint instability",
            suggested_fix="Check endpoint load and consider reducing concurrency",
        )
    )


def _requests_completed(status: dict[str, Any]) -> int:
    """Total requests completed across every benchmark phase."""
    phases = status.get("phases", {})
    if not isinstance(phases, dict):
        return 0
    return sum(
        p.get("requestsCompleted", 0) for p in phases.values() if isinstance(p, dict)
    )


def _check_stalled(
    status: dict[str, Any], elapsed_seconds: float, findings: list[BenchmarkFinding]
) -> None:
    """Flag a job that has been up long enough that silence is suspicious.

    A Running job is only called stalled when there is no evidence of work at
    all: zero throughput *and* zero completed requests. Throughput alone can
    read as 0.0 between liveMetrics windows on a perfectly healthy run, which
    is why the completed-request count is checked as well.
    """
    phase = status.get("phase")
    settings = K8sEnvironment.DIAGNOSIS
    if (
        phase == "Pending"
        and elapsed_seconds > settings.STALLED_PENDING_THRESHOLD_SECONDS
    ):
        findings.append(
            BenchmarkFinding(
                id="stalled_pending",
                severity="warning",
                title="Job stuck in Pending",
                detail=(
                    f"Pending for {elapsed_seconds:.0f}s "
                    f"(threshold: {settings.STALLED_PENDING_THRESHOLD_SECONDS:.0f}s)"
                ),
                impact="Benchmark has not started; may be waiting for resources",
                suggested_fix="Check node resources and pod scheduling events",
            )
        )
        return

    if (
        phase != "Running"
        or elapsed_seconds <= settings.STALLED_RUNNING_THRESHOLD_SECONDS
    ):
        return
    throughput = _metric_avg(_live_metrics(status), "request_throughput")
    if throughput > 0 or _requests_completed(status) > 0:
        return
    findings.append(
        BenchmarkFinding(
            id="stalled_running",
            severity="warning",
            title="Benchmark appears stalled",
            detail=f"Running for {elapsed_seconds:.0f}s with no progress",
            impact="No forward progress detected",
            suggested_fix="Check endpoint health and worker pod logs",
        )
    )


def diagnose_benchmark(
    status: dict[str, Any], *, elapsed_seconds: float = 0.0
) -> list[BenchmarkFinding]:
    """Run every benchmark-metric detector over an AIPerfJob ``.status``.

    Args:
        status: The AIPerfJob CR's ``.status`` mapping. Missing or partially
            populated keys are tolerated; a status with no ``liveMetrics``
            simply produces no metric findings.
        elapsed_seconds: Seconds since the job started, used by the stall
            detectors. Pass 0.0 to skip them.

    Returns:
        Findings in detection order. Empty means nothing tripped -- which is
        not the same as "healthy", since pod-level problems are the
        watchdog's job, not this module's.

    Example:
        ```python
        findings = diagnose_benchmark(cr["status"], elapsed_seconds=90.0)
        for f in findings:
            print(f.severity, f.title, f.detail)
        ```
    """
    findings: list[BenchmarkFinding] = []
    _check_stalled(status, elapsed_seconds, findings)
    _check_error_rate(status, findings)
    _check_tail_latency(status, findings)
    return findings
