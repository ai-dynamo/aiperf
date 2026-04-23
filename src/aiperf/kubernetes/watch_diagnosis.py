# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Diagnosis engine for aiperf kube watch.

Pure-function pattern matcher that analyzes a WatchSnapshot and produces a
DiagnosisResult with health state, detected issues, and actionable fixes.
No I/O, no state -- suitable for use in any context.
"""

from __future__ import annotations

from aiperf.kubernetes.watch_models import (
    DiagnosisIssue,
    DiagnosisResult,
    WatchSnapshot,
)
from aiperf.operator.status import Phase

_STALLED_PENDING_THRESHOLD_S = 60.0
_STALLED_RUNNING_THRESHOLD_S = 30.0
_CRASH_LOOP_RESTART_THRESHOLD = 3
_HIGH_ERROR_RATE_THRESHOLD = 0.05
_HIGH_LATENCY_P99_MULTIPLIER = 10.0


def diagnose(snapshot: WatchSnapshot) -> DiagnosisResult:
    """Analyze a snapshot and return health state with detected issues."""
    issues: list[DiagnosisIssue] = []

    if snapshot.phase == Phase.COMPLETED:
        _check_results_fetch_failed(snapshot, issues)
        return DiagnosisResult(
            health="completed",
            issues=issues,
            error_rate=_compute_error_rate(snapshot),
        )

    if snapshot.phase == Phase.FAILED:
        return DiagnosisResult(
            health="failed",
            issues=issues,
            error_rate=_compute_error_rate(snapshot),
        )

    has_oom, has_crash_loop = _check_pods(snapshot, issues)
    stalled, stall_reason = _check_stall(snapshot, issues)
    _check_conditions(snapshot, issues)
    _check_results_fetch_failed(snapshot, issues)
    error_rate = _check_metrics(snapshot, issues)

    health = _determine_health(
        snapshot=snapshot,
        has_oom=has_oom,
        has_crash_loop=has_crash_loop,
        stalled=stalled,
    )

    return DiagnosisResult(
        health=health,
        issues=issues,
        stalled=stalled,
        stall_reason=stall_reason,
        error_rate=error_rate,
    )


def _check_pods(
    snapshot: WatchSnapshot, issues: list[DiagnosisIssue]
) -> tuple[bool, bool]:
    """Append pod-level issues and return (has_oom, has_crash_loop) flags."""
    has_oom = False
    has_crash_loop = False
    for pod in snapshot.pods:
        if pod.oom_killed:
            has_oom = True
            issues.append(
                DiagnosisIssue(
                    id="oom_restart",
                    severity="warning",
                    title="Worker pod OOM restart",
                    detail=f"Pod {pod.name} was OOMKilled (restarts: {pod.restarts})",
                    impact="Progress may be reset; benchmark data could be lost",
                    suggested_fix="Increase memory limits in deployment config",
                )
            )
        if pod.restarts > _CRASH_LOOP_RESTART_THRESHOLD:
            has_crash_loop = True
            issues.append(
                DiagnosisIssue(
                    id="crash_loop",
                    severity="critical",
                    title="Pod in crash loop",
                    detail=f"Pod {pod.name} has {pod.restarts} restarts",
                    impact="Benchmark cannot make progress while pod keeps crashing",
                    suggested_fix=f"Check pod logs: kubectl logs {pod.name} --previous",
                )
            )
    return has_oom, has_crash_loop


def _check_stall(
    snapshot: WatchSnapshot, issues: list[DiagnosisIssue]
) -> tuple[bool, str | None]:
    """Detect pending/running stalls and return (stalled, stall_reason)."""
    if (
        snapshot.phase == Phase.PENDING
        and snapshot.elapsed_seconds > _STALLED_PENDING_THRESHOLD_S
    ):
        reason = f"Pending for {snapshot.elapsed_seconds:.0f}s (threshold: {_STALLED_PENDING_THRESHOLD_S:.0f}s)"
        issues.append(
            DiagnosisIssue(
                id="stalled_pending",
                severity="warning",
                title="Job stuck in Pending",
                detail=reason,
                impact="Benchmark has not started; may be waiting for resources",
                suggested_fix="Check node resources and pod scheduling events",
            )
        )
        return True, reason

    if (
        snapshot.phase == Phase.RUNNING
        and snapshot.elapsed_seconds > _STALLED_RUNNING_THRESHOLD_S
    ):
        # Only flag as stalled if there's no evidence of active work.
        # If metrics show throughput > 0 or progress is advancing, the
        # benchmark is working — just hasn't updated annotations yet.
        has_throughput = (
            snapshot.metrics is not None and snapshot.metrics.request_throughput_rps > 0
        )
        has_progress = (
            snapshot.progress is not None and snapshot.progress.requests_completed > 0
        )
        if not has_throughput and not has_progress:
            reason = f"Running for {snapshot.elapsed_seconds:.0f}s with no progress"
            issues.append(
                DiagnosisIssue(
                    id="stalled_running",
                    severity="warning",
                    title="Benchmark appears stalled",
                    detail=reason,
                    impact="No forward progress detected",
                    suggested_fix="Check endpoint health and worker pod logs",
                )
            )
            return True, reason

    return False, None


def _check_conditions(snapshot: WatchSnapshot, issues: list[DiagnosisIssue]) -> None:
    """Append issues for failed endpoint/preflight conditions."""
    if snapshot.conditions.get("endpoint_reachable") is False:
        issues.append(
            DiagnosisIssue(
                id="endpoint_unreachable",
                severity="critical",
                title="Inference endpoint unreachable",
                detail="Endpoint health check failed",
                impact="Workers cannot send requests; benchmark will not produce results",
                suggested_fix="Verify the endpoint URL and that the inference server is running",
            )
        )

    if snapshot.conditions.get("preflight_passed") is False:
        issues.append(
            DiagnosisIssue(
                id="preflight_failed",
                severity="critical",
                title="Preflight checks failed",
                detail="One or more preflight validation checks did not pass",
                impact="Benchmark may fail or produce invalid results",
                suggested_fix="Review preflight check output in operator logs",
            )
        )


def _check_metrics(snapshot: WatchSnapshot, issues: list[DiagnosisIssue]) -> float:
    """Append metric-derived issues and return the computed error rate."""
    error_rate = _compute_error_rate(snapshot)

    if error_rate > _HIGH_ERROR_RATE_THRESHOLD:
        issues.append(
            DiagnosisIssue(
                id="high_error_rate",
                severity="warning",
                title="High request error rate",
                detail=f"Error rate: {error_rate:.1%} ({snapshot.metrics.error_count}/{snapshot.metrics.request_count})",
                impact="Benchmark results may be unreliable due to errors",
                suggested_fix="Check endpoint capacity and error responses in logs",
            )
        )

    if snapshot.metrics is not None:
        avg = snapshot.metrics.request_latency_avg_ms
        p99 = snapshot.metrics.request_latency_p99_ms
        if avg > 0 and p99 > 0 and p99 > _HIGH_LATENCY_P99_MULTIPLIER * avg:
            issues.append(
                DiagnosisIssue(
                    id="high_latency",
                    severity="warning",
                    title="High tail latency",
                    detail=f"p99 ({p99:.0f}ms) is >{_HIGH_LATENCY_P99_MULTIPLIER:.0f}x avg ({avg:.0f}ms)",
                    impact="Significant latency outliers may indicate endpoint instability",
                    suggested_fix="Check endpoint load and consider reducing concurrency",
                )
            )

    return error_rate


def _compute_error_rate(snapshot: WatchSnapshot) -> float:
    """Compute error rate from metrics, returning 0.0 if unavailable."""
    if snapshot.metrics is None or snapshot.metrics.request_count == 0:
        return 0.0
    return snapshot.metrics.error_count / snapshot.metrics.request_count


def _check_results_fetch_failed(
    snapshot: WatchSnapshot, issues: list[DiagnosisIssue]
) -> None:
    """Check if results are unavailable for a completed job."""
    if (
        snapshot.phase == Phase.COMPLETED
        and snapshot.conditions.get("results_available") is False
    ):
        issues.append(
            DiagnosisIssue(
                id="results_fetch_failed",
                severity="warning",
                title="Results not available",
                detail="Job completed but results could not be fetched",
                impact="Benchmark data may be lost",
                suggested_fix="Check results storage and operator logs",
            )
        )


def _determine_health(
    *,
    snapshot: WatchSnapshot,
    has_oom: bool,
    has_crash_loop: bool,
    stalled: bool,
) -> str:
    """Determine overall health from individual checks.

    Priority: failing > stalled > degraded > healthy.
    """
    if has_crash_loop:
        return "failing"
    if stalled:
        return "stalled"
    if has_oom:
        return "degraded"
    return "healthy"
