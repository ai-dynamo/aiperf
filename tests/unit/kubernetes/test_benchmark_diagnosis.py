# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the benchmark-metric diagnosis detectors."""

from typing import Any

import pytest
from pytest import param

from aiperf.kubernetes.benchmark_diagnosis import (
    BenchmarkFinding,
    diagnose_benchmark,
    error_rate,
)


def _status(
    *,
    phase: str = "Running",
    requests: float | None = None,
    errors: float | None = None,
    latency_avg: float | None = None,
    latency_p99: float | None = None,
    throughput: float | None = None,
    completed: int | None = None,
) -> dict[str, Any]:
    """Build an AIPerfJob .status with only the requested metrics present."""
    metrics: dict[str, Any] = {}
    if requests is not None:
        metrics["request_count"] = {"avg": requests}
    if errors is not None:
        metrics["error_count"] = {"avg": errors}
    if latency_avg is not None or latency_p99 is not None:
        entry: dict[str, float] = {}
        if latency_avg is not None:
            entry["avg"] = latency_avg
        if latency_p99 is not None:
            entry["p99"] = latency_p99
        metrics["request_latency"] = entry
    if throughput is not None:
        metrics["request_throughput"] = {"avg": throughput}

    status: dict[str, Any] = {"phase": phase, "liveMetrics": {"metrics": metrics}}
    if completed is not None:
        status["phases"] = {"profiling": {"requestsCompleted": completed}}
    return status


def _ids(findings: list[BenchmarkFinding]) -> set[str]:
    return {f.id for f in findings}


class TestErrorRate:
    """error_rate() over liveMetrics."""

    @pytest.mark.parametrize(
        "requests,errors,expected",
        [
            param(1000.0, 200.0, 0.2, id="ordinary"),
            param(1000.0, 0.0, 0.0, id="no-errors"),
            param(0.0, 50.0, 0.0, id="zero-requests-is-not-divide-by-zero"),
            param(10.0, 50.0, 1.0, id="staggered-windows-clamp-to-one"),
            param(10.0, -5.0, 0.0, id="negative-clamps-to-zero"),
        ],
    )  # fmt: skip
    def test_error_rate_values(
        self, requests: float, errors: float, expected: float
    ) -> None:
        assert error_rate(_status(requests=requests, errors=errors)) == expected

    def test_error_rate_missing_metrics_returns_zero(self) -> None:
        assert error_rate({}) == 0.0


class TestHighErrorRate:
    """The high_error_rate detector."""

    def test_above_threshold_reports_finding(self) -> None:
        findings = diagnose_benchmark(_status(requests=1000, errors=200, throughput=5))
        assert "high_error_rate" in _ids(findings)

    def test_at_threshold_is_silent(self) -> None:
        # default threshold is 0.05; exactly 5% must not trip it
        findings = diagnose_benchmark(_status(requests=1000, errors=50, throughput=5))
        assert "high_error_rate" not in _ids(findings)

    def test_detail_reports_the_counts(self) -> None:
        findings = diagnose_benchmark(_status(requests=1000, errors=200, throughput=5))
        detail = next(f for f in findings if f.id == "high_error_rate").detail
        assert "20.0%" in detail
        assert "200/1000" in detail


class TestHighTailLatency:
    """The high_latency detector."""

    def test_p99_far_above_avg_reports_finding(self) -> None:
        findings = diagnose_benchmark(
            _status(latency_avg=50, latency_p99=900, throughput=5)
        )
        assert "high_latency" in _ids(findings)

    @pytest.mark.parametrize(
        "avg,p99",
        [
            param(50.0, 400.0, id="within-multiplier"),
            param(0.0, 900.0, id="no-avg-yet"),
            param(50.0, 0.0, id="no-p99-yet"),
        ],
    )  # fmt: skip
    def test_silent_cases(self, avg: float, p99: float) -> None:
        findings = diagnose_benchmark(
            _status(latency_avg=avg, latency_p99=p99, throughput=5)
        )
        assert "high_latency" not in _ids(findings)


class TestStallDetection:
    """The stalled_pending / stalled_running detectors."""

    def test_pending_past_threshold_reports_finding(self) -> None:
        findings = diagnose_benchmark(_status(phase="Pending"), elapsed_seconds=120)
        assert "stalled_pending" in _ids(findings)

    def test_pending_within_threshold_is_silent(self) -> None:
        findings = diagnose_benchmark(_status(phase="Pending"), elapsed_seconds=5)
        assert not findings

    def test_running_with_no_work_reports_finding(self) -> None:
        findings = diagnose_benchmark(
            _status(throughput=0, completed=0), elapsed_seconds=120
        )
        assert "stalled_running" in _ids(findings)

    def test_throughput_suppresses_stall(self) -> None:
        findings = diagnose_benchmark(
            _status(throughput=12.5, completed=0), elapsed_seconds=600
        )
        assert "stalled_running" not in _ids(findings)

    def test_completed_requests_suppress_stall_despite_zero_throughput(self) -> None:
        """Throughput reads 0.0 between liveMetrics windows on healthy runs."""
        findings = diagnose_benchmark(
            _status(throughput=0, completed=4200), elapsed_seconds=600
        )
        assert "stalled_running" not in _ids(findings)

    def test_zero_elapsed_skips_stall_detectors(self) -> None:
        findings = diagnose_benchmark(_status(throughput=0, completed=0))
        assert not _ids(findings) & {"stalled_running", "stalled_pending"}

    def test_completed_phase_is_never_stalled(self) -> None:
        findings = diagnose_benchmark(
            _status(phase="Completed", throughput=0, completed=0), elapsed_seconds=9999
        )
        assert not _ids(findings) & {"stalled_running", "stalled_pending"}


class TestRobustness:
    """Malformed or partial status payloads must not raise."""

    @pytest.mark.parametrize(
        "status",
        [
            param({}, id="empty"),
            param({"phase": "Running"}, id="no-liveMetrics"),
            param({"liveMetrics": None}, id="null-liveMetrics"),
            param({"liveMetrics": {"metrics": None}}, id="null-metrics"),
            param({"liveMetrics": {"metrics": {"request_count": 5}}}, id="scalar-metric"),
            param({"phases": []}, id="phases-not-a-mapping"),
            param({"phases": {"p": "nope"}}, id="phase-entry-not-a-mapping"),
        ],
    )  # fmt: skip
    def test_malformed_status_does_not_raise(self, status: dict[str, Any]) -> None:
        assert diagnose_benchmark(status, elapsed_seconds=120) is not None

    def test_healthy_run_produces_no_findings(self) -> None:
        findings = diagnose_benchmark(
            _status(
                requests=1000,
                errors=0,
                latency_avg=50,
                latency_p99=60,
                throughput=9.0,
                completed=1000,
            ),
            elapsed_seconds=600,
        )
        assert findings == []
