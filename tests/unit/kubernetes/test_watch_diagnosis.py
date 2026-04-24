# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the watch diagnosis engine."""

from datetime import datetime, timezone

import pytest

from aiperf.kubernetes.watch_diagnosis import diagnose
from aiperf.kubernetes.watch_models import (
    DiagnosisResult,
    MetricsSnapshot,
    PodSnapshot,
    WatchSnapshot,
    WorkersSnapshot,
)


def _snap(**kwargs: object) -> WatchSnapshot:
    defaults: dict[str, object] = {
        "timestamp": datetime.now(timezone.utc),
        "job_id": "test",
        "namespace": "ns",
        "phase": "Running",
    }
    defaults.update(kwargs)
    return WatchSnapshot(**defaults)


class TestDiagnoseHealth:
    def test_healthy_running(self) -> None:
        result = diagnose(_snap(workers=WorkersSnapshot(ready=1, total=1)))
        assert result.health == "healthy"

    def test_completed(self) -> None:
        result = diagnose(_snap(phase="Completed"))
        assert result.health == "completed"

    def test_failed(self) -> None:
        result = diagnose(_snap(phase="Failed", error="timeout"))
        assert result.health == "failed"

    def test_degraded_from_oom(self) -> None:
        pod = PodSnapshot(
            name="workers-0",
            role="worker",
            status="Running",
            restarts=1,
            oom_killed=True,
        )
        result = diagnose(_snap(pods=[pod]))
        assert result.health == "degraded"

    def test_stalled_from_pending(self) -> None:
        result = diagnose(_snap(phase="Pending", elapsed_seconds=120))
        assert result.health == "stalled"

    def test_failing_from_crash_loop(self) -> None:
        pod = PodSnapshot(
            name="workers-0",
            role="worker",
            status="Running",
            restarts=5,
            oom_killed=False,
        )
        result = diagnose(_snap(pods=[pod]))
        assert result.health == "failing"


class TestDiagnoseOOM:
    def test_oom_detected(self) -> None:
        pod = PodSnapshot(
            name="workers-0",
            role="worker",
            status="Running",
            restarts=1,
            oom_killed=True,
        )
        result = diagnose(_snap(pods=[pod]))
        assert result.health == "degraded"
        assert any(i.id == "oom_restart" for i in result.issues)

    def test_no_oom(self) -> None:
        pod = PodSnapshot(
            name="workers-0",
            role="worker",
            status="Running",
            restarts=0,
            oom_killed=False,
        )
        result = diagnose(_snap(pods=[pod]))
        assert not any(i.id == "oom_restart" for i in result.issues)

    def test_oom_issue_details(self) -> None:
        pod = PodSnapshot(
            name="workers-0",
            role="worker",
            status="Running",
            restarts=2,
            oom_killed=True,
        )
        result = diagnose(_snap(pods=[pod]))
        oom_issues = [i for i in result.issues if i.id == "oom_restart"]
        assert len(oom_issues) == 1
        assert oom_issues[0].severity == "warning"
        assert "workers-0" in oom_issues[0].detail


class TestDiagnoseCrashLoop:
    def test_crash_loop_detected(self) -> None:
        pod = PodSnapshot(
            name="workers-0",
            role="worker",
            status="Running",
            restarts=4,
            oom_killed=False,
        )
        result = diagnose(_snap(pods=[pod]))
        assert any(i.id == "crash_loop" for i in result.issues)

    def test_no_crash_loop_under_threshold(self) -> None:
        pod = PodSnapshot(
            name="workers-0",
            role="worker",
            status="Running",
            restarts=3,
            oom_killed=False,
        )
        result = diagnose(_snap(pods=[pod]))
        assert not any(i.id == "crash_loop" for i in result.issues)

    def test_crash_loop_issue_details(self) -> None:
        pod = PodSnapshot(
            name="ctrl-0",
            role="controller",
            status="CrashLoopBackOff",
            restarts=10,
            oom_killed=False,
        )
        result = diagnose(_snap(pods=[pod]))
        crash_issues = [i for i in result.issues if i.id == "crash_loop"]
        assert len(crash_issues) == 1
        assert crash_issues[0].severity == "critical"
        assert "ctrl-0" in crash_issues[0].detail


class TestDiagnoseStalled:
    def test_stalled_pending(self) -> None:
        result = diagnose(_snap(phase="Pending", elapsed_seconds=120))
        assert result.stalled is True
        assert any(i.id == "stalled_pending" for i in result.issues)

    def test_not_stalled_if_short(self) -> None:
        result = diagnose(_snap(phase="Pending", elapsed_seconds=10))
        assert result.stalled is False

    def test_stalled_running(self) -> None:
        result = diagnose(_snap(phase="Running", elapsed_seconds=60))
        assert result.stalled is True
        assert any(i.id == "stalled_running" for i in result.issues)

    def test_not_stalled_running_if_short(self) -> None:
        result = diagnose(_snap(phase="Running", elapsed_seconds=10))
        assert result.stalled is False
        assert not any(i.id == "stalled_running" for i in result.issues)

    def test_not_stalled_running_with_throughput(self) -> None:
        result = diagnose(
            _snap(
                phase="Running",
                elapsed_seconds=120,
                metrics=MetricsSnapshot(
                    request_throughput_rps=1500.0, request_count=50000
                ),
            )
        )
        assert result.stalled is False
        assert not any(i.id == "stalled_running" for i in result.issues)

    def test_not_stalled_running_with_progress(self) -> None:
        from aiperf.kubernetes.watch_models import ProgressSnapshot

        result = diagnose(
            _snap(
                phase="Running",
                elapsed_seconds=120,
                progress=ProgressSnapshot(requests_completed=1000),
            )
        )
        assert result.stalled is False
        assert not any(i.id == "stalled_running" for i in result.issues)

    def test_stall_reason_populated(self) -> None:
        result = diagnose(_snap(phase="Pending", elapsed_seconds=90))
        assert result.stall_reason is not None
        assert "Pending" in result.stall_reason


class TestDiagnoseConditions:
    def test_endpoint_unreachable(self) -> None:
        result = diagnose(
            _snap(
                conditions={"endpoint_reachable": False, "config_valid": True},
            )
        )
        assert any(i.id == "endpoint_unreachable" for i in result.issues)

    def test_endpoint_reachable_no_issue(self) -> None:
        result = diagnose(
            _snap(
                conditions={"endpoint_reachable": True},
            )
        )
        assert not any(i.id == "endpoint_unreachable" for i in result.issues)

    def test_preflight_failed(self) -> None:
        result = diagnose(
            _snap(
                conditions={"preflight_passed": False},
            )
        )
        assert any(i.id == "preflight_failed" for i in result.issues)

    def test_preflight_passed_no_issue(self) -> None:
        result = diagnose(
            _snap(
                conditions={"preflight_passed": True},
            )
        )
        assert not any(i.id == "preflight_failed" for i in result.issues)

    def test_results_fetch_failed_on_completed(self) -> None:
        result = diagnose(
            _snap(
                phase="Completed",
                conditions={"results_available": False},
            )
        )
        assert any(i.id == "results_fetch_failed" for i in result.issues)

    def test_results_fetch_not_flagged_when_running(self) -> None:
        result = diagnose(
            _snap(
                phase="Running",
                conditions={"results_available": False},
            )
        )
        assert not any(i.id == "results_fetch_failed" for i in result.issues)


class TestDiagnoseErrorRate:
    def test_high_error_rate(self) -> None:
        result = diagnose(
            _snap(
                metrics=MetricsSnapshot(request_count=100, error_count=10),
            )
        )
        assert result.error_rate == pytest.approx(0.1)
        assert any(i.id == "high_error_rate" for i in result.issues)

    def test_low_error_rate(self) -> None:
        result = diagnose(
            _snap(
                metrics=MetricsSnapshot(request_count=1000, error_count=1),
            )
        )
        assert result.error_rate == pytest.approx(0.001)
        assert not any(i.id == "high_error_rate" for i in result.issues)

    def test_zero_requests_no_error_rate(self) -> None:
        result = diagnose(
            _snap(
                metrics=MetricsSnapshot(request_count=0, error_count=0),
            )
        )
        assert result.error_rate == 0.0
        assert not any(i.id == "high_error_rate" for i in result.issues)

    def test_error_rate_computed_without_metrics(self) -> None:
        result = diagnose(_snap())
        assert result.error_rate == 0.0


class TestDiagnoseHighLatency:
    def test_high_latency_detected(self) -> None:
        result = diagnose(
            _snap(
                metrics=MetricsSnapshot(
                    request_latency_avg_ms=100.0,
                    request_latency_p99_ms=1500.0,
                ),
            )
        )
        assert any(i.id == "high_latency" for i in result.issues)

    def test_normal_latency_no_issue(self) -> None:
        result = diagnose(
            _snap(
                metrics=MetricsSnapshot(
                    request_latency_avg_ms=100.0,
                    request_latency_p99_ms=200.0,
                ),
            )
        )
        assert not any(i.id == "high_latency" for i in result.issues)

    def test_zero_latency_no_issue(self) -> None:
        result = diagnose(
            _snap(
                metrics=MetricsSnapshot(
                    request_latency_avg_ms=0.0,
                    request_latency_p99_ms=0.0,
                ),
            )
        )
        assert not any(i.id == "high_latency" for i in result.issues)


class TestDiagnoseMultipleIssues:
    def test_multiple_issues_combined(self) -> None:
        pod = PodSnapshot(
            name="workers-0",
            role="worker",
            status="Running",
            restarts=5,
            oom_killed=True,
        )
        result = diagnose(
            _snap(
                pods=[pod],
                metrics=MetricsSnapshot(request_count=100, error_count=20),
                conditions={"endpoint_reachable": False},
            )
        )
        issue_ids = {i.id for i in result.issues}
        assert "oom_restart" in issue_ids
        assert "crash_loop" in issue_ids
        assert "high_error_rate" in issue_ids
        assert "endpoint_unreachable" in issue_ids

    def test_returns_diagnosis_result_type(self) -> None:
        result = diagnose(_snap())
        assert isinstance(result, DiagnosisResult)
