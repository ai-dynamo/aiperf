# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the benchmark-metric diagnosis detectors.

The metric payloads here mirror what a real run actually publishes to
``status.liveMetrics``. That matters for the error-rate detector: an earlier
version of these tests constructed an ``error_count`` tag, which no metric
declares, so every assertion exercised arithmetic over a key that can never
occur in production while the detector itself was dead.
"""

from typing import Any

import pytest
from pytest import param

from aiperf.kubernetes.benchmark_diagnosis import (
    BenchmarkFinding,
    diagnose_benchmark,
    error_rate,
    error_rate_from_metrics,
)
from aiperf.kubernetes.environment import K8sEnvironment


def _live_metrics(
    successes: int | None,
    errors: int | None,
    *,
    error_percent: float | None = None,
) -> dict[str, Any]:
    """The count/error tags a real run publishes for ``successes``/``errors``.

    Reproduces the producer contract exactly:

    - ``request_count`` counts valid requests only, and the accumulator omits
      a tag with no rows -- so an all-error run publishes none.
    - ``error_request_count`` is ``MetricFlags.ERROR_ONLY`` and is stripped by
      ``filter_display_metrics``, so it never appears here.
    - ``completed_request_count`` is derived and requires ``request_count``,
      so it too is absent on an all-error run.
    - ``request_error_rate`` is a PERCENT and is the only error signal that
      survives on every path.

    ``error_percent`` overrides the derived percent, for payloads that are not
    reachable from a plain success/error pair.
    """
    metrics: dict[str, Any] = {}
    successes = successes or 0
    errors = errors or 0
    total = successes + errors
    if successes > 0:
        metrics["request_count"] = {"avg": float(successes), "unit": "requests"}
        metrics["completed_request_count"] = {"avg": float(total), "unit": "requests"}
    if error_percent is not None:
        metrics["request_error_rate"] = {"avg": error_percent, "unit": "%"}
    elif total > 0:
        metrics["request_error_rate"] = {"avg": 100.0 * errors / total, "unit": "%"}
    return metrics


def _status(
    *,
    phase: str = "Running",
    successes: int | None = None,
    errors: int | None = None,
    error_percent: float | None = None,
    latency_avg: float | None = None,
    latency_p99: float | None = None,
    throughput: float | None = None,
    completed: int | None = None,
) -> dict[str, Any]:
    """Build an AIPerfJob .status with only the requested metrics present."""
    metrics = _live_metrics(successes, errors, error_percent=error_percent)
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
        "successes,errors,expected",
        [
            param(800, 200, 0.2, id="ordinary"),
            param(1000, 0, 0.0, id="no-errors"),
            param(0, 0, 0.0, id="zero-requests-is-not-divide-by-zero"),
            param(240, 60, 0.2, id="partial-failure"),
            param(999, 1, 0.001, id="one-in-a-thousand"),
        ],
    )  # fmt: skip
    def test_error_rate_values(
        self, successes: int, errors: int, expected: float
    ) -> None:
        assert error_rate(_status(successes=successes, errors=errors)) == pytest.approx(
            expected
        )

    def test_error_rate_returns_a_fraction_not_a_percent(self) -> None:
        """``request_error_rate`` is a PERCENT; the thresholds are fractions.

        Without the ``/100`` conversion a single failure in a thousand reads as
        0.1 and clears the 0.05 threshold, turning a silently dead detector
        into one that fires on nearly every real run.
        """
        status = _status(successes=999, errors=1)
        published = status["liveMetrics"]["metrics"]["request_error_rate"]["avg"]

        assert published == pytest.approx(0.1), "producer emits percentage points"
        assert error_rate(status) == pytest.approx(0.001)
        assert error_rate(status) <= K8sEnvironment.DIAGNOSIS.HIGH_ERROR_RATE_THRESHOLD

    @pytest.mark.parametrize(
        "error_percent,expected",
        [
            param(150.0, 1.0, id="above-100-percent-clamps-to-one"),
            param(-20.0, 0.0, id="negative-percent-clamps-to-zero"),
        ],
    )  # fmt: skip
    def test_out_of_range_percent_is_clamped(
        self, error_percent: float, expected: float
    ) -> None:
        """The clamp survives, but its justification changed.

        It previously guarded ``error_count > request_count``, which was
        reachable because two counters were averaged independently over
        staggered liveMetrics windows. The rate now comes from a single bounded
        ``request_error_rate`` sample, so counter skew can no longer produce an
        out-of-range value -- only a malformed payload can. The clamp is kept
        because ``AIPerfJobInfo.error_rate`` and the CR status both declare a
        0..1 bound that a nonsense percent must not be allowed to violate.
        """
        status = _status(successes=100, error_percent=error_percent)
        assert error_rate(status) == expected

    def test_error_rate_missing_metrics_returns_zero(self) -> None:
        assert error_rate({}) == 0.0

    def test_unknown_is_distinguishable_from_zero(self) -> None:
        """``error_rate`` collapses unknown to 0.0 for its callers; the
        underlying reading keeps them apart so the completion gate can."""
        assert error_rate_from_metrics({}) == (None, 0, 0)
        assert error_rate_from_metrics(_live_metrics(1000, 0)) == (0.0, 0, 1000)

    def test_all_error_run_reports_rate_without_any_counter(self) -> None:
        """An all-error run publishes no count at all -- ``request_count`` has
        no valid rows and both error counters are ERROR_ONLY -- so the percent
        is the only surviving signal and must still yield a verdict."""
        metrics = _live_metrics(0, 300)

        assert set(metrics) == {"request_error_rate"}
        assert error_rate_from_metrics(metrics) == (1.0, 0, 0)


class TestHighErrorRate:
    """The high_error_rate detector."""

    def test_above_threshold_reports_finding(self) -> None:
        findings = diagnose_benchmark(_status(successes=800, errors=200, throughput=5))
        assert "high_error_rate" in _ids(findings)

    def test_at_threshold_is_silent(self) -> None:
        # default threshold is 0.05; exactly 5% must not trip it
        findings = diagnose_benchmark(_status(successes=950, errors=50, throughput=5))
        assert "high_error_rate" not in _ids(findings)

    def test_one_error_in_a_thousand_is_silent(self) -> None:
        """Regression guard for the percent-vs-fraction trap."""
        findings = diagnose_benchmark(_status(successes=999, errors=1, throughput=5))
        assert "high_error_rate" not in _ids(findings)

    def test_partial_failure_reports_finding(self) -> None:
        """A 240-ok / 60-failed run: the case the detector never caught."""
        findings = diagnose_benchmark(_status(successes=240, errors=60, throughput=5))
        detail = next(f for f in findings if f.id == "high_error_rate").detail
        assert "20.0%" in detail
        assert "(60/300)" in detail

    def test_detail_reports_the_counts(self) -> None:
        findings = diagnose_benchmark(_status(successes=800, errors=200, throughput=5))
        detail = next(f for f in findings if f.id == "high_error_rate").detail
        assert "20.0%" in detail
        assert "200/1000" in detail

    def test_detail_omits_counts_when_none_were_published(self) -> None:
        """An all-error run has no denominator, so ``(0/0)`` would contradict
        the rate it annotates."""
        findings = diagnose_benchmark(_status(successes=0, errors=300, throughput=5))
        detail = next(f for f in findings if f.id == "high_error_rate").detail
        assert detail == "Error rate: 100.0%"


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
            param({"liveMetrics": {"metrics": {"request_error_rate": "n/a"}}}, id="non-numeric-rate"),
            param({"liveMetrics": {"metrics": {"request_error_rate": {"avg": float("nan")}}}}, id="nan-rate"),
            param({"phases": []}, id="phases-not-a-mapping"),
            param({"phases": {"p": "nope"}}, id="phase-entry-not-a-mapping"),
        ],
    )  # fmt: skip
    def test_malformed_status_does_not_raise(self, status: dict[str, Any]) -> None:
        assert diagnose_benchmark(status, elapsed_seconds=120) is not None

    def test_healthy_run_produces_no_findings(self) -> None:
        findings = diagnose_benchmark(
            _status(
                successes=1000,
                errors=0,
                latency_avg=50,
                latency_p99=60,
                throughput=9.0,
                completed=1000,
            ),
            elapsed_seconds=600,
        )
        assert findings == []
