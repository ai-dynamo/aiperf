# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A finished benchmark must not report success when every request errored.

Result files are not a success signal: a run in which all requests failed still
writes profile_export_aiperf.json, so keying success on file presence reported
phase=Completed for a benchmark that measured nothing.

The payloads here are built by :func:`_live_metrics` / :func:`_export_metrics`,
which reproduce the two shapes ``ControllerFetchResult.metrics`` can actually
hold. An earlier version of this file constructed an ``error_count`` tag that no
metric declares, so all of it passed while the gate itself read zero on every
real run.
"""

from __future__ import annotations

from typing import Any

import pytest

from aiperf.kubernetes.crd_models import ControllerFetchResult
from aiperf.kubernetes.environment import K8sEnvironment
from aiperf.operator.handlers.completion import (
    _compute_result_flags,
    _result_error_rate,
)

KEY_FILE = "profile_export_aiperf.json"


def _live_metrics(successes: int, errors: int) -> dict[str, Any]:
    """The ``/api/metrics`` shape: filtered, so no error counter survives.

    ``ControllerFetchResult.metrics`` normally comes from ``/api/metrics``,
    which serves the same set ``filter_display_metrics`` published. That drops
    ``error_request_count`` (``MetricFlags.ERROR_ONLY``), leaving
    ``request_error_rate`` -- a PERCENT -- as the only error signal. An
    all-error run additionally has no ``request_count`` rows, so neither it nor
    the derived ``completed_request_count`` is emitted.
    """
    metrics: dict[str, Any] = {}
    total = successes + errors
    if successes > 0:
        metrics["request_count"] = {"avg": float(successes), "unit": "requests"}
        metrics["completed_request_count"] = {"avg": float(total), "unit": "requests"}
    if total > 0:
        metrics["request_error_rate"] = {"avg": 100.0 * errors / total, "unit": "%"}
    return metrics


def _export_metrics(successes: int, errors: int) -> dict[str, Any]:
    """The on-disk export shape: unfiltered, so the raw counters are present.

    Used when ``/api/metrics`` did not answer and completion falls back to
    parsing ``profile_export_aiperf.json``.
    """
    metrics: dict[str, Any] = {}
    total = successes + errors
    if successes > 0:
        metrics["request_count"] = {"avg": float(successes), "unit": "requests"}
        metrics["completed_request_count"] = {"avg": float(total), "unit": "requests"}
    if errors > 0:
        metrics["error_request_count"] = {"avg": float(errors), "unit": "requests"}
    if total > 0:
        metrics["request_error_rate"] = {"avg": 100.0 * errors / total, "unit": "%"}
    return metrics


def _result(metrics: dict[str, Any] | None) -> ControllerFetchResult:
    """A successful fetch carrying ``metrics`` as its per-tag payload."""
    return ControllerFetchResult(
        downloaded=[KEY_FILE],
        metrics={"metrics": metrics} if metrics else {},
        error="",
    )


def _live(successes: int, errors: int) -> ControllerFetchResult:
    return _result(_live_metrics(successes, errors))


def _export(successes: int, errors: int) -> ControllerFetchResult:
    return _result(_export_metrics(successes, errors))


class TestResultErrorRate:
    def test_all_requests_errored_is_rate_one_on_the_export_path(self) -> None:
        assert _result_error_rate(_export(0, 300)) == (1.0, 300, 300)

    def test_all_requests_errored_is_rate_one_on_the_live_path(self) -> None:
        """The filtered payload carries no counter, only the 100% rate. The
        verdict must still be reachable -- this is the run the gate exists for."""
        assert _result_error_rate(_live(0, 300)) == (1.0, 0, 0)

    @pytest.mark.parametrize("build", [_live, _export], ids=["live", "export"])
    def test_partial_errors(self, build: Any) -> None:
        """240 ok / 60 failed -- previously reported a confident ``(0.0, 0, 240)``.

        A positive denominator with a zero rate is indistinguishable from a
        genuinely clean run, which is why this case went undetected.
        """
        rate, errors, requests = _result_error_rate(build(240, 60))
        assert (round(rate, 4), requests) == (0.2, 300)
        assert errors == 60

    @pytest.mark.parametrize("build", [_live, _export], ids=["live", "export"])
    def test_rate_is_a_fraction_not_a_percent(self, build: Any) -> None:
        """One failure in a thousand is 0.001, not 0.1.

        The producer emits ``request_error_rate`` in percentage points while
        every ``AIPERF_K8S_DIAGNOSIS_*`` threshold is a fraction. Skipping the
        conversion would terminate any run with >=1% errors as ``Failed``.
        """
        rate, _, _ = _result_error_rate(build(999, 1))
        assert rate == pytest.approx(0.001)

    def test_absent_metrics_report_unknown_not_healthy(self) -> None:
        """A missing payload must never read as 'no errors'."""
        assert _result_error_rate(_result(None)) == (None, 0, 0)

    def test_zero_requests_reports_unknown(self) -> None:
        assert _result_error_rate(_result({})) == (None, 0, 0)

    def test_infinite_error_count_reports_unknown(self) -> None:
        """A non-finite final error count cannot establish a completion verdict."""
        metrics = {
            "request_count": {"avg": 100.0},
            "error_request_count": {"avg": float("inf")},
        }
        assert _result_error_rate(_result(metrics)) == (None, 0, 100)

    def test_non_finite_rate_falls_through_to_the_counters(self) -> None:
        """A NaN percent is unusable, but the raw counters beside it are not."""
        metrics = {
            "request_count": {"avg": 240.0},
            "error_request_count": {"avg": 60.0},
            "request_error_rate": {"avg": float("nan")},
        }
        assert _result_error_rate(_result(metrics)) == (0.2, 60, 300)


class TestSuccessGate:
    @pytest.mark.parametrize("build", [_live, _export], ids=["live", "export"])
    def test_total_failure_is_not_success(self, build: Any) -> None:
        flags = _compute_result_flags(build(0, 300), "job-1")

        assert flags.has_files is True
        assert flags.has_error is False, "the fetch itself succeeded"
        assert flags.success is False
        assert flags.benchmark_failure is not None
        assert "100.0%" in flags.benchmark_failure

    def test_total_failure_message_reports_counts_when_available(self) -> None:
        flags = _compute_result_flags(_export(0, 300), "job-1")
        assert flags.benchmark_failure is not None
        assert "300/300" in flags.benchmark_failure

    def test_total_failure_message_omits_unavailable_counts(self) -> None:
        """The live payload has no denominator; ``(0/0)`` would read as a
        contradiction of the 100% it annotates."""
        flags = _compute_result_flags(_live(0, 300), "job-1")
        assert flags.benchmark_failure == "Benchmark failed: 100.0% of requests errored"

    @pytest.mark.parametrize("build", [_live, _export], ids=["live", "export"])
    def test_clean_run_still_succeeds(self, build: Any) -> None:
        flags = _compute_result_flags(build(300, 0), "job-1")

        assert flags.success is True
        assert flags.benchmark_failure is None

    @pytest.mark.parametrize("build", [_live, _export], ids=["live", "export"])
    def test_heavy_but_partial_errors_still_complete_by_default(
        self, build: Any
    ) -> None:
        """Default bar is 1.0, so tightening it stays a deliberate opt-in."""
        assert K8sEnvironment.DIAGNOSIS.FAIL_ABOVE_ERROR_RATE == 1.0
        flags = _compute_result_flags(build(1, 99), "job-1")

        assert flags.success is True
        assert flags.benchmark_failure is None

    def test_one_percent_errors_does_not_fail_the_run(self) -> None:
        """Regression guard for the percent-vs-fraction trap: an unconverted
        ``request_error_rate`` of 1.0 would meet FAIL_ABOVE_ERROR_RATE=1.0 and
        flip a healthy run's terminal phase to Failed."""
        flags = _compute_result_flags(_live(99, 1), "job-1")

        assert flags.success is True
        assert flags.benchmark_failure is None

    def test_threshold_is_configurable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            K8sEnvironment.DIAGNOSIS, "FAIL_ABOVE_ERROR_RATE", 0.5, raising=False
        )
        flags = _compute_result_flags(_export(40, 60), "job-1")

        assert flags.success is False
        assert flags.benchmark_failure is not None
        assert "60/100" in flags.benchmark_failure

    def test_missing_metrics_do_not_fail_an_otherwise_good_run(self) -> None:
        """Unknown must not be treated as failure either -- only as unknown."""
        flags = _compute_result_flags(_result(None), "job-1")

        assert flags.success is True
        assert flags.benchmark_failure is None

    def test_fetch_error_still_dominates(self) -> None:
        """A fetch failure is reported as such, not as a benchmark failure."""
        result = ControllerFetchResult(
            downloaded=[KEY_FILE],
            metrics={"metrics": _export_metrics(0, 300)},
            error="partial fetch: checkpoints saved, exports missing",
        )
        flags = _compute_result_flags(result, "job-1")

        assert flags.success is False
        assert flags.has_error is True
        assert flags.benchmark_failure is None, (
            "the error-rate gate only runs when the fetch itself succeeded"
        )
