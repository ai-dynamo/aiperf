# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A finished benchmark must not report success when every request errored.

Result files are not a success signal: a run in which all requests failed still
writes profile_export_aiperf.json, so keying success on file presence reported
phase=Completed for a benchmark that measured nothing.
"""

from __future__ import annotations

import pytest

from aiperf.kubernetes.crd_models import ControllerFetchResult
from aiperf.kubernetes.environment import K8sEnvironment
from aiperf.operator.handlers.completion import (
    _compute_result_flags,
    _result_error_rate,
)

KEY_FILE = "profile_export_aiperf.json"


def _result(*, requests: float | None, errors: float | None) -> ControllerFetchResult:
    """A successful fetch whose payload carries the given request/error counts."""
    metrics: dict[str, object] = {}
    if requests is not None:
        metrics["request_count"] = {"avg": requests}
    if errors is not None:
        metrics["error_count"] = {"avg": errors}
    return ControllerFetchResult(
        downloaded=[KEY_FILE],
        metrics={"metrics": metrics} if metrics else {},
        error="",
    )


class TestResultErrorRate:
    def test_all_requests_errored_is_rate_one(self) -> None:
        assert _result_error_rate(_result(requests=300, errors=300)) == (1.0, 300, 300)

    def test_partial_errors(self) -> None:
        rate, errors, requests = _result_error_rate(_result(requests=100, errors=25))
        assert (round(rate, 4), errors, requests) == (0.25, 25, 100)

    def test_rate_is_clamped_when_errors_exceed_requests(self) -> None:
        rate, _, _ = _result_error_rate(_result(requests=10, errors=99))
        assert rate == 1.0

    def test_absent_metrics_report_unknown_not_healthy(self) -> None:
        """A missing payload must never read as 'no errors'."""
        assert _result_error_rate(_result(requests=None, errors=None)) == (0.0, 0, 0)

    def test_zero_requests_reports_unknown(self) -> None:
        assert _result_error_rate(_result(requests=0, errors=0)) == (0.0, 0, 0)

    def test_infinite_error_count_reports_unknown(self) -> None:
        """A non-finite final error count cannot establish a completion verdict."""
        assert _result_error_rate(_result(requests=100, errors=float("inf"))) == (
            0.0,
            0,
            0,
        )


class TestSuccessGate:
    def test_total_failure_is_not_success(self) -> None:
        flags = _compute_result_flags(_result(requests=300, errors=300), "job-1")

        assert flags.has_files is True
        assert flags.has_error is False, "the fetch itself succeeded"
        assert flags.success is False
        assert flags.benchmark_failure is not None
        assert "300/300" in flags.benchmark_failure

    def test_clean_run_still_succeeds(self) -> None:
        flags = _compute_result_flags(_result(requests=300, errors=0), "job-1")

        assert flags.success is True
        assert flags.benchmark_failure is None

    def test_heavy_but_partial_errors_still_complete_by_default(self) -> None:
        """Default bar is 1.0, so tightening it stays a deliberate opt-in."""
        assert K8sEnvironment.DIAGNOSIS.FAIL_ABOVE_ERROR_RATE == 1.0
        flags = _compute_result_flags(_result(requests=100, errors=99), "job-1")

        assert flags.success is True
        assert flags.benchmark_failure is None

    def test_threshold_is_configurable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            K8sEnvironment.DIAGNOSIS, "FAIL_ABOVE_ERROR_RATE", 0.5, raising=False
        )
        flags = _compute_result_flags(_result(requests=100, errors=60), "job-1")

        assert flags.success is False
        assert flags.benchmark_failure is not None
        assert "60/100" in flags.benchmark_failure

    def test_missing_metrics_do_not_fail_an_otherwise_good_run(self) -> None:
        """Unknown must not be treated as failure either -- only as unknown."""
        flags = _compute_result_flags(_result(requests=None, errors=None), "job-1")

        assert flags.success is True
        assert flags.benchmark_failure is None

    def test_fetch_error_still_dominates(self) -> None:
        """A fetch failure is reported as such, not as a benchmark failure."""
        result = ControllerFetchResult(
            downloaded=[KEY_FILE],
            metrics={
                "metrics": {"request_count": {"avg": 300}, "error_count": {"avg": 300}}
            },
            error="partial fetch: checkpoints saved, exports missing",
        )
        flags = _compute_result_flags(result, "job-1")

        assert flags.success is False
        assert flags.has_error is True
        assert flags.benchmark_failure is None, (
            "the error-rate gate only runs when the fetch itself succeeded"
        )
