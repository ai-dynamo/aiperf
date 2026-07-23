# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit coverage for ``_sum_runtime_response_counts``."""

from __future__ import annotations

from aiperf.cli_runner._aggregate import _sum_runtime_response_counts
from aiperf.common.models.export_models import JsonMetricResult
from aiperf.orchestrator.models import RunResult


def _metric(avg: float | None) -> JsonMetricResult:
    return JsonMetricResult(unit="count", avg=avg)


def _run(
    *,
    success: bool,
    request_count: float | None = None,
    error_request_count: float | None = None,
    context_overflow_count: float | None = None,
    skipped_context_overflow_count: float | None = None,
    label: str = "run",
) -> RunResult:
    summary: dict[str, JsonMetricResult] = {}
    if request_count is not None:
        summary["request_count"] = _metric(request_count)
    if error_request_count is not None:
        summary["error_request_count"] = _metric(error_request_count)
    if context_overflow_count is not None:
        summary["context_overflow_count"] = _metric(context_overflow_count)
    if skipped_context_overflow_count is not None:
        summary["skipped_context_overflow_count"] = _metric(
            skipped_context_overflow_count
        )
    return RunResult(label=label, success=success, summary_metrics=summary)


def test_sum_does_not_double_count_metric_path_overflow() -> None:
    """Metric-path overflows already live in error_request_count.

    Adding ``context_overflow_count`` again would inflate the denominator and
    can incorrectly keep ``submission_valid=True`` near the 1% boundary.
    """
    run = _run(
        success=True,
        request_count=100,
        error_request_count=5,
        context_overflow_count=3,
    )

    total_responses, overflow = _sum_runtime_response_counts([run])

    assert total_responses == 105
    assert overflow == 3


def test_sum_adds_skip_path_overflow_side_channel() -> None:
    """AGENTIC_REPLAY skip-path overflows are not in error_request_count."""
    run = _run(
        success=True,
        request_count=100,
        error_request_count=5,
        context_overflow_count=3,
        skipped_context_overflow_count=3,
    )

    total_responses, overflow = _sum_runtime_response_counts([run])

    assert total_responses == 108
    assert overflow == 3


def test_sum_across_multiple_successful_runs() -> None:
    """Counts accumulate across every successful run."""
    runs = [
        _run(success=True, request_count=10, error_request_count=1, label="a"),
        _run(
            success=True,
            request_count=20,
            error_request_count=2,
            context_overflow_count=4,
            skipped_context_overflow_count=4,
            label="b",
        ),
    ]

    total_responses, overflow = _sum_runtime_response_counts(runs)

    # (10+1+0) + (20+2+4) = 37 ; overflow = 0 + 4
    assert total_responses == 37
    assert overflow == 4


def test_failed_runs_are_skipped() -> None:
    """A non-successful run contributes nothing to either total."""
    runs = [
        _run(success=True, request_count=50, error_request_count=0, label="ok"),
        _run(
            success=False,
            request_count=999,
            error_request_count=999,
            context_overflow_count=999,
            skipped_context_overflow_count=999,
            label="failed",
        ),
    ]

    total_responses, overflow = _sum_runtime_response_counts(runs)

    assert total_responses == 50
    assert overflow == 0


def test_empty_results_returns_zero() -> None:
    """No runs -> (0, 0), not a crash on the empty reduction."""
    assert _sum_runtime_response_counts([]) == (0, 0)


def test_missing_metrics_contribute_zero() -> None:
    """Absent metric tags and avg=None both coerce to 0 gracefully."""
    runs = [
        # success run with no summary metrics at all
        _run(success=True, label="bare"),
        # success run whose metrics exist but carry avg=None
        RunResult(
            label="none-avg",
            success=True,
            summary_metrics={
                "request_count": JsonMetricResult(unit="count", avg=None),
                "context_overflow_count": JsonMetricResult(unit="count", avg=None),
            },
        ),
    ]

    total_responses, overflow = _sum_runtime_response_counts(runs)

    assert total_responses == 0
    assert overflow == 0


def test_avg_truncated_to_int() -> None:
    """Averaged floats are truncated via ``int(...)`` per the source."""
    run = _run(
        success=True,
        request_count=10.9,
        error_request_count=0.0,
        context_overflow_count=2.7,
        skipped_context_overflow_count=2.7,
    )

    total_responses, overflow = _sum_runtime_response_counts([run])

    # int(10.9) + int(0.0) + int(2.7) = 10 + 0 + 2 = 12 ; overflow = int(2.7) = 2
    assert total_responses == 12
    assert overflow == 2
