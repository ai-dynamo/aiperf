# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for SystemController exit status decisions."""

from aiperf.common.models import ErrorDetails
from aiperf.common.models.error_models import ErrorDetailsCount
from aiperf.common.models.metric_result_models import MetricResult, ProfileResults
from aiperf.controller.system_controller import profile_results_have_successes


def _results(records: list[MetricResult]) -> ProfileResults:
    return ProfileResults(completed=1, start_ns=0, end_ns=1, records=records)


def test_profile_results_have_successes_false_when_only_error_metrics() -> None:
    results = _results(
        [
            MetricResult(
                tag="error_request_count",
                header="Error Request Count",
                unit="requests",
                avg=1.0,
            ),
            MetricResult(tag="error_isl", header="Error ISL", unit="tokens", avg=550.0),
        ]
    )
    results.error_summary = [
        ErrorDetailsCount(
            error_details=ErrorDetails(type="ClientConnectorError", message="boom"),
            count=1,
        )
    ]

    assert not profile_results_have_successes(results)


def test_profile_results_have_successes_true_with_request_count_metric() -> None:
    results = _results(
        [
            MetricResult(
                tag="request_count",
                header="Request Count",
                unit="requests",
                avg=1.0,
            )
        ]
    )

    assert profile_results_have_successes(results)
