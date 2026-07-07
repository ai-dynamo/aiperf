# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.metrics.metric_dicts import MetricResultsDict
from aiperf.metrics.types.completed_request_count_metric import (
    CompletedRequestCountMetric,
)
from aiperf.metrics.types.error_request_count import ErrorRequestCountMetric
from aiperf.metrics.types.request_count_metric import RequestCountMetric


class TestCompletedRequestCountMetric:
    def test_mixed_success_and_error_sums_both(self):
        metric = CompletedRequestCountMetric()
        results = MetricResultsDict()
        results[RequestCountMetric.tag] = 90
        results[ErrorRequestCountMetric.tag] = 10
        assert metric.derive_value(results) == 100

    def test_100pct_fail_only_error_count_present_returns_error_count(self):
        # request_count (successes) is absent when every request fails; the
        # completion total must equal the error count, not be omitted.
        metric = CompletedRequestCountMetric()
        results = MetricResultsDict()
        results[ErrorRequestCountMetric.tag] = 42
        assert metric.derive_value(results) == 42

    def test_clean_run_only_success_count_present_returns_success_count(self):
        # error_request_count is ERROR_ONLY and absent on a zero-error run;
        # the completion total must equal the success count, not be omitted.
        metric = CompletedRequestCountMetric()
        results = MetricResultsDict()
        results[RequestCountMetric.tag] = 100
        assert metric.derive_value(results) == 100

    def test_no_required_metrics_so_metric_never_vanishes(self):
        assert CompletedRequestCountMetric.required_metrics is None

    def test_registered_in_metric_registry(self):
        from aiperf.metrics.metric_registry import MetricRegistry

        cls = MetricRegistry.get_class("completed_request_count")
        assert cls is CompletedRequestCountMetric
