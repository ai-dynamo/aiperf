# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pytest import approx

from aiperf.common.exceptions import NoMetricValue
from aiperf.metrics.metric_dicts import MetricResultsDict
from aiperf.metrics.types.cancelled_request_count_metric import (
    CancelledRequestCountMetric,
)
from aiperf.metrics.types.error_request_count import ErrorRequestCountMetric
from aiperf.metrics.types.request_count_metric import RequestCountMetric
from aiperf.metrics.types.request_error_rate_metric import RequestErrorRateMetric


class TestRequestErrorRateMetric:
    def test_mixed_success_and_error_computes_rate(self):
        metric = RequestErrorRateMetric()
        results = MetricResultsDict()
        results[RequestCountMetric.tag] = 90
        results[ErrorRequestCountMetric.tag] = 10
        # 10 errors / 100 completed == 10%
        assert metric.derive_value(results) == approx(10.0)

    def test_100pct_fail_only_error_count_present_returns_100(self):
        # When every request fails, request_count (successes) is absent from
        # metric_results and only error_request_count is present. The rate must
        # be 100.0 and MUST NOT be omitted -- this is the case a caller most
        # needs (a fully-down server).
        metric = RequestErrorRateMetric()
        results = MetricResultsDict()
        results[ErrorRequestCountMetric.tag] = 42
        assert metric.derive_value(results) == approx(100.0)

    def test_clean_run_only_success_count_present_returns_zero(self):
        # error_request_count is ERROR_ONLY, so it is absent on a zero-error
        # run. The rate must be 0.0, not omitted.
        metric = RequestErrorRateMetric()
        results = MetricResultsDict()
        results[RequestCountMetric.tag] = 100
        assert metric.derive_value(results) == approx(0.0)

    def test_truly_empty_run_raises_no_metric_value(self):
        metric = RequestErrorRateMetric()
        results = MetricResultsDict()
        with pytest.raises(NoMetricValue):
            metric.derive_value(results)

    def test_no_required_metrics_so_metric_never_vanishes(self):
        # Deliberately None: both counters can be legitimately absent, so
        # _check_metrics must not gate this metric out before it runs.
        assert RequestErrorRateMetric.required_metrics is None

    def test_cancellations_excluded_from_numerator_and_denominator(self):
        # A 40%-cancel/0-error run: 48 successes, 0 errors, 32 cancellations.
        # The record-processor never adds cancellations to error_request_count
        # or request_count, so this metric sees no cancellations at all and must
        # report 0%, agreeing with the credit-side `cancelled` bucket. Even if a
        # cancelled_request_count entry is present in the results dict, it must
        # not leak into the rate.
        metric = RequestErrorRateMetric()
        results = MetricResultsDict()
        results[RequestCountMetric.tag] = 48
        results[CancelledRequestCountMetric.tag] = 32
        assert metric.derive_value(results) == approx(0.0)

    def test_real_errors_present_with_cancellations_only_errors_count(self):
        # M=6 real errors, N=32 cancellations, 42 successes: rate is
        # 100 * 6 / (42 + 6) == 12.5% -- the 32 cancellations are absent from
        # both the numerator (error_request_count) and the denominator.
        metric = RequestErrorRateMetric()
        results = MetricResultsDict()
        results[RequestCountMetric.tag] = 42
        results[ErrorRequestCountMetric.tag] = 6
        results[CancelledRequestCountMetric.tag] = 32
        assert metric.derive_value(results) == approx(100.0 * 6 / 48)

    def test_registered_in_metric_registry(self):
        from aiperf.metrics.metric_registry import MetricRegistry

        cls = MetricRegistry.get_class("request_error_rate")
        assert cls is RequestErrorRateMetric
