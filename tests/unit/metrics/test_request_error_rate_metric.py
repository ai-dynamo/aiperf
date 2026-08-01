# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pytest import approx

from aiperf.common.exceptions import NoMetricValue
from aiperf.metrics.metric_dicts import MetricResultsDict
from aiperf.metrics.types.error_request_count import ErrorRequestCountMetric
from aiperf.metrics.types.request_count_metric import RequestCountMetric
from aiperf.metrics.types.request_error_rate_metric import RequestErrorRateMetric


class TestRequestErrorRateMetric:
    def test_error_rate_basic(self):
        results = MetricResultsDict()
        results[RequestCountMetric.tag] = 82
        results[ErrorRequestCountMetric.tag] = 18
        value = RequestErrorRateMetric().derive_value(results)
        assert value == approx(18.0)

    def test_error_rate_zero_errors(self):
        results = MetricResultsDict()
        results[RequestCountMetric.tag] = 100
        results[ErrorRequestCountMetric.tag] = 0
        value = RequestErrorRateMetric().derive_value(results)
        assert value == approx(0.0)

    def test_error_rate_missing_error_count_treated_as_zero(self):
        results = MetricResultsDict()
        results[RequestCountMetric.tag] = 100
        value = RequestErrorRateMetric().derive_value(results)
        assert value == approx(0.0)

    def test_error_rate_none_error_value_treated_as_zero(self):
        """``.get(..., 0) or 0`` defends against an explicit None value."""
        results = MetricResultsDict()
        results[RequestCountMetric.tag] = 100
        results[ErrorRequestCountMetric.tag] = None  # type: ignore[assignment]
        value = RequestErrorRateMetric().derive_value(results)
        assert value == approx(0.0)

    def test_error_rate_all_errors(self):
        results = MetricResultsDict()
        results[RequestCountMetric.tag] = 0
        results[ErrorRequestCountMetric.tag] = 10
        # successes=0 + errors=10 = 10 total -> 100%
        value = RequestErrorRateMetric().derive_value(results)
        assert value == approx(100.0)

    def test_error_rate_no_completed_requests_raises(self):
        results = MetricResultsDict()
        results[RequestCountMetric.tag] = 0
        results[ErrorRequestCountMetric.tag] = 0
        with pytest.raises(NoMetricValue, match="No completed requests"):
            RequestErrorRateMetric().derive_value(results)

    def test_error_rate_missing_request_count_raises(self):
        results = MetricResultsDict()
        results[ErrorRequestCountMetric.tag] = 5
        with pytest.raises(NoMetricValue):
            RequestErrorRateMetric().derive_value(results)

    def test_error_rate_required_metrics_declared(self):
        assert RequestErrorRateMetric.required_metrics == frozenset(
            {RequestCountMetric.tag}
        )
