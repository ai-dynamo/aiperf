# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.exceptions import NoMetricValue
from aiperf.metrics.metric_dicts import MetricResultsDict
from aiperf.metrics.types.completed_request_count_metric import (
    CompletedRequestCountMetric,
)
from aiperf.metrics.types.error_request_count import ErrorRequestCountMetric
from aiperf.metrics.types.request_count_metric import RequestCountMetric


class TestCompletedRequestCountMetric:
    def test_completed_count_sums_success_and_error(self):
        results = MetricResultsDict()
        results[RequestCountMetric.tag] = 100
        results[ErrorRequestCountMetric.tag] = 18
        value = CompletedRequestCountMetric().derive_value(results)
        assert value == 118

    def test_completed_count_none_error_value_treated_as_zero(self):
        """``.get(..., 0) or 0`` defends against an explicit None value."""
        results = MetricResultsDict()
        results[RequestCountMetric.tag] = 50
        results[ErrorRequestCountMetric.tag] = None  # type: ignore[assignment]
        value = CompletedRequestCountMetric().derive_value(results)
        assert value == 50

    def test_completed_count_zero_errors_explicit(self):
        results = MetricResultsDict()
        results[RequestCountMetric.tag] = 50
        results[ErrorRequestCountMetric.tag] = 0
        value = CompletedRequestCountMetric().derive_value(results)
        assert value == 50

    def test_completed_count_missing_request_count_raises(self):
        """RequestCountMetric is required — derive raises when absent."""
        results = MetricResultsDict()
        results[ErrorRequestCountMetric.tag] = 5
        with pytest.raises(NoMetricValue):
            CompletedRequestCountMetric().derive_value(results)

    def test_completed_count_required_metrics_declared(self):
        """Required-metric declaration drives MetricRegistry dependency order."""
        assert CompletedRequestCountMetric.required_metrics == frozenset(
            {RequestCountMetric.tag, ErrorRequestCountMetric.tag}
        )
