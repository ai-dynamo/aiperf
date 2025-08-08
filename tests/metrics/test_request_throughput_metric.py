# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pytest import approx

from aiperf.metrics.metric_dicts import MetricResultsDict
from aiperf.metrics.types.benchmark_duration_metric import BenchmarkDurationMetric
from aiperf.metrics.types.request_count_metric import RequestCountMetric
from aiperf.metrics.types.request_throughput_metric import RequestThroughputMetric


class TestRequestThroughputMetric:
    def test_derive_value_calculates_throughput(self):
        metric = RequestThroughputMetric()

        # Create metric results with required values
        metric_results = MetricResultsDict()
        metric_results[RequestCountMetric.tag] = 100  # requests
        metric_results[BenchmarkDurationMetric.tag] = (
            5_000_000_000  # 5 seconds in nanoseconds
        )

        result = metric.derive_value(metric_results)
        expected = 100 / 5.0  # 20 requests per second
        assert result == approx(expected)

    def test_derive_value_with_zero_duration_raises(self):
        metric = RequestThroughputMetric()

        metric_results = MetricResultsDict()
        metric_results[RequestCountMetric.tag] = 100
        metric_results[BenchmarkDurationMetric.tag] = 0

        with pytest.raises(
            ValueError,
            match="Benchmark duration is required and must be greater than 0",
        ):
            metric.derive_value(metric_results)

    def test_derive_value_with_none_duration_raises(self):
        metric = RequestThroughputMetric()

        metric_results = MetricResultsDict()
        metric_results[RequestCountMetric.tag] = 100
        metric_results[BenchmarkDurationMetric.tag] = None

        with pytest.raises(
            ValueError,
            match="Benchmark duration is required and must be greater than 0",
        ):
            metric.derive_value(metric_results)
