# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.metrics.metric_dicts import MetricResultsDict
from aiperf.metrics.types.benchmark_duration_metric import BenchmarkDurationMetric
from aiperf.metrics.types.benchmark_token_count import BenchmarkTokenCountMetric
from aiperf.metrics.types.output_token_throughput_metrics import (
    OutputTokenThroughputMetric,
)


class TestOutputTokenThroughputMetric:
    def test_output_token_throughput_calculation(self):
        """Test basic throughput calculation: tokens / duration"""
        metric = OutputTokenThroughputMetric()

        # 1000 tokens in 2 seconds = 500 tokens/second
        metric_results = MetricResultsDict()
        metric_results[BenchmarkTokenCountMetric.tag] = 1000
        metric_results[BenchmarkDurationMetric.tag] = (
            2_000_000_000  # 2 seconds in nanoseconds
        )

        result = metric.derive_value(metric_results)
        assert result == 500.0

    def test_output_token_throughput_fractional_duration(self):
        """Test throughput with fractional duration"""
        metric = OutputTokenThroughputMetric()

        # 750 tokens in 1.5 seconds = 500 tokens/second
        metric_results = MetricResultsDict()
        metric_results[BenchmarkTokenCountMetric.tag] = 750
        metric_results[BenchmarkDurationMetric.tag] = (
            1_500_000_000  # 1.5 seconds in nanoseconds
        )

        result = metric.derive_value(metric_results)
        assert result == 500.0

    def test_output_token_throughput_zero_duration_error(self):
        """Test error when benchmark duration is zero"""
        metric = OutputTokenThroughputMetric()

        metric_results = MetricResultsDict()
        metric_results[BenchmarkTokenCountMetric.tag] = 1000
        metric_results[BenchmarkDurationMetric.tag] = 0.0

        with pytest.raises(ValueError, match="Benchmark duration is not available"):
            metric.derive_value(metric_results)

    def test_output_token_throughput_none_duration_error(self):
        """Test error when benchmark duration is None"""
        metric = OutputTokenThroughputMetric()

        metric_results = MetricResultsDict()
        metric_results[BenchmarkTokenCountMetric.tag] = 1000
        metric_results[BenchmarkDurationMetric.tag] = None

        with pytest.raises(ValueError, match="Benchmark duration is not available"):
            metric.derive_value(metric_results)
