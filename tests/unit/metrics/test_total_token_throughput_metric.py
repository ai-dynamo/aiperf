# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.exceptions import NoMetricValue
from aiperf.metrics.metric_dicts import MetricResultsDict
from aiperf.metrics.types.benchmark_duration_metric import BenchmarkDurationMetric
from aiperf.metrics.types.input_sequence_length_metric import (
    TotalInputSequenceLengthMetric,
)
from aiperf.metrics.types.output_sequence_length_metric import (
    TotalOutputSequenceLengthMetric,
)
from aiperf.metrics.types.total_token_throughput import TotalTokenThroughputMetric


class TestTotalTokenThroughputMetric:
    @pytest.mark.parametrize(
        "input_tokens,output_tokens,duration,expected",
        [
            (600, 400, 2, 500.0),  # basic: (600+400) / 2s
            (500, 250, 1.5, 500.0),  # fractional duration: (500+250) / 1.5s
            (0, 0, 1, 0.0),  # zero tokens
            (1000, 0, 1, 1000.0),  # only input tokens
            (0, 1000, 1, 1000.0),  # only output tokens
            (1, 1, 1, 2.0),  # minimal tokens
            (500_000, 500_000, 1, 1_000_000.0),  # large token counts
            (50, 50, 0.1, 1000.0),  # small duration: (50+50) / 0.1s
        ],
    )  # fmt: skip
    def test_total_token_throughput_calculation(
        self, input_tokens: int, output_tokens: int, duration: float, expected: float
    ):
        """Test throughput calculation: (input_tokens + output_tokens) / duration"""
        metric = TotalTokenThroughputMetric()

        metric_results = MetricResultsDict()
        metric_results[TotalInputSequenceLengthMetric.tag] = input_tokens
        metric_results[TotalOutputSequenceLengthMetric.tag] = output_tokens
        metric_results[BenchmarkDurationMetric.tag] = duration * NANOS_PER_SECOND

        result = metric.derive_value(metric_results)
        assert result == pytest.approx(expected)

    @pytest.mark.parametrize("duration", [0, 0.0, None])
    def test_total_token_throughput_invalid_duration_raises(self, duration: float):
        """Test error when benchmark duration is zero or None"""
        metric = TotalTokenThroughputMetric()

        metric_results = MetricResultsDict()
        metric_results[TotalInputSequenceLengthMetric.tag] = 600
        metric_results[TotalOutputSequenceLengthMetric.tag] = 400
        metric_results[BenchmarkDurationMetric.tag] = duration

        with pytest.raises(NoMetricValue):
            metric.derive_value(metric_results)
