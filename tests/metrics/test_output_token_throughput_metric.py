# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.enums import MetricTag
from aiperf.metrics.types.output_token_throughput_metric import (
    OutputTokenThroughputMetric,
)


class MockBenchmarkDuration:
    tag = MetricTag.BENCHMARK_DURATION

    def values(self):
        return 5_000_000_000  # 5 seconds


class MockOutputTokenCount:
    tag = MetricTag.OUTPUT_TOKEN_COUNT

    def values(self):
        return [10, 20, 30]  # total = 60


@pytest.fixture
def mock_metrics():
    return {
        MetricTag.BENCHMARK_DURATION: MockBenchmarkDuration(),
        MetricTag.OUTPUT_TOKEN_COUNT: MockOutputTokenCount(),
    }


def test_output_token_throughput_metric(mock_metrics):
    metric = OutputTokenThroughputMetric()
    metric.update_value(metrics=mock_metrics)
    expected = 60 / 5  # 60 tokens / 5 seconds
    assert metric.values() == expected
