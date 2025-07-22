# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.services.records_manager.metrics.types.request_throughput_metric import (
    RequestThroughputMetric,
)


@pytest.fixture
def mock_metrics():
    class MockBenchmarkDuration:
        tag = "benchmark_duration"

        def values(self):
            return 3_000_000_000  # 3 seconds

    class MockRequestCount:
        tag = "request_count"

        def values(self):
            return 3

    return {
        "benchmark_duration": MockBenchmarkDuration(),
        "request_count": MockRequestCount(),
    }


def test_request_throughput(mock_metrics):
    metric = RequestThroughputMetric()
    metric.update_value(record=None, metrics=mock_metrics)

    # 3 requests / 3 seconds = 1.0 req/sec
    assert metric.values() == pytest.approx(1.0)
