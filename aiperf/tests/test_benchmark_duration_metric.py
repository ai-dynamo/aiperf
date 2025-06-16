#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
from aiperf.services.records_manager.metrics.types.benchmark_duration_metric import (
    BenchmarkDurationMetric,
)
from aiperf.services.records_manager.metrics.types.max_response_metric import (
    MaxResponseMetric,
)
from aiperf.services.records_manager.metrics.types.min_request_metric import (
    MinRequestMetric,
)
from aiperf.tests.utils.metric_test_utils import MockRecord, MockRequest, MockResponse


def test_add_multiple_records():
    metrics = {}
    metrics[MinRequestMetric.tag] = MinRequestMetric()
    metrics[MaxResponseMetric.tag] = MaxResponseMetric()
    records = [
        MockRecord(MockRequest(10), [MockResponse(15)]),
        MockRecord(MockRequest(20), [MockResponse(25)]),
        MockRecord(MockRequest(30), [MockResponse(40)]),
    ]
    # Creating metrics based on records
    for record in records:
        for metric in metrics.values():
            metric.update_value(record)

    # Creating the metrics based on the metrics already calculated
    benchmark_duration_metric = BenchmarkDurationMetric()
    benchmark_duration_metric.update_value(metrics)
    assert benchmark_duration_metric.values() == 30.0  # 40 - 10
