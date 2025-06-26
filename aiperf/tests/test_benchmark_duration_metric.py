#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
from aiperf.common.record_models import RequestRecord, SSEMessage
from aiperf.services.records_manager.metrics.types.benchmark_duration_metric import (
    BenchmarkDurationMetric,
)
from aiperf.services.records_manager.metrics.types.max_response_metric import (
    MaxResponseMetric,
)
from aiperf.services.records_manager.metrics.types.min_request_metric import (
    MinRequestMetric,
)


def test_add_multiple_records():
    metrics = {}
    metrics[MinRequestMetric.tag] = MinRequestMetric()
    metrics[MaxResponseMetric.tag] = MaxResponseMetric()
    records = [
        RequestRecord(recv_start_perf_ns=10, responses=[SSEMessage(perf_ns=15)]),
        RequestRecord(recv_start_perf_ns=20, responses=[SSEMessage(perf_ns=25)]),
        RequestRecord(recv_start_perf_ns=30, responses=[SSEMessage(perf_ns=40)]),
    ]
    # Creating metrics based on records
    for record in records:
        for metric in metrics.values():
            metric.update_value(record=record, metrics=None)

    # Creating the metrics based on the metrics already calculated
    benchmark_duration_metric = BenchmarkDurationMetric()
    benchmark_duration_metric.update_value(record=None, metrics=metrics)
    assert benchmark_duration_metric.values() == 30.0  # 40 - 10
