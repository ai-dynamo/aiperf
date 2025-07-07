#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
import pytest

from aiperf.common.record_models import (
    ParsedResponseRecord,
    RequestRecord,
    ResponseData,
)
from aiperf.services.records_manager.metrics.types.request_throughput_metric import (
    RequestThroughputMetric,
)


@pytest.fixture
def mock_benchmark_duration():
    class MockBenchmarkDuration:
        tag = "benchmark_duration"

        def values(self):
            return 3_000_000_000  # 3s

    return MockBenchmarkDuration()


@pytest.fixture
def sample_records():
    return [
        ParsedResponseRecord(
            worker_id="worker-1",
            request=RequestRecord(start_perf_ns=0, timestamp_ns=0, has_error=False),
            responses=[
                ResponseData(
                    perf_ns=10,
                    token_count=10,
                    raw_text=["hello"],
                    parsed_text=["hello"],
                )
            ],
        ),
        ParsedResponseRecord(
            worker_id="worker-1",
            request=RequestRecord(start_perf_ns=10, timestamp_ns=10, has_error=False),
            responses=[
                ResponseData(
                    perf_ns=20,
                    token_count=10,
                    raw_text=["world"],
                    parsed_text=["world"],
                )
            ],
        ),
        ParsedResponseRecord(
            worker_id="worker-1",
            request=RequestRecord(start_perf_ns=20, timestamp_ns=20, has_error=False),
            responses=[
                ResponseData(
                    perf_ns=30, token_count=10, raw_text=["done"], parsed_text=["done"]
                )
            ],
        ),
    ]


def test_add_multiple_records(mock_benchmark_duration, sample_records):
    request_throughput = RequestThroughputMetric()
    for record in sample_records:
        request_throughput.update_value(record)

    metrics = {"benchmark_duration": mock_benchmark_duration}
    request_throughput.update_value(record=None, metrics=metrics)

    assert request_throughput.values() == pytest.approx(1.0)


def test_no_records(mock_benchmark_duration):
    metric = RequestThroughputMetric()

    metric.update_value(
        record=None, metrics={"benchmark_duration": mock_benchmark_duration}
    )

    assert metric.values() == 0.0
