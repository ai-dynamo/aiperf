# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pytest import approx

from aiperf.common.enums import AggregationKind
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.types.max_response_metric import MaxResponseTimestampMetric
from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric
from tests.unit.metrics.conftest import create_record, run_simple_metrics_pipeline


class TestMaxResponseTimestampMetric:
    def test_max_response_calculation(self):
        """Test max response timestamp calculation with request latency"""
        # Record with start=100, response=150, latency=50
        record = create_record(start_ns=100, responses=[150])

        request_latency_metric = RequestLatencyMetric()
        max_response_metric = MaxResponseTimestampMetric()

        # Calculate request latency
        latency = request_latency_metric.parse_record(record, MetricRecordDict())
        assert latency == 50  # 150 - 100

        # Calculate max response timestamp (uses timestamp_ns + latency)
        metrics_dict = MetricRecordDict()
        metrics_dict[RequestLatencyMetric.tag] = latency

        result = max_response_metric.parse_record(record, metrics_dict)
        expected = 100 + 50  # start_ns + latency = timestamp_ns + latency
        assert result == approx(expected)

    def test_uses_max_aggregation_kind(self) -> None:
        """The accumulator folds the per-record response timestamps to the run max."""
        assert MaxResponseTimestampMetric.aggregation_kind == AggregationKind.MAX

    def test_max_response_aggregation(self) -> None:
        """Per-record final-response timestamps fold to the maximum via aggregation_kind."""
        records = [
            create_record(start_ns=100, responses=[150]),  # latency: 50, final: 150
            create_record(start_ns=200, responses=[300]),  # latency: 100, final: 300
            create_record(
                start_ns=300, responses=[350]
            ),  # latency: 50, final: 350 (max)
        ]

        results = run_simple_metrics_pipeline(records, MaxResponseTimestampMetric.tag)

        assert results[MaxResponseTimestampMetric.tag] == approx(350)
