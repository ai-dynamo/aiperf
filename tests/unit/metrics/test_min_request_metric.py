# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import AggregationKind
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.types.min_request_metric import MinRequestTimestampMetric
from tests.unit.metrics.conftest import create_record, run_simple_metrics_pipeline


class TestMinRequestTimestampMetric:
    def test_min_request_timestamp(self) -> None:
        """Test min request timestamp extraction"""
        record = create_record(start_ns=1500)

        metric = MinRequestTimestampMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 1500  # Uses timestamp_ns which equals start_ns

    def test_uses_min_aggregation_kind(self) -> None:
        """The accumulator folds the per-record timestamps to the run minimum."""
        assert MinRequestTimestampMetric.aggregation_kind == AggregationKind.MIN

    def test_min_request_aggregation(self) -> None:
        """Per-record parse values fold to the minimum via aggregation_kind."""
        records = [
            create_record(start_ns=2000),  # timestamp: 2000
            create_record(start_ns=1000),  # timestamp: 1000 (minimum)
            create_record(start_ns=3000),  # timestamp: 3000
        ]

        results = run_simple_metrics_pipeline(records, MinRequestTimestampMetric.tag)

        assert results[MinRequestTimestampMetric.tag] == 1000
