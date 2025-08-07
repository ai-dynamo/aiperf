# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.utils import close_enough
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric
from tests.metrics.conftest import create_record, run_simple_metrics_pipeline


class TestRequestLatencyMetric:
    def test_request_latency_basic(self):
        """Test basic request latency calculation"""
        # Start at 100ns, response at 150ns = 50ns latency
        record = create_record(start_ns=100, responses=[150])

        metric_results = run_simple_metrics_pipeline(
            [record],
            RequestLatencyMetric.tag,
        )
        assert close_enough(metric_results[RequestLatencyMetric.tag], 50)

    def test_request_latency_multiple_responses(self):
        """Test latency with multiple responses uses final response timestamp"""
        # Start at 10ns, responses at 15ns, 25ns, 35ns = 25ns latency (final - start)
        record = create_record(start_ns=10, responses=[15, 25, 35])

        metric_results = run_simple_metrics_pipeline(
            [record],
            RequestLatencyMetric.tag,
        )
        assert close_enough(metric_results[RequestLatencyMetric.tag], 25)

    def test_request_latency_multiple_records(self):
        """Test processing multiple records"""
        records = [
            create_record(start_ns=10, responses=[25]),  # 15ns latency
            create_record(start_ns=20, responses=[35]),  # 15ns latency
            create_record(start_ns=30, responses=[50]),  # 20ns latency
        ]

        metric_results = run_simple_metrics_pipeline(
            records,
            RequestLatencyMetric.tag,
        )

        assert close_enough(metric_results[RequestLatencyMetric.tag], [15, 15, 20])

    def test_request_latency_invalid_timestamp(self):
        """Test error when response timestamp is before request start"""
        # Response at 90ns before request start at 100ns - should raise error
        record = create_record(start_ns=100, responses=[90])

        metric = RequestLatencyMetric()
        with pytest.raises(ValueError, match="Invalid Record"):
            metric.parse_record(record, MetricRecordDict())
