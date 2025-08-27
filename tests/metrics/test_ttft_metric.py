# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.exceptions import NoMetricValue
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.types.ttft_metric import TTFTMetric
from tests.metrics.conftest import create_record, run_simple_metrics_pipeline


class TestTTFTMetric:
    def test_ttft_basic(self):
        """Test basic time to first token calculation"""
        # Start at 100ns, first response at 110ns = 10ns TTFT
        record = create_record(start_ns=100, responses=[110])

        metric_results = run_simple_metrics_pipeline(
            [record],
            TTFTMetric.tag,
        )
        assert metric_results[TTFTMetric.tag] == [10]

    def test_ttft_multiple_responses(self):
        """Test TTFT with multiple responses uses first response only"""
        # Start at 100ns, first response at 105ns, second at 115ns = 5ns TTFT
        record = create_record(start_ns=100, responses=[105, 115])

        metric_results = run_simple_metrics_pipeline(
            [record],
            TTFTMetric.tag,
        )
        assert metric_results[TTFTMetric.tag] == [5]

    def test_ttft_multiple_records(self):
        """Test processing multiple records"""
        records = [
            create_record(start_ns=100, responses=[105, 115]),  # 5ns TTFT
            create_record(start_ns=200, responses=[205, 215]),  # 5ns TTFT
            create_record(start_ns=300, responses=[310, 320]),  # 10ns TTFT
        ]

        metric_results = run_simple_metrics_pipeline(
            records,
            TTFTMetric.tag,
        )
        assert metric_results[TTFTMetric.tag] == [5, 5, 10]

    def test_ttft_invalid_timestamp(self):
        """Test error when first response timestamp is before request start"""
        record = create_record(start_ns=100, responses=[90])

        metric = TTFTMetric()
        with pytest.raises(NoMetricValue, match="Invalid Record"):
            metric.parse_record(record, MetricRecordDict())

    def test_ttft_no_responses(self):
        """Test error when no responses are present"""
        record = create_record(start_ns=100)
        record.responses = []

        metric = TTFTMetric()
        with pytest.raises(NoMetricValue, match="Invalid Record"):
            metric.parse_record(record, MetricRecordDict())
