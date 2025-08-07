# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.utils import close_enough
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.types.ttst_metric import TTSTMetric
from tests.metrics.conftest import create_record, run_simple_metrics_pipeline


class TestTTSTMetric:
    def test_ttst_basic(self):
        """Test basic time to second token calculation"""
        # Start at 100ns, first response at 110ns, second response at 120ns = 10ns TTST
        record = create_record(start_ns=100, responses=[110, 120])

        metric_results = run_simple_metrics_pipeline(
            [record],
            TTSTMetric.tag,
        )
        assert close_enough(metric_results[TTSTMetric.tag], 10)

    def test_ttst_multiple_responses(self):
        """Test TTST with multiple responses uses first and second only"""
        # Start at 100ns, responses at 105ns, 110ns, 130ns = 5ns TTST (110 - 105)
        record = create_record(start_ns=100, responses=[105, 110, 130])

        metric_results = run_simple_metrics_pipeline(
            [record],
            TTSTMetric.tag,
        )
        assert close_enough(metric_results[TTSTMetric.tag], 5)

    def test_ttst_multiple_records(self):
        """Test processing multiple records"""
        records = [
            create_record(start_ns=100, responses=[105, 110]),  # 5ns TTST
            create_record(start_ns=200, responses=[205, 210]),  # 5ns TTST
            create_record(start_ns=300, responses=[310, 325]),  # 15ns TTST
        ]

        metric_results = run_simple_metrics_pipeline(
            records,
            TTSTMetric.tag,
        )
        assert close_enough(metric_results[TTSTMetric.tag], [5, 5, 15])

    def test_ttst_invalid_order(self):
        """Test error when second response is before first response"""
        record = create_record(
            start_ns=100, responses=[120, 110]
        )  # Second before first

        metric = TTSTMetric()
        with pytest.raises(
            ValueError,
            match="Second response timestamp must be greater than or equal to the first response timestamp",
        ):
            metric.parse_record(record, MetricRecordDict())

    def test_ttst_insufficient_responses(self):
        """Test error when less than two responses"""
        record = create_record(start_ns=100, responses=[110])  # Only one response

        metric = TTSTMetric()
        with pytest.raises(
            ValueError,
            match="Record must have at least two responses to calculate TTST",
        ):
            metric.parse_record(record, MetricRecordDict())
