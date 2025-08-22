# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.exceptions import NoMetricValue
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.types.inter_token_latency_metric import InterTokenLatencyMetric
from aiperf.metrics.types.output_token_throughput_metrics import (
    OutputTokenThroughputPerUserMetric,
)
from tests.metrics.conftest import create_record


class TestOutputTokenThroughputPerUserMetric:
    def test_output_token_throughput_per_user_calculation(self):
        """Test throughput per user calculation: 1 / ITL"""
        record = create_record()  # Simple record, ITL value will be provided directly

        metric = OutputTokenThroughputPerUserMetric()

        # Provide ITL value in nanoseconds (will be converted to seconds internally)
        metric_dict = MetricRecordDict()
        metric_dict[InterTokenLatencyMetric.tag] = (
            100_000_000  # 0.1 seconds in nanoseconds
        )

        result = metric.parse_record(record, metric_dict)
        assert result == 10.0  # 1 / 0.1 = 10 tokens/second

    def test_output_token_throughput_per_user_zero_itl_error(self):
        """Test error when ITL is zero"""
        record = create_record()

        metric = OutputTokenThroughputPerUserMetric()
        metric_dict = MetricRecordDict()
        metric_dict[InterTokenLatencyMetric.tag] = 0.0

        with pytest.raises(NoMetricValue):
            metric.parse_record(record, metric_dict)

    def test_output_token_throughput_per_user_none_itl_error(self):
        """Test error when ITL is None"""
        record = create_record()

        metric = OutputTokenThroughputPerUserMetric()
        metric_dict = MetricRecordDict()
        metric_dict[InterTokenLatencyMetric.tag] = None

        with pytest.raises(NoMetricValue):
            metric.parse_record(record, metric_dict)
