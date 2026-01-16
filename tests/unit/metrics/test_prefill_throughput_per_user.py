# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.exceptions import NoMetricValue
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.types.input_sequence_length_metric import InputSequenceLengthMetric
from aiperf.metrics.types.prefill_throughput_per_user import (
    PrefillThroughputPerUserMetric,
)
from aiperf.metrics.types.ttft_metric import TTFTMetric
from tests.unit.metrics.conftest import create_record


class TestPrefillThroughputPerUserMetric:
    def test_prefill_throughput_per_user_calculation(self):
        """Test prefill throughput per user calculation: ISL / TTFT"""
        record = create_record()

        metric = PrefillThroughputPerUserMetric()

        # Provide ISL and TTFT values
        metric_dict = MetricRecordDict()
        metric_dict[InputSequenceLengthMetric.tag] = 1000  # 1000 tokens
        metric_dict[TTFTMetric.tag] = 100_000_000  # 0.1 seconds in nanoseconds

        result = metric.parse_record(record, metric_dict)
        assert result == 10000.0  # 1000 / 0.1 = 10,000 tokens/second

    def test_prefill_throughput_per_user_calculation_various_values(self):
        """Test with various ISL and TTFT values."""
        record = create_record()
        metric = PrefillThroughputPerUserMetric()

        test_cases = [
            (500, 50_000_000, 10000.0),  # 500 tokens, 0.05s -> 10,000 tps
            (100, 10_000_000, 10000.0),  # 100 tokens, 0.01s -> 10,000 tps
            (2000, 1_000_000_000, 2000.0),  # 2000 tokens, 1s -> 2,000 tps
            (50, 5_000_000, 10000.0),  # 50 tokens, 0.005s -> 10,000 tps
        ]

        for isl, ttft_ns, expected in test_cases:
            metric_dict = MetricRecordDict()
            metric_dict[InputSequenceLengthMetric.tag] = isl
            metric_dict[TTFTMetric.tag] = ttft_ns

            result = metric.parse_record(record, metric_dict)
            assert result == pytest.approx(expected, rel=1e-6)

    def test_prefill_throughput_per_user_zero_ttft_error(self):
        """Test error when TTFT is zero."""
        record = create_record()

        metric = PrefillThroughputPerUserMetric()
        metric_dict = MetricRecordDict()
        metric_dict[InputSequenceLengthMetric.tag] = 1000
        metric_dict[TTFTMetric.tag] = 0.0

        with pytest.raises(
            NoMetricValue,
            match="TTFT is zero, cannot calculate prefill throughput per user metric",
        ):
            metric.parse_record(record, metric_dict)

    def test_prefill_throughput_per_user_missing_isl_error(self):
        """Test error when ISL is missing."""
        record = create_record()

        metric = PrefillThroughputPerUserMetric()
        metric_dict = MetricRecordDict()
        metric_dict[TTFTMetric.tag] = 100_000_000

        # Should raise when trying to get ISL
        with pytest.raises(NoMetricValue):
            metric.parse_record(record, metric_dict)

    def test_prefill_throughput_per_user_missing_ttft_error(self):
        """Test error when TTFT is missing."""
        record = create_record()

        metric = PrefillThroughputPerUserMetric()
        metric_dict = MetricRecordDict()
        metric_dict[InputSequenceLengthMetric.tag] = 1000

        # Should raise when trying to get TTFT
        with pytest.raises(NoMetricValue):
            metric.parse_record(record, metric_dict)

    def test_prefill_throughput_per_user_metric_properties(self):
        """Test that metric properties are correctly defined."""
        metric = PrefillThroughputPerUserMetric()

        assert metric.tag == "prefill_throughput_per_user"
        assert metric.header == "Prefill Throughput Per User"
        assert metric.short_header == "Prefill TPS/User"
        assert metric.short_header_hide_unit is True
        assert InputSequenceLengthMetric.tag in metric.required_metrics
        assert TTFTMetric.tag in metric.required_metrics
