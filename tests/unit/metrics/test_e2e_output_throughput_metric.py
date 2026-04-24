# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.exceptions import NoMetricValue
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.types.e2e_output_throughput_metric import (
    E2EOutputTokenThroughputMetric,
)
from aiperf.metrics.types.output_sequence_length_metric import (
    OutputSequenceLengthMetric,
)
from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric
from tests.unit.metrics.conftest import create_record


class TestE2EOutputTokenThroughputMetric:
    def test_e2e_output_token_throughput_calculation(self):
        """100 tokens / 2 seconds = 50 tokens/sec"""
        record = create_record()
        metric = E2EOutputTokenThroughputMetric()

        metric_dict = MetricRecordDict()
        metric_dict[OutputSequenceLengthMetric.tag] = 100
        metric_dict[RequestLatencyMetric.tag] = 2_000_000_000  # 2 seconds in ns

        result = metric.parse_record(record, metric_dict)
        assert result == 50.0

    def test_e2e_output_token_throughput_calculation_various_values(self):
        """Test with various OSL and latency values."""
        record = create_record()
        metric = E2EOutputTokenThroughputMetric()

        test_cases = [
            (200, 1_000_000_000, 200.0),  # 200 tokens, 1s -> 200 tps
            (500, 500_000_000, 1000.0),  # 500 tokens, 0.5s -> 1000 tps
            (50, 10_000_000_000, 5.0),  # 50 tokens, 10s -> 5 tps
        ]

        for osl, latency_ns, expected in test_cases:
            metric_dict = MetricRecordDict()
            metric_dict[OutputSequenceLengthMetric.tag] = osl
            metric_dict[RequestLatencyMetric.tag] = latency_ns

            result = metric.parse_record(record, metric_dict)
            assert result == pytest.approx(expected, rel=1e-6)

    def test_e2e_output_token_throughput_zero_latency_raises_no_metric_value(self):
        record = create_record()
        metric = E2EOutputTokenThroughputMetric()

        metric_dict = MetricRecordDict()
        metric_dict[OutputSequenceLengthMetric.tag] = 100
        metric_dict[RequestLatencyMetric.tag] = 0

        with pytest.raises(
            NoMetricValue,
            match="Request latency is zero, cannot calculate E2E output token throughput",
        ):
            metric.parse_record(record, metric_dict)

    def test_e2e_output_token_throughput_missing_osl_raises_no_metric_value(self):
        record = create_record()
        metric = E2EOutputTokenThroughputMetric()

        metric_dict = MetricRecordDict()
        metric_dict[RequestLatencyMetric.tag] = 1_000_000_000

        with pytest.raises(NoMetricValue):
            metric.parse_record(record, metric_dict)

    def test_e2e_output_token_throughput_missing_latency_raises_no_metric_value(self):
        record = create_record()
        metric = E2EOutputTokenThroughputMetric()

        metric_dict = MetricRecordDict()
        metric_dict[OutputSequenceLengthMetric.tag] = 100

        with pytest.raises(NoMetricValue):
            metric.parse_record(record, metric_dict)

    def test_e2e_output_token_throughput_metric_properties(self):
        metric = E2EOutputTokenThroughputMetric()

        assert metric.tag == "e2e_output_token_throughput"
        assert metric.header == "E2E Output Token Throughput"
        assert metric.short_header == "E2E Output TPS/User"
        assert metric.short_header_hide_unit is True
        assert OutputSequenceLengthMetric.tag in metric.required_metrics
        assert RequestLatencyMetric.tag in metric.required_metrics
