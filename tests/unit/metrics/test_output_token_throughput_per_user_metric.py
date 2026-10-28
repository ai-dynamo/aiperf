# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.exceptions import NoMetricValue
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.types.inter_token_latency_metric import (
    FullResponseInterTokenLatencyMetric,
    InterTokenLatencyMetric,
)
from aiperf.metrics.types.output_token_throughput_metrics import (
    FullResponseOutputTokenThroughputPerUserMetric,
    OutputTokenThroughputPerUserMetric,
)
from tests.unit.metrics.conftest import create_record, run_simple_metrics_pipeline


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


class TestFullResponseOutputTokenThroughputPerUserMetric:
    def test_kimi_parser_gap_uses_full_request_end(self) -> None:
        request_start_ns = 1_000_000_000
        record = create_record(
            start_ns=request_start_ns,
            responses=[1_529_058_811, 1_610_559_573],
        )
        record.request.end_perf_ns = request_start_ns + 145_861_451_008
        assert record.token_counts is not None
        record.token_counts.output = 26_571

        metric_results = run_simple_metrics_pipeline(
            [record],
            OutputTokenThroughputPerUserMetric.tag,
            FullResponseOutputTokenThroughputPerUserMetric.tag,
        )

        assert metric_results[OutputTokenThroughputPerUserMetric.tag] == pytest.approx(
            [326_009.2218524288]
        )
        assert metric_results[
            FullResponseOutputTokenThroughputPerUserMetric.tag
        ] == pytest.approx([182.82228456622666])

    def test_calculates_inverse_of_full_response_itl(self) -> None:
        record = create_record()
        metric_dict = MetricRecordDict(
            {
                FullResponseInterTokenLatencyMetric.tag: 100_000_000,
            }
        )

        result = FullResponseOutputTokenThroughputPerUserMetric().parse_record(
            record, metric_dict
        )

        assert result == 10.0

    def test_rejects_zero_full_response_itl(self) -> None:
        record = create_record()
        metric_dict = MetricRecordDict(
            {
                FullResponseInterTokenLatencyMetric.tag: 0,
            }
        )

        with pytest.raises(NoMetricValue, match="ITL is zero"):
            FullResponseOutputTokenThroughputPerUserMetric().parse_record(
                record, metric_dict
            )
