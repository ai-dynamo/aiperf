# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.enums import MetricFlags
from aiperf.common.exceptions import NoMetricValue
from aiperf.metrics.metric_dicts import MetricRecordDict, MetricResultsDict
from aiperf.metrics.types.output_token_count import (
    OutputTokenCountMetric,
    TotalOutputTokensMetric,
)
from tests.metrics.conftest import (
    create_metric_array,
    create_record,
    run_simple_metrics_pipeline,
)


class TestOutputTokenCountMetric:
    def test_output_token_count_basic(self):
        """Test basic output token count extraction"""
        record = create_record(output_tokens_per_response=25)

        metric = OutputTokenCountMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 25

    def test_output_token_count_zero(self):
        """Test handling of zero output tokens raises error"""
        record = create_record(output_tokens_per_response=0)

        metric = OutputTokenCountMetric()
        with pytest.raises(NoMetricValue):
            metric.parse_record(record, MetricRecordDict())

    def test_output_token_count_none(self):
        """Test handling of None output tokens raises error"""
        record = create_record()
        record.output_token_count = None

        metric = OutputTokenCountMetric()
        with pytest.raises(NoMetricValue):
            metric.parse_record(record, MetricRecordDict())

    def test_output_token_count_multiple_records(self):
        """Test processing multiple records with different token counts"""
        token_counts = [15, 30, 45]
        records = [create_record(output_tokens_per_response=tc) for tc in token_counts]

        metric_results = run_simple_metrics_pipeline(
            records,
            OutputTokenCountMetric.tag,
        )
        assert metric_results[OutputTokenCountMetric.tag] == token_counts

    def test_output_token_count_metadata(self):
        """Test that OutputTokenCountMetric has correct metadata"""
        assert OutputTokenCountMetric.has_flags(MetricFlags.PRODUCES_TOKENS_ONLY)
        assert OutputTokenCountMetric.has_flags(MetricFlags.NO_CONSOLE)
        assert OutputTokenCountMetric.missing_flags(MetricFlags.INTERNAL)


class TestTotalOutputTokensMetric:
    @pytest.mark.parametrize(
        "values, expected_sum",
        [
            ([10, 25, 40], 75),
            ([150], 150),
            ([], 0),
            ([1], 1),
            ([0, 0, 0], 0),
        ],
    )
    def test_sum_calculation(self, values, expected_sum):
        """Test that TotalOutputTokensMetric correctly sums all output token counts"""
        metric = TotalOutputTokensMetric()
        metric_results = MetricResultsDict()
        metric_results[OutputTokenCountMetric.tag] = create_metric_array(values)

        result = metric.derive_value(metric_results)
        assert result == expected_sum

    def test_metric_metadata(self):
        """Test that TotalOutputTokensMetric has correct metadata"""
        assert TotalOutputTokensMetric.tag == "total_output_tokens"
        assert TotalOutputTokensMetric.has_flags(MetricFlags.PRODUCES_TOKENS_ONLY)
        assert TotalOutputTokensMetric.has_flags(MetricFlags.LARGER_IS_BETTER)
        assert TotalOutputTokensMetric.has_flags(MetricFlags.NO_CONSOLE)
        assert TotalOutputTokensMetric.missing_flags(MetricFlags.INTERNAL)
