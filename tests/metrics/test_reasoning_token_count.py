# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.enums import MetricFlags
from aiperf.common.exceptions import NoMetricValue
from aiperf.metrics.metric_dicts import MetricRecordDict, MetricResultsDict
from aiperf.metrics.types.reasoning_token_count import (
    ReasoningTokenCountMetric,
    TotalReasoningTokensMetric,
)
from tests.metrics.conftest import (
    create_metric_array,
    create_record,
    run_simple_metrics_pipeline,
)


class TestReasoningTokenCountMetric:
    def test_reasoning_token_count_basic(self):
        """Test basic reasoning token count extraction"""
        record = create_record(output_tokens_per_response=10)
        record.reasoning_token_count = 5

        metric = ReasoningTokenCountMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 5

    def test_reasoning_token_count_zero(self):
        """Test handling of zero reasoning tokens"""
        record = create_record(output_tokens_per_response=10)
        record.reasoning_token_count = 0

        metric = ReasoningTokenCountMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 0

    def test_reasoning_token_count_none(self):
        """Test handling of None reasoning tokens raises error"""
        record = create_record(output_tokens_per_response=10)
        record.reasoning_token_count = None

        metric = ReasoningTokenCountMetric()
        with pytest.raises(NoMetricValue):
            metric.parse_record(record, MetricRecordDict())

    def test_reasoning_token_count_multiple_records(self):
        """Test processing multiple records with different reasoning token counts"""
        records = []
        reasoning_counts = [5, 10, 15]
        for count in reasoning_counts:
            record = create_record(output_tokens_per_response=20)
            record.reasoning_token_count = count
            records.append(record)

        metric_results = run_simple_metrics_pipeline(
            records,
            ReasoningTokenCountMetric.tag,
        )
        assert metric_results[ReasoningTokenCountMetric.tag] == reasoning_counts

    def test_reasoning_token_count_metadata(self):
        """Test that ReasoningTokenCountMetric has correct metadata"""
        assert ReasoningTokenCountMetric.has_flags(MetricFlags.PRODUCES_TOKENS_ONLY)
        assert ReasoningTokenCountMetric.has_flags(MetricFlags.SUPPORTS_REASONING)
        assert ReasoningTokenCountMetric.has_flags(MetricFlags.NO_CONSOLE)
        assert ReasoningTokenCountMetric.missing_flags(MetricFlags.INTERNAL)


class TestTotalReasoningTokensMetric:
    @pytest.mark.parametrize(
        "values, expected_sum",
        [
            ([5, 10, 20], 35),
            ([50], 50),
            ([], 0),
            ([1], 1),
            ([0, 0, 0], 0),
        ],
    )
    def test_sum_calculation(self, values, expected_sum):
        """Test that TotalReasoningTokensMetric correctly sums all reasoning token counts"""
        metric = TotalReasoningTokensMetric()
        metric_results = MetricResultsDict()
        metric_results[ReasoningTokenCountMetric.tag] = create_metric_array(values)

        result = metric.derive_value(metric_results)
        assert result == expected_sum

    def test_metric_metadata(self):
        """Test that TotalReasoningTokensMetric has correct metadata and does not inherit SUPPORTS_REASONING"""
        assert TotalReasoningTokensMetric.tag == "total_reasoning_tokens"
        assert TotalReasoningTokensMetric.has_flags(MetricFlags.PRODUCES_TOKENS_ONLY)
        assert TotalReasoningTokensMetric.has_flags(MetricFlags.NO_CONSOLE)
        assert TotalReasoningTokensMetric.missing_flags(MetricFlags.SUPPORTS_REASONING)
        assert TotalReasoningTokensMetric.missing_flags(MetricFlags.INTERNAL)
