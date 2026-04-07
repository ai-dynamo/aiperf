# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.enums import MetricFlags
from aiperf.common.exceptions import NoMetricValue
from aiperf.metrics.metric_dicts import MetricRecordDict, MetricResultsDict
from aiperf.metrics.types.reasoning_token_count import (
    ReasoningTokenCountLocalMetric,
    ReasoningTokenCountMetric,
    ReasoningTokenCountServerMetric,
    TotalReasoningTokensMetric,
)
from tests.unit.metrics.conftest import (
    create_metric_array,
    create_record,
    run_simple_metrics_pipeline,
)


class TestReasoningTokenCountMetric:
    def test_reasoning_token_count_basic(self):
        """Test basic reasoning token count extraction"""
        record = create_record(output_tokens_per_response=10, reasoning_tokens=5)

        metric = ReasoningTokenCountMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 5

    def test_reasoning_token_count_zero(self):
        """Test handling of zero reasoning tokens"""
        record = create_record(output_tokens_per_response=10, reasoning_tokens=0)

        metric = ReasoningTokenCountMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 0

    def test_reasoning_token_count_none(self):
        """Test handling of None reasoning tokens raises error"""
        record = create_record(output_tokens_per_response=10)
        # reasoning defaults to None

        metric = ReasoningTokenCountMetric()
        with pytest.raises(NoMetricValue):
            metric.parse_record(record, MetricRecordDict())

    def test_reasoning_token_count_fallback_to_local(self):
        """Test that combined metric falls back to reasoning_local when server is None"""
        record = create_record(
            output_tokens_per_response=10,
            reasoning_local_tokens=7,
        )
        # reasoning defaults to None

        metric = ReasoningTokenCountMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 7

    def test_reasoning_token_count_prefers_server(self):
        """Test that combined metric prefers server over local"""
        record = create_record(
            output_tokens_per_response=10,
            reasoning_tokens=5,
            reasoning_local_tokens=7,
        )

        metric = ReasoningTokenCountMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 5

    def test_reasoning_token_count_both_none(self):
        """Test that NoMetricValue is raised when both server and local are None"""
        record = create_record(output_tokens_per_response=10)
        record.token_counts.reasoning = None
        record.token_counts.reasoning_local = None

        metric = ReasoningTokenCountMetric()
        with pytest.raises(NoMetricValue):
            metric.parse_record(record, MetricRecordDict())

    def test_reasoning_token_count_multiple_records(self):
        """Test processing multiple records with different reasoning token counts"""
        records = []
        reasoning_counts = [5, 10, 15]
        for count in reasoning_counts:
            record = create_record(
                output_tokens_per_response=20, reasoning_tokens=count
            )
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


class TestReasoningTokenCountServerMetric:
    def test_server_metric_basic(self):
        """Test server metric returns server-reported value"""
        record = create_record(output_tokens_per_response=10, reasoning_tokens=5)

        metric = ReasoningTokenCountServerMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 5

    def test_server_metric_zero(self):
        """Test server metric returns zero when server reports zero"""
        record = create_record(output_tokens_per_response=10, reasoning_tokens=0)

        metric = ReasoningTokenCountServerMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 0

    def test_server_metric_none_raises(self):
        """Test server metric raises NoMetricValue when server value is None"""
        record = create_record(output_tokens_per_response=10)
        # reasoning defaults to None

        metric = ReasoningTokenCountServerMetric()
        with pytest.raises(NoMetricValue):
            metric.parse_record(record, MetricRecordDict())

    def test_server_metric_metadata(self):
        """Test that ReasoningTokenCountServerMetric has correct metadata"""
        assert ReasoningTokenCountServerMetric.tag == "reasoning_token_count_server"
        assert ReasoningTokenCountServerMetric.has_flags(
            MetricFlags.PRODUCES_TOKENS_ONLY
        )
        assert ReasoningTokenCountServerMetric.has_flags(MetricFlags.LARGER_IS_BETTER)
        assert ReasoningTokenCountServerMetric.has_flags(MetricFlags.SUPPORTS_REASONING)
        assert ReasoningTokenCountServerMetric.has_flags(MetricFlags.NO_CONSOLE)


class TestReasoningTokenCountLocalMetric:
    def test_local_metric_basic(self):
        """Test local metric returns client-computed value"""
        record = create_record(reasoning_local_tokens=7)

        metric = ReasoningTokenCountLocalMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 7

    def test_local_metric_zero(self):
        """Test local metric returns zero when client computes zero"""
        record = create_record(reasoning_local_tokens=0)

        metric = ReasoningTokenCountLocalMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 0

    def test_local_metric_none_raises(self):
        """Test local metric raises NoMetricValue when local value is None"""
        record = create_record()
        # reasoning_local defaults to None

        metric = ReasoningTokenCountLocalMetric()
        with pytest.raises(NoMetricValue):
            metric.parse_record(record, MetricRecordDict())

    def test_local_metric_metadata(self):
        """Test that ReasoningTokenCountLocalMetric has correct metadata"""
        assert ReasoningTokenCountLocalMetric.tag == "reasoning_token_count_local"
        assert ReasoningTokenCountLocalMetric.has_flags(
            MetricFlags.PRODUCES_TOKENS_ONLY
        )
        assert ReasoningTokenCountLocalMetric.has_flags(MetricFlags.LARGER_IS_BETTER)
        assert ReasoningTokenCountLocalMetric.has_flags(MetricFlags.SUPPORTS_REASONING)
        assert ReasoningTokenCountLocalMetric.has_flags(MetricFlags.NO_CONSOLE)


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
