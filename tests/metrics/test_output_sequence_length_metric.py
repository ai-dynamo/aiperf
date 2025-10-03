# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pytest import approx

from aiperf.common.enums import MetricFlags
from aiperf.common.exceptions import NoMetricValue
from aiperf.metrics.metric_dicts import MetricRecordDict, MetricResultsDict
from aiperf.metrics.types.output_sequence_length_metric import (
    OutputSequenceLengthMetric,
    TotalOutputSequenceLengthMetric,
)
from tests.metrics.conftest import (
    create_metric_array,
    create_record,
    run_simple_metrics_pipeline,
)


class TestOutputSequenceLengthMetric:
    def test_output_sequence_length_basic(self):
        """Test basic output sequence length extraction"""
        record = create_record(output_tokens_per_response=10)

        metric = OutputSequenceLengthMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 10

    def test_output_sequence_length_zero(self):
        """Test handling of zero output tokens"""
        record = create_record(output_tokens_per_response=0)

        metric = OutputSequenceLengthMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 0

    def test_output_sequence_length_none(self):
        """Test handling of None output tokens raises error"""
        record = create_record()
        record.output_token_count = None

        metric = OutputSequenceLengthMetric()
        with pytest.raises(NoMetricValue):
            metric.parse_record(record, MetricRecordDict())

    def test_output_sequence_length_multiple_records(self):
        """Test processing multiple records with different token counts"""
        records = [
            create_record(output_tokens_per_response=5),
            create_record(output_tokens_per_response=10),
            create_record(output_tokens_per_response=15),
        ]
        metric_results = run_simple_metrics_pipeline(
            records,
            OutputSequenceLengthMetric.tag,
        )

        assert metric_results[OutputSequenceLengthMetric.tag] == approx([5, 10, 15])

    def test_output_sequence_length_with_reasoning_tokens(self):
        """Test output sequence length includes reasoning tokens"""
        record = create_record(output_tokens_per_response=10)
        record.reasoning_token_count = 5

        metric = OutputSequenceLengthMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 15


class TestTotalOutputSequenceLengthMetric:
    @pytest.mark.parametrize(
        "values, expected_sum",
        [
            ([10, 20, 30], 60),
            ([100], 100),
            ([], 0),
            ([1], 1),
            ([0, 0, 0], 0),
        ],
    )
    def test_sum_calculation(self, values, expected_sum):
        """Test that TotalOutputSequenceLengthMetric correctly sums all output tokens"""
        metric = TotalOutputSequenceLengthMetric()
        metric_results = MetricResultsDict()
        metric_results[OutputSequenceLengthMetric.tag] = create_metric_array(values)

        result = metric.derive_value(metric_results)
        assert result == expected_sum

    def test_metric_metadata(self):
        """Test that TotalOutputSequenceLengthMetric has correct metadata"""
        assert TotalOutputSequenceLengthMetric.tag == "total_osl"
        assert TotalOutputSequenceLengthMetric.has_flags(
            MetricFlags.PRODUCES_TOKENS_ONLY
        )
        assert TotalOutputSequenceLengthMetric.has_flags(MetricFlags.LARGER_IS_BETTER)
        assert TotalOutputSequenceLengthMetric.has_flags(MetricFlags.NO_CONSOLE)
        assert TotalOutputSequenceLengthMetric.missing_flags(MetricFlags.INTERNAL)
