# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.enums import MetricFlags
from aiperf.common.exceptions import NoMetricValue
from aiperf.metrics.metric_dicts import MetricRecordDict, MetricResultsDict
from aiperf.metrics.types.input_sequence_length_metric import (
    ErrorInputSequenceLengthMetric,
    InputSequenceLengthMetric,
    TotalErrorInputSequenceLengthMetric,
    TotalInputSequenceLengthMetric,
)
from tests.metrics.conftest import (
    create_metric_array,
    create_record,
    run_simple_metrics_pipeline,
)


class TestInputSequenceLengthMetric:
    def test_input_sequence_length_basic(self):
        """Test basic input sequence length extraction"""
        record = create_record(input_tokens=15)

        metric = InputSequenceLengthMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 15

    def test_input_sequence_length_zero(self):
        """Test handling of zero input tokens"""
        record = create_record(input_tokens=0)

        metric = InputSequenceLengthMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 0

    def test_input_sequence_length_none(self):
        """Test handling of None input tokens raises error"""
        record = create_record(input_tokens=None)

        metric = InputSequenceLengthMetric()
        with pytest.raises(NoMetricValue):
            metric.parse_record(record, MetricRecordDict())

    def test_input_sequence_length_multiple_records(self):
        """Test processing multiple records with different token counts"""
        isl_values = [5, 10, 20]
        records = [create_record(input_tokens=isl) for isl in isl_values]

        metric_results = run_simple_metrics_pipeline(
            records,
            InputSequenceLengthMetric.tag,
        )
        assert metric_results[InputSequenceLengthMetric.tag] == isl_values


class TestTotalInputSequenceLengthMetric:
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
        """Test that TotalInputSequenceLengthMetric correctly sums all input tokens"""
        metric = TotalInputSequenceLengthMetric()
        metric_results = MetricResultsDict()
        metric_results[InputSequenceLengthMetric.tag] = create_metric_array(values)

        result = metric.derive_value(metric_results)
        assert result == expected_sum

    def test_metric_metadata(self):
        """Test that TotalInputSequenceLengthMetric has correct metadata"""
        assert TotalInputSequenceLengthMetric.tag == "total_isl"
        assert TotalInputSequenceLengthMetric.has_flags(
            MetricFlags.PRODUCES_TOKENS_ONLY
        )
        assert TotalInputSequenceLengthMetric.has_flags(MetricFlags.LARGER_IS_BETTER)
        assert TotalInputSequenceLengthMetric.has_flags(MetricFlags.NO_CONSOLE)
        assert TotalInputSequenceLengthMetric.missing_flags(MetricFlags.INTERNAL)


class TestErrorInputSequenceLengthMetric:
    def test_error_isl_basic(self):
        """Test basic error input sequence length extraction"""
        from aiperf.common.models import ErrorDetails

        record = create_record(
            input_tokens=15,
            error=ErrorDetails(code=500, message="Error", type="ServerError"),
        )

        metric = ErrorInputSequenceLengthMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 15

    def test_error_isl_none_raises(self):
        """Test handling of None input tokens raises error"""
        from aiperf.common.models import ErrorDetails

        record = create_record(
            input_tokens=None,
            error=ErrorDetails(code=500, message="Error", type="ServerError"),
        )

        metric = ErrorInputSequenceLengthMetric()
        with pytest.raises(NoMetricValue):
            metric.parse_record(record, MetricRecordDict())

    def test_error_isl_metadata(self):
        """Test that ErrorInputSequenceLengthMetric has correct flags"""
        assert ErrorInputSequenceLengthMetric.tag == "error_isl"
        assert ErrorInputSequenceLengthMetric.has_flags(MetricFlags.ERROR_ONLY)
        assert ErrorInputSequenceLengthMetric.has_flags(MetricFlags.NO_CONSOLE)


class TestTotalErrorInputSequenceLengthMetric:
    @pytest.mark.parametrize(
        "values, expected_sum",
        [
            ([10, 20, 30], 60),
            ([100], 100),
            ([], 0),
        ],
    )
    def test_sum_calculation(self, values, expected_sum):
        """Test that TotalErrorInputSequenceLengthMetric correctly sums error input tokens"""
        metric = TotalErrorInputSequenceLengthMetric()
        metric_results = MetricResultsDict()
        metric_results[ErrorInputSequenceLengthMetric.tag] = create_metric_array(values)

        result = metric.derive_value(metric_results)
        assert result == expected_sum

    def test_metric_metadata(self):
        """Test that TotalErrorInputSequenceLengthMetric has correct metadata"""
        assert TotalErrorInputSequenceLengthMetric.tag == "total_error_isl"
        assert TotalErrorInputSequenceLengthMetric.has_flags(MetricFlags.ERROR_ONLY)
        assert TotalErrorInputSequenceLengthMetric.has_flags(MetricFlags.NO_CONSOLE)
