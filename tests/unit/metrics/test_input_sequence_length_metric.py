# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.enums import MetricFlags
from aiperf.common.exceptions import NoMetricValue
from aiperf.metrics.metric_dicts import MetricRecordDict, MetricResultsDict
from aiperf.metrics.types.input_sequence_length_metric import (
    ErrorInputSequenceLengthMetric,
    InputSequenceLengthLocalMetric,
    InputSequenceLengthMetric,
    InputSequenceLengthServerMetric,
    TotalErrorInputSequenceLengthMetric,
    TotalInputSequenceLengthMetric,
)
from tests.unit.metrics.conftest import (
    create_metric_array,
    create_record,
    run_simple_metrics_pipeline,
)


class TestInputSequenceLengthMetric:
    @pytest.mark.parametrize(
        "input_tokens,input_local_tokens,expected",
        [
            (150, 8, 150),
            (0, 8, 0),
            (None, 8, 8),
            (200, None, 200),
        ],
        ids=["server_preferred", "server_zero", "server_none_fallback", "server_only"],
    )
    def test_input_sequence_length_prefers_server(
        self, input_tokens, input_local_tokens, expected
    ):
        """Test ISL prefers server-reported input over client-side input_local."""
        record = create_record(
            input_tokens=input_tokens, input_local_tokens=input_local_tokens
        )

        metric = InputSequenceLengthMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == expected

    def test_input_sequence_length_both_none_raises(self):
        """Test that NoMetricValue is raised when both server and local are None."""
        record = create_record(input_tokens=None, input_local_tokens=None)
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


class TestInputSequenceLengthServerMetric:
    @pytest.mark.parametrize(
        "input_tokens,should_raise",
        [
            (150, False),
            (0, False),
            (None, True),
        ],
    )
    def test_server_metric_parse_record(self, input_tokens, should_raise):
        """Test server metric returns server count or raises NoMetricValue."""
        record = create_record(input_tokens=input_tokens)

        metric = InputSequenceLengthServerMetric()

        if should_raise:
            with pytest.raises(NoMetricValue):
                metric.parse_record(record, MetricRecordDict())
        else:
            result = metric.parse_record(record, MetricRecordDict())
            assert result == input_tokens

    def test_server_metric_ignores_local(self):
        """Server metric should raise even when local tokens are available."""
        record = create_record(input_tokens=None, input_local_tokens=42)

        metric = InputSequenceLengthServerMetric()
        with pytest.raises(NoMetricValue):
            metric.parse_record(record, MetricRecordDict())

    def test_server_metric_flags(self):
        """Verify NO_CONSOLE flag is set."""
        assert InputSequenceLengthServerMetric.has_flags(MetricFlags.NO_CONSOLE)
        assert InputSequenceLengthServerMetric.has_flags(
            MetricFlags.TOKENIZES_INPUT_ONLY
        )
        assert InputSequenceLengthServerMetric.has_flags(MetricFlags.LARGER_IS_BETTER)


class TestInputSequenceLengthLocalMetric:
    @pytest.mark.parametrize(
        "input_local_tokens,should_raise",
        [
            (42, False),
            (0, False),
            (None, True),
        ],
    )
    def test_local_metric_parse_record(self, input_local_tokens, should_raise):
        """Test local metric returns local count or raises NoMetricValue."""
        record = create_record(input_local_tokens=input_local_tokens)

        metric = InputSequenceLengthLocalMetric()

        if should_raise:
            with pytest.raises(NoMetricValue):
                metric.parse_record(record, MetricRecordDict())
        else:
            result = metric.parse_record(record, MetricRecordDict())
            assert result == input_local_tokens

    def test_local_metric_ignores_server(self):
        """Local metric should raise even when server tokens are available."""
        record = create_record(input_tokens=99, input_local_tokens=None)

        metric = InputSequenceLengthLocalMetric()
        with pytest.raises(NoMetricValue):
            metric.parse_record(record, MetricRecordDict())

    def test_local_metric_flags(self):
        """Verify NO_CONSOLE flag is set."""
        assert InputSequenceLengthLocalMetric.has_flags(MetricFlags.NO_CONSOLE)
        assert InputSequenceLengthLocalMetric.has_flags(
            MetricFlags.TOKENIZES_INPUT_ONLY
        )
        assert InputSequenceLengthLocalMetric.has_flags(MetricFlags.LARGER_IS_BETTER)


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
            MetricFlags.TOKENIZES_INPUT_ONLY
        )
        assert TotalInputSequenceLengthMetric.has_flags(MetricFlags.LARGER_IS_BETTER)
        assert TotalInputSequenceLengthMetric.has_flags(MetricFlags.NO_CONSOLE)
        assert TotalInputSequenceLengthMetric.missing_flags(MetricFlags.INTERNAL)


class TestErrorInputSequenceLengthMetric:
    @pytest.mark.parametrize(
        "input_tokens,should_raise",
        [
            (15, False),
            (0, False),
            (None, True),
        ],
    )
    def test_error_isl_parse_record(self, input_tokens, should_raise):
        """Test error input sequence length extraction."""
        from aiperf.common.models import ErrorDetails

        record = create_record(
            input_tokens=input_tokens,
            error=ErrorDetails(code=500, message="Error", type="ServerError"),
        )

        metric = ErrorInputSequenceLengthMetric()

        if should_raise:
            with pytest.raises(NoMetricValue):
                metric.parse_record(record, MetricRecordDict())
        else:
            result = metric.parse_record(record, MetricRecordDict())
            assert result == input_tokens

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
