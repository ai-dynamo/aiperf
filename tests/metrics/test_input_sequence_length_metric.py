# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.utils import close_enough
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.types.input_sequence_length_metric import InputSequenceLengthMetric
from tests.metrics.conftest import create_record, run_simple_metrics_pipeline


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
        with pytest.raises(ValueError, match="Input Token Count is not available"):
            metric.parse_record(record, MetricRecordDict())

    def test_input_sequence_length_multiple_records(self):
        """Test processing multiple records with different token counts"""
        isl_values = [5, 10, 20]
        records = [create_record(input_tokens=isl) for isl in isl_values]

        metric_results = run_simple_metrics_pipeline(
            records,
            InputSequenceLengthMetric.tag,
        )
        assert close_enough(metric_results[InputSequenceLengthMetric.tag], isl_values)
