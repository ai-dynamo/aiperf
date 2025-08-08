# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pytest import approx

from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.types.output_sequence_length_metric import (
    OutputSequenceLengthMetric,
)
from tests.metrics.conftest import create_record, run_simple_metrics_pipeline


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
        with pytest.raises(ValueError, match="Output token count is missing"):
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
