# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pytest import approx

from aiperf.common.exceptions import NoMetricValue
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.types.output_sequence_length_metric import (
    OutputSequenceLengthMetric,
)
from aiperf.metrics.types.tokens_per_chunk import TokensPerChunkMetric
from tests.metrics.conftest import create_record, run_simple_metrics_pipeline


class TestTokensPerChunkMetric:
    @pytest.mark.parametrize(
        "responses,tokens_per_response,expected_tpc,description",
        [
            ([100, 110], 3, 3.0, "basic calculation"),
            ([100, 110, 120, 130], 5, 5.0, "multiple chunks"),
            ([1000, 1010, 1020, 1030, 1040], 5, 5.0, "streaming scenario"),
        ],  # fmt: skip
    )
    def test_tokens_per_chunk_calculations(
        self,
        responses: list[int],
        tokens_per_response: int,
        expected_tpc: float,
        description: str,
    ):
        """Test TPC calculations with various response patterns"""
        record = create_record(
            responses=responses, output_tokens_per_response=tokens_per_response
        )

        metric_results = run_simple_metrics_pipeline(
            [record], OutputSequenceLengthMetric.tag, TokensPerChunkMetric.tag
        )
        assert metric_results[TokensPerChunkMetric.tag] == approx([expected_tpc])

    def test_tokens_per_chunk_fractional_result(self):
        """Test TPC with fractional result"""
        record = create_record(
            responses=[100, 110, 120], output_tokens_per_response=int(10 / 3)
        )
        record.output_token_count = 10  # Override for exact calculation

        metric_results = run_simple_metrics_pipeline(
            [record], OutputSequenceLengthMetric.tag, TokensPerChunkMetric.tag
        )
        assert metric_results[TokensPerChunkMetric.tag] == approx([10.0 / 3.0])

    def test_tokens_per_chunk_multiple_records(self):
        """Test processing multiple records with different TPC values"""
        records = [
            create_record(responses=[100, 110], output_tokens_per_response=4),  # 4.0
            create_record(responses=[200, 210, 220], output_tokens_per_response=3),  # 3.0
            create_record(responses=[300, 310, 320, 330], output_tokens_per_response=2),  # 2.0
        ]  # fmt: skip

        metric_results = run_simple_metrics_pipeline(
            records, OutputSequenceLengthMetric.tag, TokensPerChunkMetric.tag
        )
        assert metric_results[TokensPerChunkMetric.tag] == approx([4.0, 3.0, 2.0])

    def test_tokens_per_chunk_insufficient_responses(self):
        """Test error when record has only 1 response (needs valid record for custom validation)"""
        record = create_record(
            start_ns=100, responses=[110], output_tokens_per_response=5
        )

        metric = TokensPerChunkMetric()
        metric_dict = MetricRecordDict()
        metric_dict[OutputSequenceLengthMetric.tag] = 5

        with pytest.raises(
            NoMetricValue, match="Record must have at least 2 responses"
        ):
            metric.parse_record(record, metric_dict)

    def test_tokens_per_chunk_no_responses_invalid(self):
        """Test error when record has no responses (Invalid Record)"""
        record = create_record(responses=[], output_tokens_per_response=5)
        record.responses = []

        metric = TokensPerChunkMetric()
        metric_dict = MetricRecordDict()
        metric_dict[OutputSequenceLengthMetric.tag] = 5

        with pytest.raises(NoMetricValue, match="Invalid Record"):
            metric.parse_record(record, metric_dict)

    def test_tokens_per_chunk_missing_dependency(self):
        """Test error when required OutputSequenceLengthMetric is missing"""
        record = create_record(
            start_ns=100, responses=[110, 120], output_tokens_per_response=5
        )

        metric = TokensPerChunkMetric()
        empty_metrics = MetricRecordDict()

        with pytest.raises(NoMetricValue, match="Missing required metric"):
            metric.parse_record(record, empty_metrics)

    def test_tokens_per_chunk_zero_tokens_pipeline(self):
        """Test zero tokens error through pipeline (get_or_raise treats 0 as falsy)"""
        record = create_record(responses=[100, 110], output_tokens_per_response=0)

        with pytest.raises(
            NoMetricValue, match="Metric output_sequence_length is not available"
        ):
            run_simple_metrics_pipeline(
                [record], OutputSequenceLengthMetric.tag, TokensPerChunkMetric.tag
            )
