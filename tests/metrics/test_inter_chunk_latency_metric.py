# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pytest import approx

from aiperf.common.exceptions import NoMetricValue
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.types.inter_chunk_latency_metric import InterChunkLatencyMetric
from tests.metrics.conftest import create_record, run_simple_metrics_pipeline


class TestInterChunkLatencyMetric:
    @pytest.mark.parametrize(
        "responses,expected_icl,description",
        [
            ([110, 130], 20.0, "basic two responses"),
            ([110, 120, 140], 15.0, "multiple chunks (avg of 10,20)"),
            ([1010, 1025, 1040, 1060, 1075], 16.25, "streaming scenario"),
        ],
    )
    def test_inter_chunk_latency_calculations(
        self, responses: list[int], expected_icl: float, description: str
    ):
        """Test ICL calculations with various response patterns"""
        record = create_record(start_ns=100, responses=responses)

        metric_results = run_simple_metrics_pipeline(
            [record], InterChunkLatencyMetric.tag
        )
        assert metric_results[InterChunkLatencyMetric.tag] == approx([expected_icl])

    def test_inter_chunk_latency_multiple_records(self):
        """Test processing multiple records with different ICL values"""
        records = [
            create_record(start_ns=100, responses=[110, 130]),  # ICL = 20ns
            create_record(start_ns=200, responses=[210, 220, 240]),  # ICL = 15ns
            create_record(start_ns=300, responses=[310, 320]),  # ICL = 10ns
        ]  # fmt: skip

        metric_results = run_simple_metrics_pipeline(
            records, InterChunkLatencyMetric.tag
        )
        assert metric_results[InterChunkLatencyMetric.tag] == approx([20.0, 15.0, 10.0])

    @pytest.mark.parametrize(
        "responses,error_type,error_match,description",
        [
            (
                [110],
                NoMetricValue,
                "Record must have at least two responses",
                "single response",
            ),
            ([], NoMetricValue, "Invalid Record", "no responses"),
            (
                [130, 110],
                ValueError,
                "Inter chunk latencies must be positive",
                "descending timestamps",
            ),
            (
                [110, 110],
                ValueError,
                "Inter chunk latencies must be positive",
                "equal timestamps",
            ),
        ],
    )
    def test_inter_chunk_latency_errors(
        self, responses: list[int], error_type: type, error_match: str, description: str
    ):
        """Test error conditions for ICL metric"""
        record = create_record(start_ns=100, responses=responses)
        if not responses:
            record.responses = []

        metric = InterChunkLatencyMetric()
        with pytest.raises(error_type, match=error_match):
            metric.parse_record(record, MetricRecordDict())
