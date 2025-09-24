# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.exceptions import NoMetricValue
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.types.inter_chunk_latency_metric import InterChunkLatencyMetric
from tests.metrics.conftest import create_record, run_simple_metrics_pipeline


class TestInterChunkLatencyMetric:
    def test_inter_chunk_latency_basic(self):
        """Test basic inter-chunk latency calculation"""
        # Start at 100ns, responses at 110ns, 120ns, 135ns
        # ICL = [120-110, 135-120] = [10, 15]
        record = create_record(start_ns=100, responses=[110, 120, 135])

        metric_results = run_simple_metrics_pipeline(
            [record],
            InterChunkLatencyMetric.tag,
        )
        assert metric_results[InterChunkLatencyMetric.tag] == [[10, 15]]

    def test_inter_chunk_latency_two_responses(self):
        """Test ICL with exactly two responses"""
        # Start at 100ns, responses at 110ns, 125ns
        # ICL = [125-110] = [15]
        record = create_record(start_ns=100, responses=[110, 125])

        metric_results = run_simple_metrics_pipeline(
            [record],
            InterChunkLatencyMetric.tag,
        )
        assert metric_results[InterChunkLatencyMetric.tag] == [[15]]

    @pytest.mark.parametrize(
        "responses, expected_icl",
        [
            ([100, 110, 120], [10, 10]),  # Equal intervals
            ([100, 105, 115, 130], [5, 10, 15]),  # Increasing intervals
            ([100, 120, 125, 135], [20, 5, 10]),  # Mixed intervals
            ([100, 150], [50]),  # Single interval
            ([1000, 1040, 1080, 1120, 1200], [40, 40, 40, 80]),  # Larger values
        ],
    )  # fmt: skip
    def test_inter_chunk_latency_various_intervals(self, responses, expected_icl):
        """Test ICL calculation with various response intervals"""
        record = create_record(start_ns=50, responses=responses)

        metric_results = run_simple_metrics_pipeline(
            [record],
            InterChunkLatencyMetric.tag,
        )
        assert metric_results[InterChunkLatencyMetric.tag] == [expected_icl]

    def test_inter_chunk_latency_multiple_records(self):
        """Test processing multiple records"""
        records = [
            create_record(start_ns=100, responses=[110, 120, 130]),  # ICL = [10, 10]
            create_record(start_ns=200, responses=[205, 210]),  # ICL = [5]
            create_record(start_ns=300, responses=[310, 325, 340, 360]),  # ICL = [15, 15, 20]
        ]  # fmt: skip

        metric_results = run_simple_metrics_pipeline(
            records,
            InterChunkLatencyMetric.tag,
        )
        assert metric_results[InterChunkLatencyMetric.tag] == [
            [10, 10],
            [5],
            [15, 15, 20],
        ]

    @pytest.mark.parametrize("num_responses", [2, 3, 5, 10, 100])
    def test_inter_chunk_latency_scaling(self, num_responses):
        """Test ICL calculation scales correctly with number of responses"""
        # Create responses with 10ns intervals starting at 100ns
        responses = [100 + (i * 10) for i in range(num_responses)]
        expected_icl = [10] * (num_responses - 1)

        record = create_record(start_ns=50, responses=responses)

        metric_results = run_simple_metrics_pipeline(
            [record],
            InterChunkLatencyMetric.tag,
        )
        assert metric_results[InterChunkLatencyMetric.tag] == [expected_icl]

    def test_inter_chunk_latency_insufficient_responses(self):
        """Test error when less than two responses"""
        record = create_record(start_ns=100, responses=[110])  # Only one response

        metric = InterChunkLatencyMetric()
        with pytest.raises(
            NoMetricValue,
            match="Record must have at least two responses to calculate Inter Chunk Latency",
        ):
            metric.parse_record(record, MetricRecordDict())

    def test_inter_chunk_latency_empty_responses(self):
        """Test error when no responses"""
        record = create_record(start_ns=100, responses=[])  # No responses

        metric = InterChunkLatencyMetric()
        with pytest.raises(
            NoMetricValue,
            match="Record must have at least two responses to calculate Inter Chunk Latency",
        ):
            metric.parse_record(record, MetricRecordDict())

    def test_inter_chunk_latency_invalid_order(self):
        """Test error when response timestamps are not in chronological order"""
        record = create_record(
            start_ns=100, responses=[120, 110, 130]
        )  # Second response before first

        metric = InterChunkLatencyMetric()
        with pytest.raises(
            ValueError,
            match="Each inter chunk latency must be positive",
        ):
            metric.parse_record(record, MetricRecordDict())

    def test_inter_chunk_latency_equal_timestamps(self):
        """Test error when consecutive responses have equal timestamps"""
        record = create_record(
            start_ns=100, responses=[110, 110, 120]
        )  # Equal timestamps

        metric = InterChunkLatencyMetric()
        with pytest.raises(
            ValueError,
            match="Each inter chunk latency must be positive",
        ):
            metric.parse_record(record, MetricRecordDict())

    @pytest.mark.parametrize(
        "responses",
        [
            [100, 90, 110],  # Second response before first
            [100, 110, 105],  # Third response before second
            [100, 110, 120, 115],  # Fourth response before third
            [100, 100, 110],  # Equal first two timestamps
            [100, 110, 110],  # Equal last two timestamps
        ],
    )  # fmt: skip
    def test_inter_chunk_latency_invalid_timestamp_order(self, responses):
        """Test various invalid timestamp ordering scenarios"""
        record = create_record(start_ns=50, responses=responses)

        metric = InterChunkLatencyMetric()
        with pytest.raises(
            ValueError,
            match="Each inter chunk latency must be positive",
        ):
            metric.parse_record(record, MetricRecordDict())

    def test_inter_chunk_latency_streaming_scenario(self):
        """Test ICL in a realistic streaming scenario"""
        # Simulate a streaming response with varying chunk arrival times
        # First chunk comes quickly (TTFT), then subsequent chunks arrive with varying delays
        record = create_record(
            start_ns=1000,
            responses=[1050, 1080, 1120, 1180, 1250, 1330],  # Increasing delays
        )

        metric_results = run_simple_metrics_pipeline(
            [record],
            InterChunkLatencyMetric.tag,
        )

        # Expected ICL = [1080-1050, 1120-1080, 1180-1120, 1250-1180, 1330-1250]
        # = [30, 40, 60, 70, 80]
        assert metric_results[InterChunkLatencyMetric.tag] == [[30, 40, 60, 70, 80]]

    def test_inter_chunk_latency_direct_parse_record(self):
        """Test direct parse_record method call"""
        record = create_record(start_ns=100, responses=[110, 125, 140])
        metric = InterChunkLatencyMetric()
        metric_dict = MetricRecordDict()

        result = metric.parse_record(record, metric_dict)

        # Expected ICL = [125-110, 140-125] = [15, 15]
        assert result == [15, 15]

    @pytest.mark.parametrize("num_records", [1, 5, 10, 50])
    def test_inter_chunk_latency_multiple_records_scaling(self, num_records):
        """Test ICL calculation with multiple records of varying sizes"""
        records = []
        expected_results = []

        for i in range(num_records):
            # Create records with different numbers of responses
            num_responses = 2 + (i % 5)  # 2-6 responses per record
            start_time = 1000 + (i * 1000)
            responses = [start_time + (j * 20) for j in range(num_responses)]

            records.append(
                create_record(start_ns=start_time - 100, responses=responses)
            )
            expected_results.append([20] * (num_responses - 1))

        metric_results = run_simple_metrics_pipeline(
            records,
            InterChunkLatencyMetric.tag,
        )

        assert metric_results[InterChunkLatencyMetric.tag] == expected_results
