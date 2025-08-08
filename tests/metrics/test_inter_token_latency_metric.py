# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pytest import approx

from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.types.inter_token_latency_metric import InterTokenLatencyMetric
from aiperf.metrics.types.output_sequence_length_metric import (
    OutputSequenceLengthMetric,
)
from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric
from aiperf.metrics.types.ttft_metric import TTFTMetric
from tests.metrics.conftest import create_record, run_simple_metrics_pipeline


class TestInterTokenLatencyMetric:
    def test_inter_token_latency_basic_calculation(self):
        """Test ITL calculation: (request_latency - ttft) / (output_tokens - 1)"""

        record = create_record(
            start_ns=100, responses=[120, 200], output_tokens_per_response=3
        )

        metric_results = run_simple_metrics_pipeline(
            [record],
            RequestLatencyMetric.tag,
            TTFTMetric.tag,
            OutputSequenceLengthMetric.tag,
            InterTokenLatencyMetric.tag,
        )

        # start=100, first_response=120 (ttft=20), last_response=200 (request_latency=100)
        # 2 responses, 3 tokens per response, 6 total tokens
        # ITL = (100 - 20) / (6 - 1) = 16.0
        assert metric_results[InterTokenLatencyMetric.tag] == approx([16.0])

    def test_inter_token_latency_streaming_scenario(self):
        """Test ITL with multi-response streaming scenario"""
        record = create_record(
            start_ns=1000, responses=[1040, 1080, 1120], output_tokens_per_response=3
        )

        metric_results = run_simple_metrics_pipeline(
            [record],
            RequestLatencyMetric.tag,
            TTFTMetric.tag,
            OutputSequenceLengthMetric.tag,
            InterTokenLatencyMetric.tag,
        )

        # start=1000, responses at 1040, 1080, 1120
        # 3 responses, 3 tokens per response, 9 total tokens
        # TTFT=40, total latency=120, output=9 tokens
        # ITL = (120 - 40) / (9 - 1) = 10.0
        assert metric_results[InterTokenLatencyMetric.tag] == approx([10.0])

    def test_inter_token_latency_insufficient_tokens(self):
        """Test that ITL raises error when output tokens < 2"""
        record = create_record(output_tokens_per_response=1)

        with pytest.raises(
            ValueError, match="Output sequence length must be at least 2"
        ):
            run_simple_metrics_pipeline(
                [record],
                RequestLatencyMetric.tag,
                OutputSequenceLengthMetric.tag,
                TTFTMetric.tag,
                InterTokenLatencyMetric.tag,
            )

    def test_inter_token_latency_missing_required_metrics(self):
        """Test that ITL requires all dependency metrics"""
        record = create_record()
        empty_metrics = MetricRecordDict()

        with pytest.raises(ValueError, match="Missing required metric"):
            InterTokenLatencyMetric().parse_record(record, empty_metrics)
