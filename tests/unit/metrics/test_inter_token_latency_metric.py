# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pytest import approx

from aiperf.common.exceptions import NoMetricValue
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.types.inter_token_latency_metric import InterTokenLatencyMetric
from aiperf.metrics.types.output_sequence_length_metric import (
    OutputSequenceLengthMetric,
)
from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric
from aiperf.metrics.types.ttft_metric import TTFTMetric
from tests.unit.metrics.conftest import create_record, run_simple_metrics_pipeline


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

        metric_results = run_simple_metrics_pipeline(
            [record],
            RequestLatencyMetric.tag,
            OutputSequenceLengthMetric.tag,
            TTFTMetric.tag,
            InterTokenLatencyMetric.tag,
        )

        # ITL should not be available when output tokens < 2
        assert (
            InterTokenLatencyMetric.tag not in metric_results
            or len(metric_results[InterTokenLatencyMetric.tag]) == 0
        )

    def test_inter_token_latency_missing_required_metrics(self):
        """Test that ITL requires all dependency metrics"""
        record = create_record()
        empty_metrics = MetricRecordDict()

        with pytest.raises(NoMetricValue):
            InterTokenLatencyMetric().parse_record(record, empty_metrics)

    def test_inter_token_latency_subtracts_bundled_first_chunk(self):
        """When the server reports the first content chunk carried N tokens, ITL
        divides the decode window by (OSL - N), not (OSL - 1), so a server that
        bundles the first chunk cannot inflate TPS/user."""
        record = create_record(
            start_ns=100,
            responses=[120, 200],
            output_tokens_per_response=3,
            first_content_chunk_tokens=3,  # first chunk carried 3 tokens, not 1
        )

        metric_results = run_simple_metrics_pipeline(
            [record],
            RequestLatencyMetric.tag,
            TTFTMetric.tag,
            OutputSequenceLengthMetric.tag,
            InterTokenLatencyMetric.tag,
        )

        # ttft=20, latency=100, OSL=6, first chunk=3
        # ITL = (100 - 20) / (6 - 3) = 26.6667, vs the inflated (100-20)/(6-1)=16
        assert metric_results[InterTokenLatencyMetric.tag] == approx([80.0 / 3.0])

    def test_inter_token_latency_first_chunk_one_matches_legacy(self):
        """A one-token first chunk reproduces the legacy (OSL - 1) formula exactly."""
        record = create_record(
            start_ns=100,
            responses=[120, 200],
            output_tokens_per_response=3,
            first_content_chunk_tokens=1,
        )

        metric_results = run_simple_metrics_pipeline(
            [record],
            RequestLatencyMetric.tag,
            TTFTMetric.tag,
            OutputSequenceLengthMetric.tag,
            InterTokenLatencyMetric.tag,
        )

        assert metric_results[InterTokenLatencyMetric.tag] == approx([16.0])

    def test_inter_token_latency_absent_first_chunk_falls_back_to_one(self):
        """When per-chunk usage is unavailable (None), ITL falls back to (OSL - 1)."""
        record = create_record(
            start_ns=100,
            responses=[120, 200],
            output_tokens_per_response=3,
            first_content_chunk_tokens=None,
        )

        metric_results = run_simple_metrics_pipeline(
            [record],
            RequestLatencyMetric.tag,
            TTFTMetric.tag,
            OutputSequenceLengthMetric.tag,
            InterTokenLatencyMetric.tag,
        )

        assert metric_results[InterTokenLatencyMetric.tag] == approx([16.0])

    def test_inter_token_latency_all_tokens_in_first_chunk_not_emitted(self):
        """When the entire output arrived in the first chunk there is no decode
        window, so ITL is undefined and not emitted."""
        record = create_record(
            start_ns=100,
            responses=[120, 200],
            output_tokens_per_response=3,
            first_content_chunk_tokens=6,  # all 6 tokens in the first chunk
        )

        metric_results = run_simple_metrics_pipeline(
            [record],
            RequestLatencyMetric.tag,
            TTFTMetric.tag,
            OutputSequenceLengthMetric.tag,
            InterTokenLatencyMetric.tag,
        )

        assert (
            InterTokenLatencyMetric.tag not in metric_results
            or len(metric_results[InterTokenLatencyMetric.tag]) == 0
        )
