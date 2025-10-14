# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for validating aggregate metrics against raw JSONL data."""

import pytest

from tests.integration.conftest import AIPerfCLI
from tests.integration.metric_validators import (
    compute_stats,
    extract_metric_values,
    validate_all_metrics,
    validate_metric_stats,
)
from tests.integration.models import FakeAIServer


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skip(reason="These tests need to be improved")
class TestMetricValidation:
    """Tests for validating aggregate metrics against raw JSONL data."""

    async def test_validate_single_metric(
        self, cli: AIPerfCLI, fakeai_server: FakeAIServer
    ):
        """Validate a single metric's statistics are computed correctly."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --streaming \
                --request-count 50 \
                --concurrency 5 \
                --ui simple
            """
        )

        latency_values = extract_metric_values(result.jsonl, "request_latency")
        assert len(latency_values) == 50

        computed = compute_stats(latency_values)

        validate_metric_stats(computed, result.json.request_latency, "request_latency")

        ttft_values = extract_metric_values(result.jsonl, "time_to_first_token")
        computed_ttft = compute_stats(ttft_values)
        validate_metric_stats(
            computed_ttft, result.json.time_to_first_token, "time_to_first_token"
        )

    async def test_validate_all_metrics(
        self, cli: AIPerfCLI, fakeai_server: FakeAIServer
    ):
        """Validate all metrics in JSON export match computed values from JSONL."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model deepseek-ai/DeepSeek-R1 \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --streaming \
                --request-count 100 \
                --concurrency 10 \
                --ui simple
            """
        )

        computed_metrics = validate_all_metrics(result.jsonl, result.json)

        assert len(computed_metrics) > 0
        assert "request_latency" in computed_metrics
        assert "time_to_first_token" in computed_metrics
        assert "inter_token_latency" in computed_metrics

        for stats in computed_metrics.values():
            assert stats.count == result.request_count

    async def test_validate_non_streaming_metrics(
        self, cli: AIPerfCLI, fakeai_server: FakeAIServer
    ):
        """Validate non-streaming metrics are computed correctly."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model nvidia/llama-3.1-nemotron-70b-instruct \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --request-count 30 \
                --concurrency 3 \
                --ui simple
            """
        )

        computed_metrics = validate_all_metrics(result.jsonl, result.json)

        assert "request_latency" in computed_metrics
        assert "output_sequence_length" in computed_metrics
        assert "input_sequence_length" in computed_metrics

        assert result.json.time_to_first_token is None
