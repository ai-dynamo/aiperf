# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for API usage field parsing with mock server."""

import pytest
from pytest import approx

from tests.harness.utils import AIPerfCLI
from tests.integration.conftest import IntegrationTestDefaults as defaults


@pytest.mark.integration
@pytest.mark.asyncio
class TestUseServerTokenCounts:
    """Integration tests for use server token counts."""

    async def test_use_server_token_counts_non_streaming(
        self, cli: AIPerfCLI, mock_server_factory
    ):
        """Test that when --use-server-token-count is set, the server token counts
        are used for non-streaming endpoints and match the expected values."""
        # Use isolated mock server with workers=1 to ensure consistent behavior
        async with mock_server_factory(fast=True, workers=1) as aiperf_mock_server:
            result = await cli.run(
                f"""
                aiperf profile \
                    --model openai/gpt-oss-120b \
                    --url {aiperf_mock_server.url} \
                    --endpoint-type chat \
                    --request-count {defaults.request_count} \
                    --concurrency {defaults.concurrency} \
                    --workers-max {defaults.workers_max} \
                    --use-server-token-count \
                    --ui {defaults.ui}
                """
            )

            for key in ["avg", "min", "max", "p50", "p75", "p90", "p95", "p99"]:
                assert getattr(result.json.input_sequence_length, key) == approx(
                    result.json.usage_prompt_tokens[key]
                )
                assert getattr(result.json.output_token_count, key) == approx(
                    result.json.usage_completion_tokens[key]
                    - result.json.usage_reasoning_tokens[key]
                )
                assert getattr(result.json.reasoning_token_count, key) == approx(
                    result.json.usage_reasoning_tokens[key]
                )
            # Make sure the usage diff metrics are not present.
            json_data = result.json.model_dump()
            assert "usage_prompt_tokens_diff_pct" not in json_data
            assert "usage_completion_tokens_diff_pct" not in json_data
            assert "usage_reasoning_tokens_diff_pct" not in json_data
            assert "usage_discrepancy_count" not in json_data

    async def test_use_server_token_counts_streaming(
        self, cli: AIPerfCLI, mock_server_factory
    ):
        """Test that when --use-server-token-count is set, the server token counts
        are used for streaming endpoints and match the expected values."""
        # Use isolated mock server with workers=1 to ensure consistent behavior
        async with mock_server_factory(fast=True, workers=1) as aiperf_mock_server:
            result = await cli.run(
                f"""
                aiperf profile \
                    --model openai/gpt-oss-120b \
                    --url {aiperf_mock_server.url} \
                    --endpoint-type chat \
                    --streaming \
                    --extra-inputs '{{"stream_options": {{"include_usage": true}}}}' \
                    --use-server-token-count \
                    --request-count {defaults.request_count} \
                    --concurrency {defaults.concurrency} \
                    --workers-max {defaults.workers_max} \
                    --ui {defaults.ui}
                """
            )

            for key in ["avg", "min", "max", "p50", "p75", "p90", "p95", "p99"]:
                assert getattr(result.json.input_sequence_length, key) == approx(
                    result.json.usage_prompt_tokens[key]
                )
                assert getattr(result.json.output_token_count, key) == approx(
                    result.json.usage_completion_tokens[key]
                    - result.json.usage_reasoning_tokens[key]
                )
                assert getattr(result.json.reasoning_token_count, key) == approx(
                    result.json.usage_reasoning_tokens[key]
                )
            # Make sure the usage diff metrics are not present.
            json_data = result.json.model_dump()
            assert "usage_prompt_tokens_diff_pct" not in json_data
            assert "usage_completion_tokens_diff_pct" not in json_data
            assert "usage_reasoning_tokens_diff_pct" not in json_data
            assert "usage_discrepancy_count" not in json_data
