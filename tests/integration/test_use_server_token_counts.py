# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for --use-server-token-count flag behavior."""

import pytest
from pytest import approx

from tests.harness.utils import AIPerfCLI
from tests.integration.conftest import IntegrationTestDefaults as defaults


@pytest.mark.integration
@pytest.mark.asyncio
class TestUseServerTokenCounts:
    """Tests that --use-server-token-count correctly uses server-reported token counts."""

    @pytest.mark.parametrize(
        "streaming,extra_inputs",
        [
            (False, ""),
            (True, """--extra-inputs '{"stream_options": {"include_usage": true}}'"""),
        ],
        ids=["non_streaming", "streaming"],
    )
    async def test_server_token_counts_match_primary_metrics(
        self, cli: AIPerfCLI, mock_server_factory, streaming: bool, extra_inputs: str
    ):
        """Verify primary metrics use server-reported token counts when flag is enabled.

        When --use-server-token-count is set:
        - input_sequence_length should equal usage_prompt_tokens
        - output_token_count should equal usage_completion_tokens - usage_reasoning_tokens
        - reasoning_token_count should equal usage_reasoning_tokens
        - Usage diff metrics should NOT be present (they compare client vs server)
        """
        streaming_flag = "--streaming" if streaming else ""
        async with mock_server_factory(fast=True, workers=1) as aiperf_mock_server:
            result = await cli.run(
                f"""
                aiperf profile \
                    --model openai/gpt-oss-120b \
                    --url {aiperf_mock_server.url} \
                    --endpoint-type chat \
                    {streaming_flag} \
                    {extra_inputs} \
                    --use-server-token-count \
                    --request-count {defaults.request_count} \
                    --concurrency {defaults.concurrency} \
                    --workers-max {defaults.workers_max} \
                    --ui {defaults.ui}
                """
            )

            # Verify primary metrics match server-reported usage fields
            for key in ["avg", "min", "max", "p50", "p75", "p90", "p95", "p99"]:
                assert getattr(result.json.input_sequence_length, key) == approx(
                    result.json.usage_prompt_tokens[key]
                ), f"input_sequence_length.{key} should match usage_prompt_tokens.{key}"

                assert getattr(result.json.output_token_count, key) == approx(
                    result.json.usage_completion_tokens[key]
                    - result.json.usage_reasoning_tokens[key]
                ), (
                    f"output_token_count.{key} should match usage_completion - usage_reasoning"
                )

                assert getattr(result.json.reasoning_token_count, key) == approx(
                    result.json.usage_reasoning_tokens[key]
                ), (
                    f"reasoning_token_count.{key} should match usage_reasoning_tokens.{key}"
                )

            # Usage diff metrics compare client vs server counts, so they should
            # not be present when using server token counts exclusively
            json_data = result.json.model_dump()
            assert "usage_prompt_tokens_diff_pct" not in json_data
            assert "usage_completion_tokens_diff_pct" not in json_data
            assert "usage_reasoning_tokens_diff_pct" not in json_data
            assert "usage_discrepancy_count" not in json_data
