# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for --use-server-token-count flag behavior (deprecated, now always-on)."""

import pytest
from pytest import approx

from tests.harness.utils import AIPerfCLI
from tests.integration.conftest import IntegrationTestDefaults as defaults


@pytest.mark.integration
@pytest.mark.asyncio
class TestUseServerTokenCounts:
    """Tests that server-reported token counts are always used for output metrics."""

    @pytest.mark.parametrize(
        "streaming,extra_inputs",
        [
            (False, ""),
            (True, ""),
        ],
        ids=["non_streaming", "streaming"],
    )
    async def test_server_token_counts_match_primary_metrics(
        self, cli: AIPerfCLI, mock_server_factory, streaming: bool, extra_inputs: str
    ):
        """Verify primary metrics use server-reported token counts.

        With the always-both approach:
        - input_sequence_length should equal usage_prompt_tokens (server-preferred)
        - output_token_count should equal usage_completion_tokens - usage_reasoning_tokens
        - reasoning_token_count should equal usage_reasoning_tokens
        - Usage prompt diff metric should be present (compares client vs server input)
        - Usage completion/reasoning diff metrics should NOT be present (require --tokenize-output)
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

            # Prompt diff should be present (always computed now)
            json_data = result.json.model_dump()
            assert "usage_prompt_tokens_diff_pct" in json_data
            assert "usage_discrepancy_count" in json_data

            # Completion and reasoning diff metrics require --tokenize-output to
            # populate both server and client values; without it they produce no value.
            assert "usage_completion_tokens_diff_pct" not in json_data
            assert "usage_reasoning_tokens_diff_pct" not in json_data
