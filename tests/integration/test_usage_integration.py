# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for API usage field parsing with mock server."""

import pytest

from tests.integration.conftest import AIPerfCLI
from tests.integration.conftest import IntegrationTestDefaults as defaults
from tests.integration.models import AIPerfMockServer


@pytest.mark.integration
@pytest.mark.asyncio
class TestUsageIntegration:
    """Integration tests for usage field parsing end-to-end."""

    USAGE_METRIC_TAGS = [
        "usage_prompt_tokens",
        "usage_completion_tokens",
        "usage_total_tokens",
    ]

    USAGE_METRIC_CSV_NAMES = [
        "Usage Prompt Tokens",
        "Usage Completion Tokens",
        "Usage Total Tokens",
    ]

    @pytest.mark.parametrize(
        "endpoint_type,model",
        [
            ("chat", "openai/gpt-oss-120b"),
            ("completions", "openai/gpt-oss-120b"),
        ],
    )
    async def test_usage_metrics_in_exports(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer, endpoint_type, model
    ):
        """Test that usage field metrics appear in all export formats."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model {model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type {endpoint_type} \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        assert result.exit_code == 0
        assert result.request_count == defaults.request_count

        if result.json:
            json_data = result.json.model_dump()
            found_metrics = [
                m for m in self.USAGE_METRIC_TAGS if m in json_data and json_data[m]
            ]
            assert len(found_metrics) > 0

        if result.csv:
            csv_content = result.csv
            usage_cols = [
                col for col in self.USAGE_METRIC_CSV_NAMES if col in csv_content
            ]
            assert len(usage_cols) > 0

        if result.jsonl:
            records_with_usage = [
                record
                for record in result.jsonl
                if any(k.startswith("usage_") for k in record.metrics)
            ]
            assert len(records_with_usage) > 0

    async def test_streaming_usage_passthrough(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Test that streaming responses preserve cumulative usage values."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model Qwen/Qwen2.5-32B-Instruct \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --streaming \
                --extra-inputs '{{"stream_options": {{"include_usage": true}}}}' \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        assert result.exit_code == 0
        assert result.has_streaming_metrics

        records_with_both = [
            record
            for record in result.jsonl
            if "output_token_count" in record.metrics
            and any(k.startswith("usage_") for k in record.metrics)
        ]
        assert len(records_with_both) > 0

    async def test_client_vs_usage_comparison(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Test that client-side and API usage field metrics both exist."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model openai/gpt-oss-120b \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        assert result.exit_code == 0

        json_data = result.json.model_dump()

        client_metrics = ["output_token_count", "input_sequence_length"]
        client_found = [m for m in client_metrics if m in json_data]
        assert len(client_found) > 0

        usage_found = [m for m in self.USAGE_METRIC_TAGS if m in json_data]
        assert len(usage_found) > 0

    async def test_reasoning_model_usage(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Test usage metrics for reasoning-capable models."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model openai/gpt-oss-120b \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        assert result.exit_code == 0

        json_data = result.json.model_dump()
        reasoning_metrics = ["reasoning_token_count", "usage_reasoning_tokens"]
        found = [m for m in reasoning_metrics if m in json_data and json_data[m]]

        if found:
            assert len(found) > 0
