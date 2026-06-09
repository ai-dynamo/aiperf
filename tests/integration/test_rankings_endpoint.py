# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0


from pathlib import Path

import pytest

from tests.harness.utils import AIPerfCLI, AIPerfMockServer
from tests.integration.conftest import IntegrationTestDefaults as defaults
from tests.integration.utils import create_rankings_dataset


@pytest.mark.integration
@pytest.mark.asyncio
class TestRankingsEndpoint:
    """Integration tests for ranking-type endpoints."""

    @pytest.mark.parametrize(
        "endpoint_type",
        ["nim_rankings", "hf_tei_rankings", "cohere_rankings"],
    )
    async def test_rankings_with_custom_dataset(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        tmp_path: Path,
        endpoint_type: str,
    ):
        """Test ranking endpoints with a custom dataset file."""
        dataset_path = create_rankings_dataset(tmp_path, 5)

        result = await cli.run(
            f"""
            aiperf profile \
                --model test-reranker \
                --url {aiperf_mock_server.url} \
                --tokenizer gpt2 \
                --endpoint-type {endpoint_type} \
                --input-file {dataset_path} \
                --custom-dataset-type single_turn \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        assert result.request_count == defaults.request_count

    async def test_rankings_with_synthetic_data(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Test ranking endpoint with synthetic data generation parameters."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model test-reranker \
                --url {aiperf_mock_server.url} \
                --tokenizer gpt2 \
                --endpoint-type nim_rankings \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --rankings-passages-mean 6 \
                --rankings-passages-stddev 2 \
                --rankings-passages-prompt-token-mean 32 \
                --rankings-passages-prompt-token-stddev 8 \
                --rankings-query-prompt-token-mean 16 \
                --rankings-query-prompt-token-stddev 4 \
                --ui {defaults.ui}
            """
        )

        assert result.request_count == defaults.request_count
