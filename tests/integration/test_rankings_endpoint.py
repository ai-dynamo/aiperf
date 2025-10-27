# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0


from pathlib import Path

import pytest

from tests.integration.conftest import AIPerfCLI
from tests.integration.conftest import IntegrationTestDefaults as defaults
from tests.integration.models import AIPerfMockServer
from tests.integration.utils import create_rankings_dataset


@pytest.mark.integration
@pytest.mark.asyncio
class TestRankingsEndpoint:
    """Integration tests for all ranking-type endpoints."""

    async def test_nim_rankings(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer, tmp_path: Path
    ):
        """Basic rankings test for NIM Rankings endpoint (/v1/ranking)."""
        dataset_path = create_rankings_dataset(tmp_path, 5)

        result = await cli.run(
            f"""
            aiperf profile \
                --model nvidia/nv-rerank-qa-mistral-4b \
                --url {aiperf_mock_server.url} \
                --tokenizer gpt2 \
                --endpoint-type nim_rankings \
                --input-file {dataset_path} \
                --custom-dataset-type single_turn \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        assert result.request_count == defaults.request_count

    async def test_hf_tei_rankings(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer, tmp_path: Path
    ):
        """Test for HuggingFace TEI Rankings endpoint (/rerank)."""
        dataset_path = create_rankings_dataset(tmp_path, 5)

        result = await cli.run(
            f"""
            aiperf profile \
                --model Cohere/rerank-v3.5 \
                --url {aiperf_mock_server.url} \
                --tokenizer bert-base-uncased \
                --endpoint-type hf_tei_rankings \
                --input-file {dataset_path} \
                --custom-dataset-type single_turn \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        assert result.request_count == defaults.request_count

    async def test_cohere_rankings(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer, tmp_path: Path
    ):
        """Test for Cohere Rankings endpoint (/v2/rerank)."""
        dataset_path = create_rankings_dataset(tmp_path, 5)

        result = await cli.run(
            f"""
            aiperf profile \
                --model rerank-v3.5 \
                --url {aiperf_mock_server.url} \
                --tokenizer bert-base-uncased \
                --endpoint-type cohere_rankings \
                --input-file {dataset_path} \
                --custom-dataset-type single_turn \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        assert result.request_count == defaults.request_count
