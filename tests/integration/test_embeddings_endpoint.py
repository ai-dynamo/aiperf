# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for embedding endpoints."""

import pytest

from tests.harness.utils import AIPerfCLI, AIPerfMockServer
from tests.integration.conftest import IntegrationTestDefaults as defaults


@pytest.mark.integration
@pytest.mark.asyncio
class TestEmbeddingsEndpoint:
    """Tests for embedding endpoints."""

    @pytest.mark.parametrize("endpoint_type", ["embeddings", "cohere_embeddings"])
    async def test_basic_embeddings(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        endpoint_type: str,
    ):
        """Basic embeddings request completes with expected request count."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model nomic-ai/nomic-embed-text-v1.5 \
                --tokenizer gpt2 \
                --url {aiperf_mock_server.url} \
                --endpoint-type {endpoint_type} \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.request_count == defaults.request_count
        # Embeddings are non-streaming, so streaming metrics should not be present
        assert result.json.time_to_first_token is None
