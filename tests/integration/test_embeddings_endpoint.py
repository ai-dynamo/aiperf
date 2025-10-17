# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for /v1/embeddings endpoint."""

import pytest

from tests.integration.conftest import AIPerfCLI
from tests.integration.models import AIPerfMockServer


@pytest.mark.integration
@pytest.mark.asyncio
class TestEmbeddingsEndpoint:
    """Tests for /v1/embeddings endpoint."""

    async def test_basic_embeddings(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Basic embeddings request."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model nomic-ai/nomic-embed-text-v1.5 \
                --tokenizer gpt2 \
                --url {aiperf_mock_server.url} \
                --endpoint-type embeddings \
                --request-count 10 \
                --concurrency 2 \
                --ui simple
            """
        )
        assert result.request_count == 10
        assert (
            not hasattr(result.json, "time_to_first_token")
            or result.json.time_to_first_token is None
        )
