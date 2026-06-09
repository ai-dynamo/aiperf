# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for /rag/api/prompt endpoint."""

import pytest

from tests.harness.utils import AIPerfCLI, AIPerfMockServer
from tests.integration.conftest import IntegrationTestDefaults as defaults


@pytest.mark.integration
@pytest.mark.asyncio
class TestSolidoRAGEndpoint:
    """Tests for /rag/api/prompt endpoint."""

    async def test_basic_solido_rag(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Basic SOLIDO RAG request."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model rag-model \
                --tokenizer gpt2 \
                --url {aiperf_mock_server.url} \
                --endpoint-type solido_rag \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.request_count == defaults.request_count
