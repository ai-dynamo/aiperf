# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for /v1/completions endpoint."""

import pytest

from tests.integration.conftest import AIPerfCLI
from tests.integration.models import AIPerfMockServer


@pytest.mark.integration
@pytest.mark.asyncio
class TestCompletionsEndpoint:
    """Tests for /v1/completions endpoint."""

    async def test_basic_completions(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Basic non-streaming completions."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model Qwen/Qwen3-0.6B \
                --url {aiperf_mock_server.url} \
                --endpoint-type completions \
                --request-count 10 \
                --concurrency 2 \
                --ui simple
            """
        )
        assert result.request_count == 10
