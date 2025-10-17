# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from tests.integration.conftest import AIPerfCLI
from tests.integration.models import AIPerfMockServer


@pytest.mark.integration
@pytest.mark.asyncio
class TestChatEndpoint:
    """Tests for /v1/chat/completions endpoint."""

    async def test_basic_chat(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Basic non-streaming chat completion."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model microsoft/phi-4 \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --request-count 10 \
                --concurrency 2 \
                --ui simple
            """
        )
        assert result.request_count == 10

    async def test_streaming_chat(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Streaming chat completion with metrics validation."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model Qwen/Qwen2.5-32B-Instruct \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --streaming \
                --request-count 10 \
                --concurrency 2 \
                --workers-max 1 \
                --ui simple
            """
        )
        assert result.request_count == 10
        assert result.has_streaming_metrics
