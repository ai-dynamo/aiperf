# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for streaming responses across endpoints."""

import pytest

from tests.integration.conftest import AIPerfCLI
from tests.integration.models import FakeAIServer


@pytest.mark.integration
@pytest.mark.asyncio
class TestStreaming:
    """Tests for streaming responses across endpoints."""

    async def test_streaming_chat(self, cli: AIPerfCLI, fakeai_server: FakeAIServer):
        """Streaming chat completion with metrics validation."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model Qwen/Qwen2.5-32B-Instruct \
                --url {fakeai_server.url} \
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

    async def test_streaming_completions(
        self, cli: AIPerfCLI, fakeai_server: FakeAIServer
    ):
        """Streaming completions with metrics validation."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
                --url {fakeai_server.url} \
                --endpoint-type completions \
                --streaming \
                --request-count 10 \
                --concurrency 2 \
                --workers-max 1 \
                --ui simple
            """
        )
        assert result.request_count == 10
        assert result.has_streaming_metrics
