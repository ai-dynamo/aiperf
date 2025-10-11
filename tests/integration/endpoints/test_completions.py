# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from tests.integration.helpers import AIPerfCLI, FakeAIServer


@pytest.mark.integration
@pytest.mark.asyncio
class TestCompletionsEndpoint:
    """Tests for /v1/completions endpoint."""

    async def test_basic_completions(self, cli: AIPerfCLI, fakeai_server: FakeAIServer):
        """Basic non-streaming completions."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model openai/gpt-oss-20b \
                --url {fakeai_server.url} \
                --endpoint-type completions \
                --request-count 10 \
                --concurrency 2 \
                --ui simple
            """
        )
        assert result.request_count == 10

    async def test_streaming(self, cli: AIPerfCLI, fakeai_server: FakeAIServer):
        """Streaming completions."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model openai/gpt-oss-20b \
                --url {fakeai_server.url} \
                --endpoint-type completions \
                --streaming \
                --request-count 10 \
                --concurrency 2 \
                --ui simple
            """
        )
        assert result.request_count == 10
        assert result.has_streaming_metrics
