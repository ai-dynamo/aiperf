# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for high concurrency and performance scenarios."""

import pytest

from tests.integration.conftest import AIPerfCLI
from tests.integration.models import FakeAIServer


@pytest.mark.integration
@pytest.mark.asyncio
class TestPerformance:
    """Tests for high concurrency and performance scenarios."""

    async def test_high_concurrency_streaming(
        self, cli: AIPerfCLI, fakeai_server: FakeAIServer
    ):
        """High concurrency streaming (100 concurrent requests)."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model Qwen/Qwen3-0.6B \
                --url {fakeai_server.url} \
                --gpu-telemetry {fakeai_server.dcgm_url} \
                --endpoint-type chat \
                --concurrency 100 \
                --request-count 100 \
                --streaming \
                --workers-max 5 \
                --ui simple
            """
        )
        assert result.request_count == 100
        assert result.has_streaming_metrics

    @pytest.mark.performance
    async def test_high_concurrency_multimodal(
        self, cli: AIPerfCLI, fakeai_server: FakeAIServer
    ):
        """Extreme concurrency (1000) with streaming and multimodal inputs."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
                --url {fakeai_server.url} \
                --gpu-telemetry {fakeai_server.dcgm_url} \
                --endpoint-type chat \
                --streaming \
                --request-count 1000 \
                --concurrency 1000 \
                --image-width-mean 64 \
                --image-height-mean 64 \
                --workers-max 5 \
                --ui simple
            """,
            timeout=180.0,
        )
        assert result.request_count == 1000
        assert result.has_streaming_metrics

    async def test_high_concurrency_embeddings(
        self, cli: AIPerfCLI, fakeai_server: FakeAIServer
    ):
        """High concurrency embeddings (50 concurrent requests)."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model nomic-ai/nomic-embed-text-v1.5 \
                --tokenizer gpt2 \
                --url {fakeai_server.url} \
                --endpoint-type embeddings \
                --concurrency 50 \
                --request-count 200 \
                --workers-max 5 \
                --ui simple
            """
        )
        assert result.request_count == 200
