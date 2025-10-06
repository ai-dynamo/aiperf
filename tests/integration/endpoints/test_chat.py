# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from tests.integration.helpers import AIPerfCLI, FakeAIServer


@pytest.mark.integration
@pytest.mark.asyncio
class TestChatEndpoint:
    """Tests for /v1/chat/completions endpoint."""

    async def test_basic_chat(self, cli: AIPerfCLI, fakeai_server: FakeAIServer):
        """Basic non-streaming chat."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model openai/gpt-oss-20b \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --request-count 10 \
                --concurrency 2 \
                --ui simple
            """
        )
        assert result.request_count == 10

    async def test_streaming(self, cli: AIPerfCLI, fakeai_server: FakeAIServer):
        """Streaming chat."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model openai/gpt-oss-20b \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --streaming \
                --request-count 10 \
                --concurrency 2 \
                --ui simple
            """
        )
        assert result.request_count == 10
        assert result.has_streaming_metrics

    async def test_high_concurrency_streaming(
        self, cli: AIPerfCLI, fakeai_server: FakeAIServer
    ):
        """High concurrency streaming."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model Qwen/Qwen3-0.6B \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --concurrency 100 \
                --request-count 100 \
                --streaming \
                --ui simple
            """
        )
        assert result.request_count == 100
        assert result.has_streaming_metrics


@pytest.mark.integration
@pytest.mark.asyncio
class TestChatEndpointMultimodal:
    """Tests for /v1/chat/completions endpoint with multimodal inputs."""

    async def test_with_images(self, cli: AIPerfCLI, fakeai_server: FakeAIServer):
        """Chat with image inputs."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model openai/gpt-oss-20b \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --request-count 10 \
                --concurrency 2 \
                --image-width-mean 64 \
                --image-height-mean 64 \
                --ui simple
            """
        )
        assert result.request_count == 10
        assert result.has_input_images

    async def test_with_audio(self, cli: AIPerfCLI, fakeai_server: FakeAIServer):
        """Chat with audio inputs."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model openai/gpt-oss-20b \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --request-count 10 \
                --concurrency 2 \
                --audio-length-mean 0.1 \
                --ui simple
            """
        )
        assert result.request_count == 10
        assert result.has_input_audio

    async def test_multimodal(self, cli: AIPerfCLI, fakeai_server: FakeAIServer):
        """Chat with images and audio."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model openai/gpt-oss-20b \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --request-count 10 \
                --concurrency 2 \
                --image-width-mean 64 \
                --image-height-mean 64 \
                --audio-length-mean 0.1 \
                --ui simple
            """
        )
        assert result.request_count == 10
        assert result.has_input_images
        assert result.has_input_audio

    async def test_streaming_multimodal(
        self, cli: AIPerfCLI, fakeai_server: FakeAIServer
    ):
        """Streaming with images."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model openai/gpt-oss-20b \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --streaming \
                --request-count 10 \
                --concurrency 2 \
                --image-width-mean 64 \
                --image-height-mean 64 \
                --ui simple
            """
        )
        assert result.request_count == 10
        assert result.has_streaming_metrics

    async def test_high_concurrency_multimodal(
        self, cli: AIPerfCLI, fakeai_server: FakeAIServer
    ):
        """High concurrency with multimodal."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model openai/gpt-oss-20b \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --streaming \
                --request-count 1000 \
                --concurrency 1000 \
                --image-width-mean 64 \
                --image-height-mean 64 \
                --ui simple
            """,
            timeout=180.0,
        )
        assert result.request_count == 1000
        assert result.has_streaming_metrics

    async def test_request_cancellation(
        self, cli: AIPerfCLI, fakeai_server: FakeAIServer
    ):
        """Request cancellation doesn't break pipeline."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model openai/gpt-oss-20b \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --streaming \
                --request-count 50 \
                --concurrency 5 \
                --image-width-mean 64 \
                --image-height-mean 64 \
                --request-cancellation-rate 0.3 \
                --request-cancellation-delay 0.5 \
                --ui simple
            """,
            timeout=120.0,
        )
        # With 30% cancellation rate, we should get ~35 completed requests
        assert result.request_count >= 30
