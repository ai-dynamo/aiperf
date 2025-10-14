# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for multimodal inputs (images, audio, video)."""

import pytest
from pytest import approx

from tests.integration.conftest import AIPerfCLI
from tests.integration.models import FakeAIServer
from tests.integration.utils import extract_base64_video_details


@pytest.mark.integration
@pytest.mark.asyncio
class TestMultimodal:
    """Tests for multimodal inputs (images, audio, video)."""

    async def test_images(self, cli: AIPerfCLI, fakeai_server: FakeAIServer):
        """Chat with image inputs."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model openai/gpt-oss-120b \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --request-count 10 \
                --concurrency 2 \
                --image-width-mean 64 \
                --image-height-mean 64 \
                --workers-max 1 \
                --ui simple
            """
        )
        assert result.request_count == 10
        assert result.has_input_images

    async def test_audio(self, cli: AIPerfCLI, fakeai_server: FakeAIServer):
        """Chat with audio inputs."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model openai/gpt-oss-120b \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --request-count 10 \
                --concurrency 2 \
                --audio-length-mean 0.1 \
                --workers-max 1 \
                --ui simple
            """
        )
        assert result.request_count == 10
        assert result.has_input_audio

    async def test_images_and_audio(self, cli: AIPerfCLI, fakeai_server: FakeAIServer):
        """Chat with combined image and audio inputs."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model openai/gpt-oss-120b \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --request-count 10 \
                --concurrency 2 \
                --image-width-mean 64 \
                --image-height-mean 64 \
                --audio-length-mean 0.1 \
                --workers-max 1 \
                --ui simple
            """
        )
        assert result.request_count == 10
        assert result.has_input_images
        assert result.has_input_audio

    @pytest.mark.ffmpeg
    async def test_video(self, cli: AIPerfCLI, fakeai_server: FakeAIServer):
        """Video generation with parameter validation."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model-names nvidia/cosmos-reason1-7b \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --video-width 512 \
                --video-height 288 \
                --video-duration 5.0 \
                --video-fps 4 \
                --video-synth-type moving_shapes \
                --prompt-input-tokens-mean 50 \
                --num-dataset-entries 1 \
                --request-rate 2.0 \
                --request-count 4 \
                --workers-max 1
            """
        )
        assert result.request_count == 4
        assert result.has_input_videos

        payload = result.inputs.data[0].payloads[0]
        for message in payload.get("messages", []):
            content = message.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and "video_url" in item:
                        video_data = item["video_url"]["url"].split(",")[1]
                        details = extract_base64_video_details(video_data)
                        assert details.width == 512
                        assert details.height == 288
                        assert details.fps == approx(4.0)
                        assert details.duration == approx(5.0)
