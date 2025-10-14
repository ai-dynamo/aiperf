# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for different media format support (JPEG, PNG, MP3, WAV)."""

import pytest

from tests.integration.conftest import AIPerfCLI
from tests.integration.models import FakeAIServer


@pytest.mark.integration
@pytest.mark.asyncio
class TestMediaFormats:
    """Tests for different media format support (JPEG, PNG, MP3, WAV)."""

    @pytest.mark.parametrize("image_format", ["jpeg", "png"])
    async def test_image_formats(
        self, cli: AIPerfCLI, fakeai_server: FakeAIServer, image_format: str
    ):
        """Test different image format support."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model openai/gpt-oss-120b \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --request-count 5 \
                --concurrency 1 \
                --image-width-mean 128 \
                --image-height-mean 128 \
                --image-format {image_format} \
                --workers-max 1 \
                --ui simple
            """
        )
        assert result.request_count == 5
        assert result.has_input_images

    @pytest.mark.parametrize("audio_format", ["mp3", "wav"])
    async def test_audio_formats(
        self, cli: AIPerfCLI, fakeai_server: FakeAIServer, audio_format: str
    ):
        """Test different audio format support."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model openai/gpt-oss-120b \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --request-count 5 \
                --concurrency 1 \
                --audio-length-mean 0.1 \
                --audio-format {audio_format} \
                --workers-max 1 \
                --ui simple
            """
        )
        assert result.request_count == 5
        assert result.has_input_audio
