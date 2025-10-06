# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic data format tests (JPEG, MP3, etc)."""

import pytest

from tests.integration.helpers import AIPerfCLI, FakeAIServer


@pytest.mark.integration
@pytest.mark.asyncio
class TestImageFormats:
    """Tests for different image format support."""

    async def test_jpeg_format(self, cli: AIPerfCLI, fakeai_server: FakeAIServer):
        """JPEG image format support."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model openai/gpt-oss-20b \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --request-count 5 \
                --concurrency 1 \
                --image-width-mean 128 \
                --image-height-mean 128 \
                --image-format jpeg \
                --ui simple
            """
        )
        assert result.request_count == 5
        assert result.has_input_images

    async def test_png_format(self, cli: AIPerfCLI, fakeai_server: FakeAIServer):
        """PNG image format support."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model openai/gpt-oss-20b \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --request-count 5 \
                --concurrency 1 \
                --image-width-mean 128 \
                --image-height-mean 128 \
                --image-format png \
                --ui simple
            """
        )
        assert result.request_count == 5
        assert result.has_input_images


@pytest.mark.integration
@pytest.mark.asyncio
class TestAudioFormats:
    """Tests for different audio format support."""

    async def test_mp3_format(self, cli: AIPerfCLI, fakeai_server: FakeAIServer):
        """MP3 audio format support."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model openai/gpt-oss-20b \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --request-count 5 \
                --concurrency 1 \
                --audio-length-mean 0.1 \
                --audio-format mp3 \
                --ui simple
            """
        )
        assert result.request_count == 5
        assert result.has_input_audio

    async def test_wav_format(self, cli: AIPerfCLI, fakeai_server: FakeAIServer):
        """WAV audio format support."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model openai/gpt-oss-20b \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --request-count 5 \
                --concurrency 1 \
                --audio-length-mean 0.1 \
                --audio-format wav \
                --ui simple
            """
        )
        assert result.request_count == 5
        assert result.has_input_audio
