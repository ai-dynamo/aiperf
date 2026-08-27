# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for multimodal inputs (images, audio)."""

import pytest

from tests.harness.utils import AIPerfCLI, AIPerfMockServer
from tests.integration.conftest import IntegrationTestDefaults as defaults


@pytest.mark.integration
@pytest.mark.asyncio
class TestMultimodal:
    """Tests for multimodal inputs (images, audio)."""

    async def test_images(self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer):
        """Chat with image inputs."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --image-width-mean 64 \
                --image-height-mean 64 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.request_count == defaults.request_count
        assert result.has_input_images

    async def test_audio(self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer):
        """Chat with audio inputs."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --audio-length-mean 0.1 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.request_count == defaults.request_count
        assert result.has_input_audio

    async def test_images_and_audio(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Chat with combined image and audio inputs."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --image-width-mean 64 \
                --image-height-mean 64 \
                --audio-length-mean 0.1 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.request_count == defaults.request_count
        assert result.has_input_images
        assert result.has_input_audio
