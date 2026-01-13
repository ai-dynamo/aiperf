# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for video inputs."""

import pytest
from pytest import approx

from tests.integration.conftest import AIPerfCLI
from tests.integration.conftest import IntegrationTestDefaults as defaults
from tests.integration.models import AIPerfMockServer
from tests.integration.utils import extract_base64_video_details


@pytest.mark.ffmpeg
@pytest.mark.integration
@pytest.mark.asyncio
class TestVideo:
    """Tests for video inputs."""

    async def test_video_moving_shapes(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Video generation with parameter validation."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --video-width 512 \
                --video-height 288 \
                --video-duration 5.0 \
                --video-fps 4 \
                --video-synth-type moving_shapes \
                --prompt-input-tokens-mean 50 \
                --num-dataset-entries 4 \
                --request-rate 2.0 \
                --request-count 4 \
                --video-format webm \
                --video-codec libvpx-vp9 \
                --workers-max {defaults.workers_max}
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

    async def test_video_grid_clock(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Video generation with parameter validation."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --video-width 640 \
                --video-height 360 \
                --video-duration 10.0 \
                --video-fps 6 \
                --video-synth-type grid_clock \
                --prompt-input-tokens-mean 50 \
                --num-dataset-entries 4 \
                --request-rate 2.0 \
                --request-count 4 \
                --video-format mp4 \
                --video-codec libx264 \
                --workers-max {defaults.workers_max}
            """
        )
        assert result.request_count == 4
        assert result.has_input_videos

        for payload in result.inputs.data:
            for payload_item in payload.payloads:
                for message in payload_item.get("messages", []):
                    content = message.get("content", [])
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and "video_url" in item:
                                video_data = item["video_url"]["url"].split(",")[1]
                                details = extract_base64_video_details(video_data)
                                assert details.width == 640
                                assert details.height == 360
                                assert details.fps == approx(6.0)
                                assert details.duration == approx(10.0)
                                # Verify MP4 is not fragmented (uses faststart)
                                assert not details.is_fragmented, (
                                    "MP4 should not be fragmented"
                                )
