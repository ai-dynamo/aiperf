# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for video inputs."""

import shutil

import pytest
from pytest import approx

from tests.harness.utils import AIPerfCLI, AIPerfMockServer
from tests.integration.conftest import IntegrationTestDefaults as defaults
from tests.integration.utils import extract_base64_video_details

FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None


@pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg not installed")
@pytest.mark.ffmpeg
@pytest.mark.integration
@pytest.mark.asyncio
class TestVideo:
    """Tests for video inputs."""

    @pytest.mark.parametrize(
        "video_format,video_codec,check_fragmentation",
        [
            ("webm", "libvpx-vp9", False),
            ("mp4", "libx264", True),
        ],
    )
    async def test_video_generation_parameters(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        video_format: str,
        video_codec: str,
        check_fragmentation: bool,
    ):
        """Verify video generation respects configured dimensions, fps, and duration."""
        width, height, fps, duration = 512, 288, 4, 5.0

        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --video-width {width} \
                --video-height {height} \
                --video-duration {duration} \
                --video-fps {fps} \
                --video-synth-type moving_shapes \
                --prompt-input-tokens-mean 50 \
                --num-dataset-entries 4 \
                --request-rate 2.0 \
                --request-count 4 \
                --video-format {video_format} \
                --video-codec {video_codec} \
                --workers-max {defaults.workers_max}
            """
        )
        assert result.request_count == 4
        assert result.has_input_videos

        # Verify video parameters in all generated payloads
        video_found = False
        for payload in result.inputs.data:
            for payload_item in payload.payloads:
                for message in payload_item.get("messages", []):
                    content = message.get("content", [])
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and "video_url" in item:
                                video_found = True
                                video_data = item["video_url"]["url"].split(",")[1]
                                details = extract_base64_video_details(video_data)
                                assert details.width == width
                                assert details.height == height
                                assert details.fps == approx(float(fps))
                                assert details.duration == approx(duration)
                                if check_fragmentation:
                                    assert not details.is_fragmented, (
                                        "MP4 should use faststart, not fragmentation"
                                    )
        assert video_found, "No video content found in payloads"
