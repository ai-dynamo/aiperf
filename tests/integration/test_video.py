# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for video inputs."""

import shutil

import pytest
from pytest import approx

from tests.harness.utils import AIPerfCLI, AIPerfMockServer
from tests.integration.conftest import IntegrationTestDefaults as defaults
from tests.integration.utils import iter_video_details

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
        videos = list(iter_video_details(result))
        assert videos, "No video content found in payloads"
        for details in videos:
            assert details.width == width
            assert details.height == height
            assert details.fps == approx(float(fps))
            assert details.duration == approx(duration)
            if check_fragmentation:
                assert not details.is_fragmented, (
                    "MP4 should use faststart, not fragmentation"
                )

    @pytest.mark.parametrize(
        "video_format,video_codec,expected_audio_codec",
        [
            ("webm", "libvpx-vp9", "vorbis"),
            ("mp4", "libx264", "aac"),
        ],
    )
    async def test_video_with_audio_embeds_correct_stream(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        video_format: str,
        video_codec: str,
        expected_audio_codec: str,
    ):
        """Verify video+audio muxing produces correct audio stream per format."""
        width, height, fps, duration = 320, 240, 4, 2.0

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
                --video-format {video_format} \
                --video-codec {video_codec} \
                --video-audio-sample-rate 44100 \
                --video-audio-num-channels 1 \
                --prompt-input-tokens-mean 50 \
                --num-dataset-entries 4 \
                --request-rate 2.0 \
                --request-count 4 \
                --workers-max {defaults.workers_max}
            """
        )
        assert result.request_count == 4
        assert result.has_input_videos

        videos = list(iter_video_details(result))
        assert videos, "No video content found in payloads"
        for details in videos:
            assert details.width == width
            assert details.height == height
            assert details.fps == approx(float(fps))
            assert details.has_audio, f"Expected audio stream in {video_format} video"
            assert details.audio_codec == expected_audio_codec
            assert details.audio_channels == 1
            assert details.audio_sample_rate == 44100

    async def test_video_without_audio_has_no_audio_stream(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
    ):
        """Verify videos without audio enabled have no audio stream (backward compat)."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --video-width 320 \
                --video-height 240 \
                --video-duration 2.0 \
                --video-fps 4 \
                --video-format webm \
                --video-codec libvpx-vp9 \
                --prompt-input-tokens-mean 50 \
                --num-dataset-entries 4 \
                --request-rate 2.0 \
                --request-count 4 \
                --workers-max {defaults.workers_max}
            """
        )
        assert result.request_count == 4
        assert result.has_input_videos

        videos = list(iter_video_details(result))
        assert videos, "No video content found in payloads"
        for details in videos:
            assert not details.has_audio, "Video should not have audio when disabled"
