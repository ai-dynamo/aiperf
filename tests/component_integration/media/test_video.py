# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for video inputs with different synthesis types and audio embedding."""

import shutil

import pytest
from pytest import approx

from tests.component_integration.conftest import (
    ComponentIntegrationTestDefaults as defaults,
)
from tests.harness.utils import AIPerfCLI
from tests.integration.utils import first_video_details

FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None


@pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg not installed")
@pytest.mark.ffmpeg
@pytest.mark.component_integration
class TestVideoSynthesisTypes:
    """Tests that different video synthesis types generate videos with correct parameters."""

    @pytest.mark.slow
    def test_moving_shapes_synthesis(self, cli: AIPerfCLI):
        """Verify moving_shapes synthesis generates video with specified dimensions and timing."""
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --endpoint-type chat \
                --video-width 512 \
                --video-height 288 \
                --video-duration 3.0 \
                --video-fps 4 \
                --video-synth-type moving_shapes \
                --prompt-input-tokens-mean 50 \
                --num-dataset-entries 4 \
                --request-rate 50.0 \
                --request-count 4 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.request_count == 4
        assert result.has_input_videos

        details = first_video_details(result)
        assert details is not None, "No video found in payload"
        assert details.width == 512
        assert details.height == 288
        assert details.fps == approx(4.0)
        assert details.duration == approx(3.0)

    @pytest.mark.slow
    def test_grid_clock_synthesis(self, cli: AIPerfCLI):
        """Verify grid_clock synthesis generates video with specified dimensions and timing."""
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --endpoint-type chat \
                --video-width 640 \
                --video-height 360 \
                --video-duration 2.0 \
                --video-fps 6 \
                --video-synth-type grid_clock \
                --prompt-input-tokens-mean 50 \
                --num-dataset-entries 4 \
                --request-rate 20.0 \
                --request-count 4 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.request_count == 4
        assert result.has_input_videos

        details = first_video_details(result)
        assert details is not None, "No video found in payload"
        assert details.width == 640
        assert details.height == 360
        assert details.fps == approx(6.0)
        assert details.duration == approx(2.0)


@pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg not installed")
@pytest.mark.ffmpeg
@pytest.mark.component_integration
class TestVideoAudio:
    """Tests that video audio embedding produces correct audio streams."""

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "video_format,video_codec,expected_audio_codec",
        [
            ("webm", "libvpx-vp9", "vorbis"),
            ("mp4", "libx264", "aac"),
        ],
    )
    def test_video_audio_auto_codec(
        self,
        cli: AIPerfCLI,
        video_format: str,
        video_codec: str,
        expected_audio_codec: str,
    ):
        """Verify auto-selected audio codec matches video format."""
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --endpoint-type chat \
                --video-width 320 \
                --video-height 240 \
                --video-duration 2.0 \
                --video-fps 4 \
                --video-format {video_format} \
                --video-codec {video_codec} \
                --video-audio-sample-rate 44100 \
                --video-audio-num-channels 1 \
                --prompt-input-tokens-mean 50 \
                --num-dataset-entries 4 \
                --request-rate 50.0 \
                --request-count 4 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.request_count == 4
        assert result.has_input_videos

        details = first_video_details(result)
        assert details is not None, "No video found in payload"
        assert details.has_audio, f"Expected audio in {video_format} video"
        assert details.audio_codec == expected_audio_codec
        assert details.audio_channels == 1
        assert details.audio_sample_rate == 44100

    @pytest.mark.slow
    def test_video_audio_stereo(self, cli: AIPerfCLI):
        """Verify stereo audio produces 2-channel audio stream."""
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --endpoint-type chat \
                --video-width 320 \
                --video-height 240 \
                --video-duration 2.0 \
                --video-fps 4 \
                --video-format webm \
                --video-codec libvpx-vp9 \
                --video-audio-num-channels 2 \
                --prompt-input-tokens-mean 50 \
                --num-dataset-entries 4 \
                --request-rate 50.0 \
                --request-count 4 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.request_count == 4

        details = first_video_details(result)
        assert details is not None, "No video found in payload"
        assert details.has_audio
        assert details.audio_channels == 2

    @pytest.mark.slow
    def test_video_without_audio_backward_compat(self, cli: AIPerfCLI):
        """Verify videos without audio enabled have no audio stream."""
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --endpoint-type chat \
                --video-width 320 \
                --video-height 240 \
                --video-duration 2.0 \
                --video-fps 4 \
                --video-format webm \
                --video-codec libvpx-vp9 \
                --prompt-input-tokens-mean 50 \
                --num-dataset-entries 4 \
                --request-rate 50.0 \
                --request-count 4 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.request_count == 4

        details = first_video_details(result)
        assert details is not None, "No video found in payload"
        assert not details.has_audio, "Video should not have audio when disabled"
