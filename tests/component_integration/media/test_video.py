# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for video inputs."""

import shutil

import pytest
from pytest import approx

from tests.component_integration.conftest import (
    ComponentIntegrationTestDefaults as defaults,
)
from tests.harness.utils import AIPerfCLI
from tests.integration.utils import extract_base64_video_details

FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None


@pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg not installed")
@pytest.mark.ffmpeg
@pytest.mark.component_integration
class TestVideo:
    """Tests for video inputs."""

    @pytest.mark.slow
    def test_video_moving_shapes(self, cli: AIPerfCLI):
        """Video generation with parameter validation."""
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
                        assert details.duration == approx(3.0)

    @pytest.mark.slow
    def test_video_grid_clock(self, cli: AIPerfCLI):
        """Video generation with grid_clock type."""
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

        # Validate video properties (same as test_video_moving_shapes)
        payload = result.inputs.data[0].payloads[0]
        for message in payload.get("messages", []):
            content = message.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and "video_url" in item:
                        video_data = item["video_url"]["url"].split(",")[1]
                        details = extract_base64_video_details(video_data)
                        assert details.width == 640
                        assert details.height == 360
                        assert details.fps == approx(6.0)
                        assert details.duration == approx(2.0)
