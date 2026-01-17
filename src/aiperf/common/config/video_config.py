# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated

from pydantic import Field, model_validator
from typing_extensions import Self

from aiperf.common.config.base_config import BaseConfig
from aiperf.common.config.cli_parameter import CLIParameter
from aiperf.common.config.config_defaults import VideoDefaults
from aiperf.common.config.groups import Groups
from aiperf.common.enums import VideoFormat, VideoSynthType


class VideoConfig(BaseConfig):
    """
    A configuration class for defining video related settings.

    Note: Video generation requires FFmpeg to be installed on your system.
    If FFmpeg is not found, you'll get installation instructions specific to your platform.
    """

    @model_validator(mode="after")
    def validate_width_and_height(self) -> Self:
        if self.width and not self.height:
            raise ValueError("Width is specified but height is not")
        if self.height and not self.width:
            raise ValueError("Height is specified but width is not")
        return self

    _CLI_GROUP = Groups.VIDEO_INPUT

    batch_size: Annotated[
        int,
        Field(
            ge=0,
            description="Number of video files to include in each multimodal request. Supported with `chat` endpoint type for video understanding models. "
            "Each video is generated synthetically with specified duration, FPS, resolution, and codec. Set to 0 to disable video inputs. "
            "Higher batch sizes test multi-video understanding and significantly increase request payload size.",
        ),
        CLIParameter(
            name=(
                "--video-batch-size",
                "--batch-size-video",
            ),
            group=_CLI_GROUP,
        ),
    ] = VideoDefaults.BATCH_SIZE

    duration: Annotated[
        float,
        Field(
            ge=0.0,
            description="Duration in seconds for each synthetically generated video clip. Combined with `--video-fps`, determines total frame count "
            "(frames = duration × FPS). Longer durations increase file size and processing time. Typical values: 1-10 seconds for testing. "
            "Requires FFmpeg for video generation.",
        ),
        CLIParameter(
            name=("--video-duration",),
            group=_CLI_GROUP,
        ),
    ] = VideoDefaults.DURATION

    fps: Annotated[
        int,
        Field(
            ge=1,
            description="Frames per second for generated video. Higher FPS creates smoother video but increases frame count and file size. "
            "Common values: `4` (minimal motion, recommended for Cosmos models), `24` (cinematic), `30` (standard video), `60` (high frame rate). "
            "Total frames = `--video-duration` × FPS.",
        ),
        CLIParameter(
            name=("--video-fps",),
            group=_CLI_GROUP,
        ),
    ] = VideoDefaults.FPS

    width: Annotated[
        int | None,
        Field(
            ge=1,
            description="Video frame width in pixels. Must be specified together with `--video-height` (both or neither). Determines video resolution "
            "and file size. Common resolutions: `640×480` (SD), `1280×720` (HD), `1920×1080` (Full HD). If not specified, uses codec/format defaults.",
        ),
        CLIParameter(
            name=("--video-width",),
            group=_CLI_GROUP,
        ),
    ] = VideoDefaults.WIDTH

    height: Annotated[
        int | None,
        Field(
            ge=1,
            description="Video frame height in pixels. Must be specified together with `--video-width` (both or neither). Combined with width "
            "determines aspect ratio and total pixel count per frame. Higher resolution increases processing demands and file size.",
        ),
        CLIParameter(
            name=("--video-height",),
            group=_CLI_GROUP,
        ),
    ] = VideoDefaults.HEIGHT

    synth_type: Annotated[
        VideoSynthType,
        Field(
            description="Algorithm for generating synthetic video content. Different types produce different visual patterns for testing. "
            "Options vary by implementation (e.g., `noise`, `gradient`, `checkerboard`). Content doesn't affect semantic meaning but may "
            "impact encoding efficiency and file size.",
        ),
        CLIParameter(
            name=("--video-synth-type",),
            group=_CLI_GROUP,
        ),
    ] = VideoDefaults.SYNTH_TYPE

    format: Annotated[
        VideoFormat,
        Field(
            description="Container format for generated video files. Supports `webm` (VP9, recommended, BSD-licensed), `mp4` (H.264/H.265, widely compatible), "
            "`avi` (legacy, larger files), `mkv` (Matroska, flexible). Format choice affects compatibility, file size, and encoding options. "
            "Use `webm` for open-source workflows, `mp4` for maximum compatibility.",
        ),
        CLIParameter(
            name=("--video-format",),
            group=_CLI_GROUP,
        ),
    ] = VideoDefaults.FORMAT

    codec: Annotated[
        str,
        Field(
            description=(
                "The video codec to use for encoding. Common options: "
                "libvpx-vp9 (CPU, BSD-licensed, default for WebM), "
                "libx264 (CPU, GPL-licensed, widely compatible), "
                "libx265 (CPU, GPL-licensed, smaller files), "
                "h264_nvenc (NVIDIA GPU), hevc_nvenc (NVIDIA GPU, smaller files). "
                "Any FFmpeg-supported codec can be used."
            ),
        ),
        CLIParameter(
            name=("--video-codec",),
            group=_CLI_GROUP,
        ),
    ] = VideoDefaults.CODEC
