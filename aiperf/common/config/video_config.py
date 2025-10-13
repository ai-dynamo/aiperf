# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated

from pydantic import Field

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

    _CLI_GROUP = Groups.VIDEO_INPUT

    batch_size: Annotated[
        int,
        Field(
            ge=0,
            description="The video batch size of the requests AIPerf should send.\n",
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
            description="Seconds per clip (default: 5.0).",
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
            description="Frames per second (default/recommended for Cosmos: 4).",
        ),
        CLIParameter(
            name=("--video-fps",),
            group=_CLI_GROUP,
        ),
    ] = VideoDefaults.FPS

    width: Annotated[
        int,
        Field(
            ge=1,
            description="Video width in pixels.",
        ),
        CLIParameter(
            name=("--video-width",),
            group=_CLI_GROUP,
        ),
    ] = VideoDefaults.WIDTH

    height: Annotated[
        int,
        Field(
            ge=1,
            description="Video height in pixels.",
        ),
        CLIParameter(
            name=("--video-height",),
            group=_CLI_GROUP,
        ),
    ] = VideoDefaults.HEIGHT

    synth_type: Annotated[
        VideoSynthType,
        Field(
            description="Synthetic generator type.",
        ),
        CLIParameter(
            name=("--video-synth-type",),
            group=_CLI_GROUP,
        ),
    ] = VideoDefaults.SYNTH_TYPE

    format: Annotated[
        VideoFormat,
        Field(
            description="The video format of the generated files.",
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
                "libx264 (CPU, widely compatible), libx265 (CPU, smaller files), "
                "h264_nvenc (NVIDIA GPU), hevc_nvenc (NVIDIA GPU, smaller files). "
                "Any FFmpeg-supported codec can be used."
            ),
        ),
        CLIParameter(
            name=("--video-codec",),
            group=_CLI_GROUP,
        ),
    ] = VideoDefaults.CODEC
