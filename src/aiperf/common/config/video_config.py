# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated, Any, Literal

from pydantic import BeforeValidator, Field, model_validator
from typing_extensions import Self

from aiperf.common.config.base_config import BaseConfig
from aiperf.common.config.cli_parameter import CLIParameter
from aiperf.common.config.config_defaults import VideoAudioDefaults, VideoDefaults
from aiperf.common.config.groups import Groups
from aiperf.common.enums import VideoAudioCodec, VideoFormat, VideoSynthType


def _coerce_int_literal(value: Any) -> Any:
    """Coerce numeric strings (e.g., from YAML/JSON configs) to int for Literal validation."""
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lstrip("-").isdigit():
            return int(stripped)
    return value


VIDEO_AUDIO_CODEC_MAP: dict[VideoFormat, VideoAudioCodec] = {
    VideoFormat.WEBM: VideoAudioCodec.LIBVORBIS,
    VideoFormat.MP4: VideoAudioCodec.AAC,
}


class VideoAudioConfig(BaseConfig):
    """Configuration for embedding an audio track in synthetic video files."""

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        if self.codec is not None and self.channels == 0:
            raise ValueError(
                f"--video-audio-codec '{self.codec}' is set but --video-audio-num-channels is 0 "
                f"(audio disabled). Set --video-audio-num-channels to 1 or 2 to enable audio."
            )
        return self

    sample_rate: Annotated[
        float,
        Field(
            ge=8.0,
            le=96.0,
            description="Audio sample rate in kHz for the embedded audio track. "
            "Common values: 8 (telephony), 16 (speech), 44.1 (CD quality), 48 (professional). "
            "Higher sample rates increase audio fidelity and file size.",
        ),
        CLIParameter(
            name=("--video-audio-sample-rate",),
            group=Groups.VIDEO_INPUT,
        ),
    ] = VideoAudioDefaults.SAMPLE_RATE

    channels: Annotated[
        int,
        Field(
            ge=0,
            le=2,
            description="Number of audio channels to embed in generated video files. "
            "0 = disabled (no audio track, default), 1 = mono, 2 = stereo. "
            "When set to 1 or 2, a Gaussian noise audio track matching the video duration "
            "is muxed into each video via FFmpeg.",
        ),
        CLIParameter(
            name=("--video-audio-num-channels",),
            group=Groups.VIDEO_INPUT,
        ),
    ] = VideoAudioDefaults.CHANNELS

    codec: Annotated[
        VideoAudioCodec | None,
        Field(
            description="Audio codec for the embedded audio track. "
            "If not specified, auto-selects based on video format: "
            "aac for MP4, libvorbis for WebM. "
            "Options: aac, libvorbis, libopus.",
        ),
        CLIParameter(
            name=("--video-audio-codec",),
            group=Groups.VIDEO_INPUT,
        ),
    ] = VideoAudioDefaults.CODEC

    depth: Annotated[
        Literal[8, 16, 24, 32],
        Field(
            description="Audio bit depth for the embedded audio track. "
            "Supported values: 8, 16, 24, or 32 bits. "
            "Higher bit depths provide greater dynamic range but increase file size.",
        ),
        BeforeValidator(_coerce_int_literal),
        CLIParameter(
            name=("--video-audio-depth",),
            group=Groups.VIDEO_INPUT,
        ),
    ] = VideoAudioDefaults.DEPTH


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
            group=Groups.VIDEO_INPUT,
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
            group=Groups.VIDEO_INPUT,
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
            group=Groups.VIDEO_INPUT,
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
            group=Groups.VIDEO_INPUT,
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
            group=Groups.VIDEO_INPUT,
        ),
    ] = VideoDefaults.HEIGHT

    synth_type: Annotated[
        VideoSynthType,
        Field(
            description="Algorithm for generating synthetic video content. Different types produce different visual patterns for testing. "
            "Options: `moving_shapes` (animated geometric shapes), `grid_clock` (grid with rotating clock hands), `noise` (random pixel frames). "
            "Content doesn't affect semantic meaning but may impact encoding efficiency and file size.",
        ),
        CLIParameter(
            name=("--video-synth-type",),
            group=Groups.VIDEO_INPUT,
        ),
    ] = VideoDefaults.SYNTH_TYPE

    format: Annotated[
        VideoFormat,
        Field(
            description="Container format for generated video files. Supports `webm` (VP9, recommended, BSD-licensed) and `mp4` (H.264/H.265, widely compatible). "
            "Format choice affects compatibility, file size, and encoding options. "
            "Use `webm` for open-source workflows, `mp4` for maximum compatibility.",
        ),
        CLIParameter(
            name=("--video-format",),
            group=Groups.VIDEO_INPUT,
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
            group=Groups.VIDEO_INPUT,
        ),
    ] = VideoDefaults.CODEC

    audio: Annotated[
        VideoAudioConfig,
        Field(
            description="Audio track configuration for embedding audio in generated videos."
        ),
    ] = VideoAudioConfig()
