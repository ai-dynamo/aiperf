# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AIPerf Configuration v2.0 - Pydantic Models

Synthetic video and embedded-audio-track configuration.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Self

from pydantic import (
    BeforeValidator,
    ConfigDict,
    Field,
    model_validator,
)

from aiperf.common.enums import (
    VideoAudioCodec,
    VideoFormat,
    VideoSynthType,
)
from aiperf.config.base import BaseConfig

VIDEO_AUDIO_CODEC_MAP: dict[VideoFormat, VideoAudioCodec] = {
    VideoFormat.WEBM: VideoAudioCodec.LIBVORBIS,
    VideoFormat.MP4: VideoAudioCodec.AAC,
}


def _coerce_int_literal(value: Any) -> Any:
    """Coerce numeric strings (e.g., from YAML/JSON configs) to int for Literal validation."""
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lstrip("-").isdigit():
            return int(stripped)
    return value


class VideoAudioConfig(BaseConfig):
    """Configuration for embedding an audio track in synthetic video files."""

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        """Reject audio codec settings when embedded audio is disabled.

        ``channels=0`` means no audio track will be muxed. If ``codec`` is set,
        ``channels`` must be 1 or 2 so the user's codec selection is not silently
        ignored.
        """
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
            default=44.1,
            description="Audio sample rate in kHz for the embedded audio track. "
            "Common values: 8 (telephony), 16 (speech), 44.1 (CD quality), 48 (professional). "
            "Higher sample rates increase audio fidelity and file size.",
        ),
    ]

    channels: Annotated[
        int,
        Field(
            ge=0,
            le=2,
            default=0,
            description="Number of audio channels to embed in generated video files. "
            "0 = disabled (no audio track, default), 1 = mono, 2 = stereo. "
            "When set to 1 or 2, a Gaussian noise audio track matching the video duration "
            "is muxed into each video via FFmpeg.",
        ),
    ]

    codec: Annotated[
        VideoAudioCodec | None,
        Field(
            default=None,
            description="Audio codec for the embedded audio track. "
            "If not specified, auto-selects based on video format: "
            "aac for MP4, libvorbis for WebM. "
            "Options: aac, libvorbis, libopus.",
        ),
    ]

    depth: Annotated[
        Literal[8, 16, 24, 32],
        Field(
            default=16,
            description="Audio bit depth for the embedded audio track. "
            "Supported values: 8, 16, 24, or 32 bits. "
            "Higher bit depths provide greater dynamic range but increase file size.",
        ),
        BeforeValidator(_coerce_int_literal),
    ]


class VideoConfig(BaseConfig):
    """
    Configuration for synthetic video generation in multimodal datasets.

    Controls the generation of synthetic videos for video understanding
    model benchmarking. Requires FFmpeg for video generation.
    """

    model_config = ConfigDict(extra="forbid")

    batch_size: Annotated[
        int,
        Field(
            ge=0,
            default=0,
            description="Number of video files to include in each multimodal request. "
            "Supported with chat endpoint type for video understanding models. "
            "Set to 0 to disable video inputs. "
            "Higher batch sizes significantly increase request payload size.",
        ),
    ]

    duration: Annotated[
        float,
        Field(
            gt=0.0,
            default=1.0,
            description="Duration in seconds for each generated video clip. "
            "Combined with fps, determines total frame count (frames = duration * fps). "
            "Longer durations increase file size and processing time. "
            "Typical values: 1-10 seconds for testing.",
        ),
    ]

    fps: Annotated[
        int,
        Field(
            ge=1,
            default=4,
            description="Frames per second for generated video. "
            "Higher FPS creates smoother video but increases frame count and file size. "
            "Common values: 4 (minimal, recommended for Cosmos models), "
            "24 (cinematic), 30 (standard), 60 (high frame rate). "
            "Total frames = duration * fps.",
        ),
    ]

    width: Annotated[
        int | None,
        Field(
            ge=1,
            default=None,
            description="Video frame width in pixels. "
            "Determines video resolution and file size. "
            "Common values: 640 (SD), 1280 (HD), 1920 (Full HD). "
            "If not specified, uses codec/format defaults.",
        ),
    ]

    height: Annotated[
        int | None,
        Field(
            ge=1,
            default=None,
            description="Video frame height in pixels. "
            "Combined with width determines aspect ratio and total pixel count per frame. "
            "Common values: 480 (SD), 720 (HD), 1080 (Full HD). "
            "If not specified, uses codec/format defaults.",
        ),
    ]

    format: Annotated[
        VideoFormat,
        Field(
            default=VideoFormat.WEBM,
            description="Container format for generated video files. "
            "webm: VP9 codec, BSD-licensed, recommended for open-source workflows. "
            "mp4: H.264/H.265, widely compatible. "
            "Format affects compatibility, file size, and encoding options.",
        ),
    ]

    codec: Annotated[
        str,
        Field(
            default="libvpx-vp9",
            description="Video codec for encoding. "
            "Common options: libvpx-vp9 (CPU, BSD-licensed, default for WebM), "
            "libx264 (CPU, GPL, widely compatible), libx265 (CPU, GPL, smaller files), "
            "h264_nvenc (NVIDIA GPU), hevc_nvenc (NVIDIA GPU, smaller files). "
            "Any FFmpeg-supported codec can be used.",
        ),
    ]

    synth_type: Annotated[
        VideoSynthType,
        Field(
            default=VideoSynthType.MOVING_SHAPES,
            description="Algorithm for generating synthetic video content. "
            "Different types produce different visual patterns for testing. "
            "Content doesn't affect semantic meaning but may impact encoding "
            "efficiency and file size.",
        ),
    ]

    audio: Annotated[
        VideoAudioConfig,
        Field(
            default_factory=VideoAudioConfig,
            description="Audio track configuration for embedding audio in generated videos.",
        ),
    ]
