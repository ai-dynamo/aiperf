# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.base_enums import CaseInsensitiveStrEnum


class AudioFormat(CaseInsensitiveStrEnum):
    """Audio file formats for synthetic audio generation."""

    WAV = "wav"
    """WAV format. Uncompressed audio, larger file sizes, best quality."""

    MP3 = "mp3"
    """MP3 format. Compressed audio, smaller file sizes, good quality."""


class ContentType(CaseInsensitiveStrEnum):
    """Defines the semantic type for synthetic text content."""

    RANDOM_TOKENS = "random_tokens"
    """Generate random token sequences."""

    SYSTEM_PROMPT = "system_prompt"
    """Generate system prompt style content."""

    CONTEXT = "context"
    """Generate contextual information."""

    INSTRUCTION = "instruction"
    """Generate instruction-style content."""

    QUESTION = "question"
    """Generate question-style content."""


class ImageFormat(CaseInsensitiveStrEnum):
    """Image file formats for synthetic image generation."""

    PNG = "png"
    """PNG format. Lossless compression, larger file sizes, best quality."""

    JPEG = "jpeg"
    """JPEG format. Lossy compression, smaller file sizes, good for photos."""

    RANDOM = "random"
    """Randomly select PNG or JPEG for each image."""


class MediaType(CaseInsensitiveStrEnum):
    """The various types of media (e.g. text, image, audio, video)."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


class VideoAudioCodec(CaseInsensitiveStrEnum):
    """Audio codecs for embedding audio in synthetic video files."""

    AAC = "aac"
    """AAC codec. Default for MP4 containers."""

    LIBVORBIS = "libvorbis"
    """Vorbis codec. Default for WebM containers."""

    LIBOPUS = "libopus"
    """Opus codec. Alternative for WebM containers."""


class VideoFormat(CaseInsensitiveStrEnum):
    """Video container formats for synthetic video generation."""

    MP4 = "mp4"
    """MP4 container. Widely compatible, good for H.264/H.265 codecs."""

    WEBM = "webm"
    """WebM container. Open format, optimized for web, good for VP9 codec."""


class VideoJobStatus(CaseInsensitiveStrEnum):
    """Status values for async video generation jobs."""

    QUEUED = "queued"
    """Job is queued and waiting to start."""

    IN_PROGRESS = "in_progress"
    """Job is currently being processed."""

    COMPLETED = "completed"
    """Job completed successfully."""

    FAILED = "failed"
    """Job failed with an error."""


class VideoSynthType(CaseInsensitiveStrEnum):
    MOVING_SHAPES = "moving_shapes"
    """Generate videos with animated geometric shapes moving across the frame"""

    GRID_CLOCK = "grid_clock"
    """Generate videos with a grid pattern and frame number overlay for frame-accurate verification"""

    NOISE = "noise"
    """Generate videos with random noise frames"""
