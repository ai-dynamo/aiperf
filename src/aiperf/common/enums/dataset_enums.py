# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.base_enums import CaseInsensitiveStrEnum


class PublicDatasetType(CaseInsensitiveStrEnum):
    """Public datasets available for benchmarking."""

    SHAREGPT = "sharegpt"
    """ShareGPT dataset from HuggingFace. Multi-turn conversational dataset with user/assistant exchanges."""


class ComposerType(CaseInsensitiveStrEnum):
    SYNTHETIC = "synthetic"
    CUSTOM = "custom"
    PUBLIC_DATASET = "public_dataset"
    SYNTHETIC_RANKINGS = "synthetic_rankings"


class CustomDatasetType(CaseInsensitiveStrEnum):
    """Custom dataset formats for loading user-provided data."""

    SINGLE_TURN = "single_turn"
    """JSONL file with one request per line. Supports multi-modal data and client-side batching. Does not support multi-turn features."""

    MULTI_TURN = "multi_turn"
    """JSONL file with conversation histories. Each line contains an array of messages. Supports multi-modal data, multi-turn features, and client-side batching."""

    RANDOM_POOL = "random_pool"
    """JSONL file with a pool of prompts randomly sampled to construct multi-turn conversations. Single file creates one pool; directory creates multiple pools."""

    MOONCAKE_TRACE = "mooncake_trace"
    """JSONL file in Mooncake trace format. Each line contains timestamp, input/output lengths, and optional session_id for replaying production workloads."""


class ImageFormat(CaseInsensitiveStrEnum):
    """Image file formats for synthetic image generation."""

    PNG = "png"
    """PNG format. Lossless compression, larger file sizes, best quality."""

    JPEG = "jpeg"
    """JPEG format. Lossy compression, smaller file sizes, good for photos."""

    RANDOM = "random"
    """Randomly select PNG or JPEG for each image."""


class AudioFormat(CaseInsensitiveStrEnum):
    """Audio file formats for synthetic audio generation."""

    WAV = "wav"
    """WAV format. Uncompressed audio, larger file sizes, best quality."""

    MP3 = "mp3"
    """MP3 format. Compressed audio, smaller file sizes, good quality."""


class VideoFormat(CaseInsensitiveStrEnum):
    """Video container formats for synthetic video generation."""

    MP4 = "mp4"
    """MP4 container. Widely compatible, good for H.264/H.265 codecs."""

    WEBM = "webm"
    """WebM container. Open format, optimized for web, good for VP9 codec."""


class VideoSynthType(CaseInsensitiveStrEnum):
    MOVING_SHAPES = "moving_shapes"
    """Generate videos with animated geometric shapes moving across the frame"""

    GRID_CLOCK = "grid_clock"
    """Generate videos with a grid pattern and timestamp overlay for frame-accurate verification"""


class PromptSource(CaseInsensitiveStrEnum):
    SYNTHETIC = "synthetic"
    FILE = "file"
    PAYLOAD = "payload"


class DatasetSamplingStrategy(CaseInsensitiveStrEnum):
    SEQUENTIAL = "sequential"
    """Iterate through the dataset sequentially, then wrap around to the beginning."""

    RANDOM = "random"
    """Randomly select a conversation from the dataset. Will randomly sample with replacement."""

    SHUFFLE = "shuffle"
    """Shuffle the dataset and iterate through it. Will randomly sample without replacement.
    Once the end of the dataset is reached, shuffle the dataset again and start over."""


class DatasetBackingStoreType(CaseInsensitiveStrEnum):
    """Types of dataset backing stores (DatasetManager side)."""

    MEMORY_MAP = "memory_map"
    """Store dataset in memory-mapped files for zero-copy worker access."""


class DatasetClientStoreType(CaseInsensitiveStrEnum):
    """Types of dataset client stores (Worker side)."""

    MEMORY_MAP = "memory_map"
    """Read from memory-mapped files (zero-copy, O(1) lookup)."""
