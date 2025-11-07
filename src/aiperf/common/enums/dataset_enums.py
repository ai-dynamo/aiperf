# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.base_enums import CaseInsensitiveStrEnum


class PublicDatasetType(CaseInsensitiveStrEnum):
    SHAREGPT = "sharegpt"


class ComposerType(CaseInsensitiveStrEnum):
    SYNTHETIC = "synthetic"
    CUSTOM = "custom"
    PUBLIC_DATASET = "public_dataset"


class CustomDatasetType(CaseInsensitiveStrEnum):
    SINGLE_TURN = "single_turn"
    MULTI_TURN = "multi_turn"
    RANDOM_POOL = "random_pool"
    MOONCAKE_TRACE = "mooncake_trace"


class ImageFormat(CaseInsensitiveStrEnum):
    PNG = "png"
    JPEG = "jpeg"
    RANDOM = "random"


class AudioFormat(CaseInsensitiveStrEnum):
    WAV = "wav"
    MP3 = "mp3"


class VideoFormat(CaseInsensitiveStrEnum):
    MP4 = "mp4"
    WEBM = "webm"


class VideoSynthType(CaseInsensitiveStrEnum):
    MOVING_SHAPES = "moving_shapes"
    GRID_CLOCK = "grid_clock"


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
