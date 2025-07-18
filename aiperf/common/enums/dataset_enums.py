# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.base_enums import CaseInsensitiveStrEnum


class ComposerType(CaseInsensitiveStrEnum):
    """
    The type of composer to use for the dataset.
    """

    SYNTHETIC = "synthetic"
    CUSTOM = "custom"
    PUBLIC_DATASET = "public_dataset"


class CustomDatasetType(CaseInsensitiveStrEnum):
    """Defines the type of JSONL custom dataset from the user."""

    SINGLE_TURN = "single_turn"
    MULTI_TURN = "multi_turn"
    RANDOM_POOL = "random_pool"
    TRACE = "trace"


class ImageFormat(CaseInsensitiveStrEnum):
    """Types of image formats supported by AIPerf."""

    PNG = "png"
    JPEG = "jpeg"
    RANDOM = "random"


class AudioFormat(CaseInsensitiveStrEnum):
    """Types of audio formats supported by AIPerf."""

    WAV = "wav"
    MP3 = "mp3"


class PromptSource(CaseInsensitiveStrEnum):
    """Source of prompts for the model."""

    SYNTHETIC = "synthetic"
    FILE = "file"
    PAYLOAD = "payload"
