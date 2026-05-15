# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset/prompt/modality default value dataclasses.

These ``*Defaults`` dataclasses mirror the corresponding CLIConfig and
``aiperf.config.dataset.*`` Pydantic field defaults. They exist so unit
tests and external callers have stable named constants to compare against;
production code should reference the per-class field defaults directly.
"""

from dataclasses import dataclass

from aiperf.common.enums import (
    AudioFormat,
    ImageFormat,
    VideoFormat,
    VideoSynthType,
)
from aiperf.plugin.enums import DatasetSamplingStrategy


@dataclass(frozen=True)
class InputDefaults:
    BATCH_SIZE = 1
    EXTRA = []
    HEADERS = []
    FILE = None
    FIXED_SCHEDULE = False
    FIXED_SCHEDULE_AUTO_OFFSET = False
    FIXED_SCHEDULE_START_OFFSET = None
    FIXED_SCHEDULE_END_OFFSET = None
    GOODPUT = None
    PUBLIC_DATASET = None
    CUSTOM_DATASET_TYPE = None
    DATASET_SAMPLING_STRATEGY = DatasetSamplingStrategy.SHUFFLE
    RANDOM_SEED = None
    NUM_DATASET_ENTRIES = 100


@dataclass(frozen=True)
class InputTokensDefaults:
    MEAN = 550
    STDDEV = 0.0
    BLOCK_SIZE = 512


@dataclass(frozen=True)
class AudioDefaults:
    BATCH_SIZE = 1
    LENGTH_MEAN = 0.0
    LENGTH_STDDEV = 0.0
    FORMAT = AudioFormat.WAV
    DEPTHS = [16]
    SAMPLE_RATES = [16.0]
    NUM_CHANNELS = 1


@dataclass(frozen=True)
class ImageDefaults:
    BATCH_SIZE = 1
    WIDTH_MEAN = 0.0
    WIDTH_STDDEV = 0.0
    HEIGHT_MEAN = 0.0
    HEIGHT_STDDEV = 0.0
    FORMAT = ImageFormat.PNG


@dataclass(frozen=True)
class VideoDefaults:
    BATCH_SIZE = 1
    DURATION = 5.0
    FPS = 4
    WIDTH = None
    HEIGHT = None
    SYNTH_TYPE = VideoSynthType.MOVING_SHAPES
    FORMAT = VideoFormat.WEBM
    CODEC = "libvpx-vp9"


@dataclass(frozen=True)
class VideoAudioDefaults:
    SAMPLE_RATE = 44100
    CHANNELS = 0
    CODEC = None
    DEPTH = 16


@dataclass(frozen=True)
class PromptDefaults:
    BATCH_SIZE = 1
    NUM = 100


@dataclass(frozen=True)
class PrefixPromptDefaults:
    POOL_SIZE = 0
    LENGTH = 0


@dataclass(frozen=True)
class ConversationDefaults:
    NUM = None


@dataclass(frozen=True)
class TurnDefaults:
    MEAN = 1
    STDDEV = 0


@dataclass(frozen=True)
class TurnDelayDefaults:
    MEAN = 0.0
    STDDEV = 0.0
    RATIO = 1.0


@dataclass(frozen=True)
class OutputTokensDefaults:
    STDDEV = 0


__all__ = [
    "AudioDefaults",
    "ConversationDefaults",
    "ImageDefaults",
    "InputDefaults",
    "InputTokensDefaults",
    "OutputTokensDefaults",
    "PrefixPromptDefaults",
    "PromptDefaults",
    "TurnDefaults",
    "TurnDelayDefaults",
    "VideoAudioDefaults",
    "VideoDefaults",
]
