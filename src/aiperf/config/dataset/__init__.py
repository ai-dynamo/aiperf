# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset configuration sub-package.

Public surface preserved: ``from aiperf.config.dataset import X`` keeps working.
"""

from aiperf.config.dataset.config import (
    DatasetConfig,
    FileDataset,
    PublicDataset,
    SyntheticDataset,
)
from aiperf.config.dataset.content import (
    AudioConfig,
    ImageConfig,
    PrefixPromptConfig,
    PromptConfig,
    PromptSelectionConfig,
    RankingsConfig,
)
from aiperf.config.dataset.narrow import as_file_dataset, as_trace_replay_dataset
from aiperf.config.dataset.resolver import DatasetResolver
from aiperf.config.dataset.trace import SynthesisConfig
from aiperf.config.dataset.video import (
    VIDEO_AUDIO_CODEC_MAP,
    VideoAudioConfig,
    VideoConfig,
)

__all__ = [
    "VIDEO_AUDIO_CODEC_MAP",
    "AudioConfig",
    "DatasetConfig",
    "DatasetResolver",
    "FileDataset",
    "ImageConfig",
    "PrefixPromptConfig",
    "PromptConfig",
    "PromptSelectionConfig",
    "PublicDataset",
    "RankingsConfig",
    "SynthesisConfig",
    "SyntheticDataset",
    "VideoAudioConfig",
    "VideoConfig",
    "as_file_dataset",
    "as_trace_replay_dataset",
]
