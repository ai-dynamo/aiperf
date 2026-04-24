# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AIPerf Configuration v2.0 - Pydantic Models

Datasets - Data source variants and their discriminated union.

Content-generation sub-configs (prompts, images, audio, video, rankings) and
trace/augment configs live in sibling ``_dataset_*`` modules and are re-exported
here so existing ``from aiperf.config.dataset import X`` imports keep working.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    ConfigDict,
    Discriminator,
    Field,
)

from aiperf.common.enums import (
    DatasetFormat,
    DatasetType,
)
from aiperf.config._base import BaseConfig
from aiperf.config._dataset_content import (
    AudioConfig,
    ImageConfig,
    PrefixPromptConfig,
    PromptConfig,
    RankingsConfig,
)
from aiperf.config._dataset_trace import (
    AugmentConfig,
    SynthesisConfig,
)
from aiperf.config._dataset_video import (
    VIDEO_AUDIO_CODEC_MAP,
    VideoAudioConfig,
    VideoConfig,
)
from aiperf.config.types import SamplingDistribution
from aiperf.plugin.enums import DatasetSamplingStrategy, PublicDatasetType

__all__ = [
    "VIDEO_AUDIO_CODEC_MAP",
    "AudioConfig",
    "AugmentConfig",
    "ComposedDataset",
    "DatasetConfig",
    "FileDataset",
    "FileSourceConfig",
    "ImageConfig",
    "PrefixPromptConfig",
    "PromptConfig",
    "PublicDataset",
    "RankingsConfig",
    "SynthesisConfig",
    "SyntheticDataset",
    "VideoAudioConfig",
    "VideoConfig",
]


# Dataset type variants using discriminated unions
class SyntheticDataset(BaseConfig):
    """
    Synthetic dataset configuration.

    Generates prompts programmatically based on token length
    specifications. Ideal for controlled experiments.
    """

    model_config = ConfigDict(extra="forbid")

    type: Annotated[
        Literal[DatasetType.SYNTHETIC],
        Field(description="Dataset type discriminator. Must be 'synthetic'."),
    ]

    entries: Annotated[
        int,
        Field(
            ge=1,
            default=100,
            description="Total number of unique entries to generate for the dataset. "
            "Each entry represents a unique prompt with sampled ISL/OSL. "
            "Entries are reused across conversations and turns according to "
            "the sampling strategy. Higher values provide more diversity.",
        ),
    ]

    random_seed: Annotated[
        int | None,
        Field(
            default=None,
            description="Random seed for deterministic dataset generation. "
            "When set, makes synthetic prompts, sampling, and other random operations "
            "reproducible across runs. Essential for A/B testing and debugging. "
            "Overrides global random_seed for this dataset.",
        ),
    ]

    sampling: Annotated[
        DatasetSamplingStrategy,
        Field(
            default=DatasetSamplingStrategy.SEQUENTIAL,
            description="Strategy for selecting entries from dataset during benchmarking. "
            "sequential: iterate in order, wrapping to start after end. "
            "random: randomly sample with replacement (entries may repeat). "
            "shuffle: random permutation without replacement, re-shuffling after exhaustion.",
        ),
    ]

    prompts: Annotated[
        PromptConfig | None,
        Field(
            default=None,
            description="Prompt/token length configuration specifying ISL, OSL, "
            "sequence distributions, and batch processing settings.",
        ),
    ]

    prefix_prompts: Annotated[
        PrefixPromptConfig | None,
        Field(
            default=None,
            description="Shared prefix configuration for KV cache testing. "
            "Generates prefix prompts that are prepended to user prompts, "
            "simulating cached context scenarios.",
        ),
    ]

    turns: Annotated[
        SamplingDistribution | None,
        Field(
            default=None,
            description="Number of request-response turns per conversation. "
            "Can be a fixed integer or {mean, stddev} distribution. "
            "Each turn consists of a user message and model response. "
            "Set to 1 for single-turn interactions. "
            "Multi-turn conversations enable testing of context retention "
            "and conversation history handling.",
        ),
    ]

    turn_delay: Annotated[
        SamplingDistribution | None,
        Field(
            default=None,
            description="Delay in milliseconds between consecutive turns within a "
            "multi-turn conversation. Can be a fixed value or {mean, stddev} distribution. "
            "Simulates user think time between receiving a response and sending "
            "the next message. Only applies when turns > 1. "
            "Set to 0 for back-to-back turns.",
        ),
    ]

    turn_delay_ratio: Annotated[
        float,
        Field(
            gt=0.0,
            default=1.0,
            description="Multiplier for scaling all turn delays. "
            "Applied after mean/stddev calculation: actual_delay = calculated_delay * ratio. "
            "Values < 1 speed up conversations, > 1 slow them down. "
            "Set to 0 to eliminate delays entirely.",
        ),
    ]

    images: Annotated[
        ImageConfig | None,
        Field(
            default=None,
            description="Synthetic image configuration for multimodal vision-language testing.",
        ),
    ]

    audio: Annotated[
        AudioConfig | None,
        Field(
            default=None,
            description="Synthetic audio configuration for multimodal speech/audio testing.",
        ),
    ]

    video: Annotated[
        VideoConfig | None,
        Field(
            default=None,
            description="Synthetic video configuration for multimodal video understanding testing.",
        ),
    ]

    rankings: Annotated[
        RankingsConfig | None,
        Field(
            default=None,
            description="Rankings/reranking configuration for generating query-passage pairs. "
            "Only relevant for rankings endpoint types.",
        ),
    ]


class FileDataset(BaseConfig):
    """
    File-based dataset configuration.

    Loads prompts from a local file in various formats.
    Supports trace replay and custom sampling strategies.
    """

    model_config = ConfigDict(extra="forbid")

    type: Annotated[
        Literal[DatasetType.FILE],
        Field(description="Dataset type discriminator. Must be 'file'."),
    ]

    path: Annotated[
        Path,
        Field(
            description="Path to file or directory containing benchmark dataset. "
            "Can be absolute or relative. Supported formats depend on the format field: "
            "JSONL for single_turn/multi_turn, JSONL trace files for mooncake_trace, "
            "directories for random_pool."
        ),
    ]

    format: Annotated[
        DatasetFormat,
        Field(
            default=DatasetFormat.SINGLE_TURN,
            description="Dataset file format determining parsing logic and expected file structure. "
            "single_turn: JSONL with single prompt-response exchanges. "
            "multi_turn: JSONL with conversation history. "
            "mooncake_trace: timestamped trace files for replay. "
            "random_pool: directory of reusable prompts.",
        ),
    ]

    sampling: Annotated[
        DatasetSamplingStrategy,
        Field(
            default=DatasetSamplingStrategy.SEQUENTIAL,
            description="Strategy for selecting entries from dataset during benchmarking. "
            "sequential: iterate in order, wrapping to start after end. "
            "random: randomly sample with replacement (entries may repeat). "
            "shuffle: random permutation without replacement, re-shuffling after exhaustion.",
        ),
    ]

    synthesis: Annotated[
        SynthesisConfig | None,
        Field(
            default=None,
            description="Trace synthesis/transformation configuration. "
            "Allows scaling timestamps and token lengths before replay. "
            "Only used with mooncake_trace format.",
        ),
    ]

    entries: Annotated[
        int | None,
        Field(
            ge=1,
            default=None,
            description="Limit number of records to use from file. "
            "If not specified, uses all records in the file.",
        ),
    ]

    random_seed: Annotated[
        int | None,
        Field(
            default=None,
            description="Random seed for deterministic sampling. "
            "When set, makes random/shuffle sampling reproducible across runs. "
            "Overrides global random_seed for this dataset.",
        ),
    ]


class PublicDataset(BaseConfig):
    """
    Public dataset configuration.

    Uses well-known public benchmarking datasets that are
    automatically downloaded and processed by AIPerf.
    """

    model_config = ConfigDict(extra="forbid")

    type: Annotated[
        Literal[DatasetType.PUBLIC],
        Field(description="Dataset type discriminator. Must be 'public'."),
    ]

    name: Annotated[
        PublicDatasetType,
        Field(
            description="Pre-configured public dataset to download and use for benchmarking. "
            "AIPerf automatically downloads and parses these datasets. "
        ),
    ]

    entries: Annotated[
        int | None,
        Field(
            ge=1,
            default=None,
            description="Limit number of records to use from the dataset. "
            "If not specified, uses all available records.",
        ),
    ]

    random_seed: Annotated[
        int | None,
        Field(
            default=None,
            description="Random seed for deterministic sampling from the dataset. "
            "Overrides global random_seed for this dataset.",
        ),
    ]

    sampling: Annotated[
        DatasetSamplingStrategy,
        Field(
            default=DatasetSamplingStrategy.SEQUENTIAL,
            description="Strategy for selecting entries from dataset during benchmarking. "
            "sequential: iterate in order, wrapping to start after end. "
            "random: randomly sample with replacement (entries may repeat). "
            "shuffle: random permutation without replacement, re-shuffling after exhaustion.",
        ),
    ]

    hf_subset: Annotated[
        str | None,
        Field(
            default=None,
            description="HuggingFace dataset subset/config name override (e.g. 'sharegpt4o'). "
            "Only applies for HuggingFace-backed public dataset loaders. "
            "Takes priority over the subset defined in the plugin registry.",
        ),
    ]


class FileSourceConfig(BaseConfig):
    """
    File source configuration for composed datasets.

    Simplified file dataset specification used within composed
    dataset source field.
    """

    model_config = ConfigDict(extra="forbid")

    type: Annotated[
        Literal[DatasetType.FILE],
        Field(description="Source type. Must be 'file' for composed datasets."),
    ]

    path: Annotated[
        Path,
        Field(description="Path to the source file. Can be absolute or relative."),
    ]

    format: Annotated[
        DatasetFormat,
        Field(
            default=DatasetFormat.SINGLE_TURN,
            description="Dataset file format determining parsing logic. "
            "single_turn: JSONL with single exchanges. "
            "multi_turn: JSONL with conversation history. "
            "mooncake_trace: timestamped trace files.",
        ),
    ]

    sampling: Annotated[
        DatasetSamplingStrategy,
        Field(
            default=DatasetSamplingStrategy.SEQUENTIAL,
            description="Strategy for selecting entries from the source file. "
            "sequential: iterate in order. "
            "random: randomly sample with replacement. "
            "shuffle: random permutation without replacement.",
        ),
    ]


class ComposedDataset(BaseConfig):
    """
    Composed dataset configuration (unique to AIPerf).

    Combines file-based data with synthetic augmentation.
    This enables advanced scenarios like:
    - Adding system prompts to existing queries
    - Testing KV cache with file-based prompts
    - Adding multimodal content to text datasets
    - Extending small datasets with padding
    """

    model_config = ConfigDict(extra="forbid")

    type: Annotated[
        Literal[DatasetType.COMPOSED],
        Field(
            default=DatasetType.COMPOSED,
            description="Dataset type discriminator. Must be 'composed'.",
        ),
    ]

    source: Annotated[
        FileSourceConfig,
        Field(description="The base file dataset to augment."),
    ]

    augment: Annotated[
        AugmentConfig,
        Field(
            description="Augmentation configuration specifying prefixes, suffixes, "
            "OSL, multimodal content, and padding for the source data."
        ),
    ]

    entries: Annotated[
        int | None,
        Field(
            ge=1,
            default=None,
            description="Final dataset size after augmentation. "
            "If source has fewer entries and pad_to_count is set in augment, "
            "synthetic padding entries are generated to reach this count.",
        ),
    ]

    random_seed: Annotated[
        int | None,
        Field(
            default=None,
            description="Random seed for deterministic augmentation. "
            "Overrides global random_seed for this dataset.",
        ),
    ]


# Union type for all dataset variants using discriminated union
DatasetConfig = Annotated[
    SyntheticDataset | FileDataset | PublicDataset | ComposedDataset,
    Discriminator("type"),
]
"""
Dataset configuration supporting multiple source types.

Discriminated by 'type' field or structure:
    - synthetic: Generated prompts (type: synthetic)
    - file: Local file data (type: file)
    - public: Public benchmark datasets (type: public)
    - composed: Combined dataset (has source + augment fields)
"""
