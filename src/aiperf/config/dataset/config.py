# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AIPerf Configuration v2.0 - Pydantic Models

Datasets - Data source variants and their discriminated union.

Content-generation sub-configs (prompts, images, audio, video, rankings) and
trace synthesis sub-configs live in sibling ``content.py`` / ``trace.py`` /
``video.py`` modules and are re-exported here so existing
``from aiperf.config.dataset import X`` imports keep working.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    ConfigDict,
    Discriminator,
    Field,
    model_validator,
)

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.enums import (
    DatasetFormat,
    DatasetType,
)
from aiperf.config.base import BaseConfig
from aiperf.config.dataset.content import (
    AudioConfig,
    ImageConfig,
    PrefixPromptConfig,
    PromptConfig,
    RankingsConfig,
)
from aiperf.config.dataset.trace import (
    SynthesisConfig,
)
from aiperf.config.dataset.video import (
    VIDEO_AUDIO_CODEC_MAP,
    VideoAudioConfig,
    VideoConfig,
)
from aiperf.config.loader.normalizers import _hoist_synthetic_prompt_fields
from aiperf.config.types import SamplingDistribution
from aiperf.plugin.enums import DatasetSamplingStrategy, PublicDatasetType

_logger = AIPerfLogger(__name__)

__all__ = [
    "VIDEO_AUDIO_CODEC_MAP",
    "AudioConfig",
    "DatasetConfig",
    "FileDataset",
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


# Shared name field for all DatasetConfig subclasses — extracted to keep each
# class definition compact (file-size budget under tools/ergonomics_baseline).
_DatasetName = Annotated[
    str,
    Field(
        min_length=1,
        description="Dataset identifier — used in result file paths. "
        "Defaults to 'default' for the singular `dataset:` shorthand.",
    ),
]


# Dataset type variants using discriminated unions
class SyntheticDataset(BaseConfig):
    """
    Synthetic dataset configuration.

    Generates prompts programmatically based on token length
    specifications. Ideal for controlled experiments.
    """

    model_config = ConfigDict(extra="forbid")

    name: _DatasetName

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

    isl: Annotated[
        Any | None,
        Field(
            default=None,
            exclude=True,
            json_schema_extra={"x-kubernetes-preserve-unknown-fields": True},
            description=(
                "Shorthand sibling for `prompts.isl`. Accepts a fixed integer or "
                "distribution dict. Hoisted into `prompts.isl` by the before-"
                "validator and excluded from serialization."
            ),
        ),
    ]

    osl: Annotated[
        Any | None,
        Field(
            default=None,
            exclude=True,
            json_schema_extra={"x-kubernetes-preserve-unknown-fields": True},
            description=(
                "Shorthand sibling for `prompts.osl`. Accepts a fixed integer or "
                "distribution dict. Hoisted into `prompts.osl` by the before-"
                "validator and excluded from serialization."
            ),
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
            ge=0.0,
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

    @model_validator(mode="before")
    @classmethod
    def _hoist_isl_osl_shortcuts(cls, data: Any) -> Any:
        """Hoist top-level isl/osl into prompts.{isl,osl} for direct validation.

        AIPerfConfig.parse_datasets already runs this hoist at the list level via
        `_normalize_single_dataset_listed`. This validator covers direct
        `SyntheticDataset.model_validate({...isl...})` callers (programmatic use).
        """
        if isinstance(data, dict):
            _hoist_synthetic_prompt_fields(data)
        return data

    @model_validator(mode="after")
    def _validate_turns_at_least_one(self) -> SyntheticDataset:
        # NormalDistribution.mean keeps ge=0.0 to support OSL=0 / turn_delay=0,
        # so turns must enforce its tighter "at least 1 turn per conversation"
        # contract here. Without this, --conversation-turn-mean 0 (or YAML
        # turns: {mean: 0}) is silently floored to 1 by the composer.
        if self.turns is not None and self.turns.expected_value < 1.0:
            raise ValueError(
                "turns expected value must be >= 1 "
                f"(got {self.turns.expected_value}); set --conversation-turn-mean "
                "to at least 1 or omit it for single-turn conversations."
            )
        return self


class FileDataset(BaseConfig):
    """
    File-based dataset configuration.

    Loads prompts from a local file in various formats.
    Supports trace replay and custom sampling strategies.
    """

    model_config = ConfigDict(extra="forbid")

    name: _DatasetName

    type: Annotated[
        Literal[DatasetType.FILE],
        Field(description="Dataset type discriminator. Must be 'file'."),
    ]

    path: Annotated[
        Path | None,
        Field(
            default=None,
            description="Path to file or directory containing benchmark dataset. "
            "Can be absolute or relative. Mutually exclusive with `records:`. "
            "Supported formats depend on the format field: "
            "JSONL for single_turn/multi_turn, JSONL trace files for mooncake_trace, "
            "directories for random_pool.",
        ),
    ]

    records: Annotated[
        list[dict[str, Any]] | dict[str, list[dict[str, Any]]] | None,
        Field(
            default=None,
            description="Inline benchmark records, embedded directly in the YAML config. "
            "Mutually exclusive with `path:`. The element schema is determined by `format:` "
            "(same shape as one line of the equivalent JSONL file). "
            "For `format: random_pool`, may be either a flat list (single pool) or a "
            "dict-of-lists (multi-pool, mirrors the directory-of-JSONLs file mode). "
            "All other formats require a flat list.",
        ),
    ]

    format: Annotated[
        DatasetFormat,
        Field(
            default=DatasetFormat.SINGLE_TURN,
            description="Dataset file format determining parsing logic and expected file structure. "
            "single_turn: JSONL with single prompt-response exchanges. "
            "multi_turn: JSONL with conversation history. "
            "mooncake_trace / bailian_trace / burst_gpt_trace: timestamped trace files for replay. "
            "sagemaker_data_capture: JSONL captured by SageMaker DataCapture. "
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

    inter_turn_delay_cap_seconds: Annotated[
        float | None,
        Field(
            default=None,
            ge=0.0,
            description="Clamp per-turn replay delays (read from JSONL trace "
            "files) to at most this many seconds. ``None`` disables the cap. "
            "Used by the DAG JSONL loader to keep long pre-recorded waits "
            "from stalling the benchmark; ``DelayCapTracker`` reports the "
            "clamp count at end of load.",
        ),
    ]

    osl: Annotated[
        SamplingDistribution | None,
        Field(
            default=None,
            description="Output sequence length to apply when records do not specify one. "
            "Can be a fixed integer or {mean, stddev} distribution. "
            "Per-line `output_length` values in the file always take precedence.",
        ),
    ]

    @model_validator(mode="after")
    def _validate_source_xor(self) -> FileDataset:
        path_set = self.path is not None
        records_set = self.records is not None
        if path_set == records_set:
            raise ValueError(
                "FileDataset requires exactly one source: set either `path:` "
                "(load from disk) or `records:` (embed in YAML), not both. "
                f"Got path={self.path!r}, records={'<set>' if records_set else None}."
            )

        if records_set and isinstance(self.records, dict):
            if self.format != DatasetFormat.RANDOM_POOL:
                raise ValueError(
                    "`records:` as a dict-of-lists (multi-pool) is only valid "
                    f"for format: random_pool, got format: {self.format}."
                )
            if not self.records:
                raise ValueError("`records:` dict must contain at least one pool.")
            for pool_name, pool_items in self.records.items():
                if not pool_items:
                    raise ValueError(
                        f"`records:` pool '{pool_name}' is empty; "
                        "every pool must contain at least one record."
                    )

        if records_set and isinstance(self.records, list) and not self.records:
            raise ValueError("`records:` must contain at least one record.")

        return self

    @model_validator(mode="after")
    def _warn_large_inline_records(self) -> FileDataset:
        if self.records is None:
            return self
        if isinstance(self.records, dict):
            total = sum(len(p) for p in self.records.values())
        else:
            total = len(self.records)

        from aiperf.common.environment import Environment

        if total > Environment.DATASET.INLINE_RECORDS_WARN_THRESHOLD:
            _logger.warning(
                f"Inline records: dataset '{self.name}' has {total} records inline, which is "
                f"large enough to make the YAML hard to scan. Consider moving "
                f"the dataset to a JSONL file and switching to `path:` instead."
            )
        return self


class PublicDataset(BaseConfig):
    """
    Public dataset configuration.

    Uses well-known public benchmarking datasets that are
    automatically downloaded and processed by AIPerf.
    """

    model_config = ConfigDict(extra="forbid")

    name: _DatasetName

    type: Annotated[
        Literal[DatasetType.PUBLIC],
        Field(description="Dataset type discriminator. Must be 'public'."),
    ]

    dataset: Annotated[
        PublicDatasetType,
        Field(
            description="Pre-configured public dataset to download and use for benchmarking. "
            "Name of the HuggingFace public dataset enum (e.g. 'sharegpt', 'alpaca'). "
            "AIPerf automatically downloads and parses these datasets.",
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

    filters: Annotated[
        dict[str, str],
        Field(
            default_factory=dict,
            description="Dataset-specific filters forwarded to public dataset loaders. "
            "Supported keys and values depend on the selected dataset.",
        ),
    ]


# Union type for all dataset variants using discriminated union
DatasetConfig = Annotated[
    SyntheticDataset | FileDataset | PublicDataset,
    Discriminator("type"),
]
"""
Dataset configuration supporting multiple source types.

Discriminated by 'type' field:
    - synthetic: Generated prompts (type: synthetic)
    - file: Local file data (type: file)
    - public: Public benchmark datasets (type: public)
"""
