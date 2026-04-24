# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AIPerf Configuration v2.0 - Pydantic Models

Trace synthesis and augmentation configs used by file and composed datasets.
Extracted from ``dataset.py`` to keep that module under the file-size threshold.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import (
    ConfigDict,
    Field,
    field_validator,
)

from aiperf.common.enums import OslMode
from aiperf.config._base import BaseConfig
from aiperf.config.types import (
    SamplingDistribution,
    SequenceDistributionEntry,
    validate_probability_distribution,
)


class SynthesisConfig(BaseConfig):
    """
    Configuration for trace synthesis/transformation.

    Used with mooncake_trace format to transform production trace
    data before replay. Allows scaling timestamps, token lengths,
    and radix tree structure.
    """

    model_config = ConfigDict(extra="forbid")

    speedup_ratio: Annotated[
        float,
        Field(
            gt=0.0,
            default=1.0,
            description="Multiplier for timestamp scaling in synthesized traces. "
            "1.0 = real-time, 2.0 = 2x faster, 0.5 = 2x slower.",
        ),
    ]

    prefix_len_multiplier: Annotated[
        float,
        Field(
            gt=0.0,
            default=1.0,
            description="Multiplier for core prefix branch lengths in the radix tree. "
            "1.5 means prefix branches are 50%% longer.",
        ),
    ]

    prefix_root_multiplier: Annotated[
        int,
        Field(
            ge=1,
            default=1,
            description="Number of independent radix trees to distribute traces across. "
            "Higher values increase prefix diversity.",
        ),
    ]

    prompt_len_multiplier: Annotated[
        float,
        Field(
            gt=0.0,
            default=1.0,
            description="Multiplier for leaf path (unique prompt) lengths. "
            "2.0 means prompts are 2x longer.",
        ),
    ]

    max_isl: Annotated[
        int | None,
        Field(
            ge=1,
            default=None,
            description="Maximum input sequence length filter. "
            "Traces with input_length > max_isl are skipped entirely.",
        ),
    ]

    max_osl: Annotated[
        int | None,
        Field(
            ge=1,
            default=None,
            description="Maximum output sequence length cap. "
            "Traces with output_length > max_osl are capped to this value (not filtered).",
        ),
    ]


class AugmentConfig(BaseConfig):
    """
    Configuration for augmenting file datasets with output length specifications.

    Used in composed datasets where file-based prompts need OSL control.
    """

    model_config = ConfigDict(extra="forbid")

    osl: Annotated[
        SamplingDistribution | None,
        Field(
            default=None,
            description="Output sequence length to apply to augmented records. "
            "Can be a fixed integer or {mean, stddev} distribution. "
            "Behavior depends on osl_mode setting.",
        ),
    ]

    osl_mode: Annotated[
        OslMode,
        Field(
            default=OslMode.FILL,
            description="How to apply OSL to records. "
            "fill: only apply if the record lacks an existing OSL value. "
            "override: always replace existing OSL.",
        ),
    ]

    output_distribution: Annotated[
        list[SequenceDistributionEntry] | None,
        Field(
            default=None,
            description="Output length probability distribution. "
            "When specified, overrides the osl field. "
            "Each entry specifies {isl (ignored), osl, probability}. "
            "Probabilities must sum to 100.",
        ),
    ]

    @field_validator("output_distribution")
    @classmethod
    def validate_output_probabilities(
        cls, v: list[SequenceDistributionEntry] | None
    ) -> list[SequenceDistributionEntry] | None:
        if v is not None:
            validate_probability_distribution(v)
        return v
