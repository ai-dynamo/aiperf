# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AIPerf Configuration v2.0 - Pydantic Models

Trace synthesis config used by file datasets.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import (
    ConfigDict,
    Field,
)

from aiperf.config.base import BaseConfig


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

    output_len_multiplier: Annotated[
        float,
        Field(
            ge=0.0,
            default=1.0,
            description="Multiplier for output lengths in synthesized traces.",
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


def synthesis_should_apply(synthesis: SynthesisConfig | None) -> bool:
    """Return True when a SynthesisConfig has any non-default transform set.

    Shared gate for every "did the user configure trace synthesis?" site
    (trace loaders and the custom-dataset validator). Trace loaders only invoke
    the Synthesizer when a transform multiplier differs from its identity
    default. Defaults: speedup_ratio=1.0, prefix_len_multiplier=1.0,
    prefix_root_multiplier=1, prompt_len_multiplier=1.0, output_len_multiplier=1.0.

    ``max_isl`` / ``max_osl`` are filters/caps, not transforms, so they are
    intentionally excluded here; the custom-composer validator ORs those in
    separately when rejecting synthesis on non-trace datasets.

    Example:
        >>> synthesis_should_apply(SynthesisConfig(output_len_multiplier=2.0))
        True
        >>> synthesis_should_apply(SynthesisConfig(max_isl=4096))
        False
    """
    if synthesis is None:
        return False
    return (
        synthesis.speedup_ratio != 1.0
        or synthesis.prefix_len_multiplier != 1.0
        or synthesis.prefix_root_multiplier != 1
        or synthesis.prompt_len_multiplier != 1.0
        or synthesis.output_len_multiplier != 1.0
    )
