# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AIPerf Configuration v2.0 - Pydantic Models

Trace synthesis config used by file datasets.
"""

from __future__ import annotations

from typing import Annotated, Literal

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

    max_context_length: Annotated[
        int | None,
        Field(
            ge=1,
            default=None,
            description="Maximum per-trace context length (tokens) for graph-plane "
            "dataset selection (`--max-context-length`). Traces whose input+output "
            "context would exceed this cap are excluded from selection. None (the "
            "default) applies no context-length filter. Raw explicit value carried "
            "verbatim from the CLI; the derived selection default is computed "
            "elsewhere. Ignored by non-graph datasets.",
        ),
    ]

    allow_dataset_wrap: Annotated[
        bool | None,
        Field(
            default=None,
            description="Whether trace-replay dataset selection may wrap (reuse the "
            "finite trace pool) to satisfy the requested load (`--allow-dataset-wrap` / "
            "`--no-allow-dataset-wrap`). None (the default) means unset -- the effective "
            "value is derived downstream and surfaced on `run.resolved.allow_dataset_wrap`; "
            "an explicit True/False here is the raw user intent carried verbatim so the "
            "resolver can distinguish unset from explicit. Consumed by both the agent-graph "
            "replay strategy and the agentic-replay trajectory source (via "
            "`TimingConfig.from_run`); ignored by synthetic/public datasets. An active "
            "cache-bust target also permits repeated agentic traces by keeping traffic "
            "distinct.",
        ),
    ]

    corpus: Annotated[
        Literal["coding", "sonnet"] | None,
        Field(
            default=None,
            description="Corpus backing recorded graph (`dynamo_trace`) real-content "
            "synthesis (`--prompt-corpus`). `coding` (the default when unset) uses "
            "the procedural CodingContentGenerator pool -- the same corpus the "
            "recorded agentic workloads were captured against. `sonnet` uses the "
            "Shakespeare PromptGenerator pool, which yields matching token counts "
            "but different bytes (useful only to reproduce golden fixtures built "
            "from that pool). Ignored by non-graph datasets.",
        ),
    ]
