# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pydantic models for synthesis and analysis data."""

from pydantic import Field

from aiperf.common.models import AIPerfBaseModel


class AnalysisStats(AIPerfBaseModel):
    """Statistics extracted from trace analysis."""

    total_requests: int = Field(description="Total number of requests in trace")
    unique_prefixes: int = Field(description="Number of unique prefix patterns")
    cache_hit_rate: float = Field(
        description="Theoretical cache hit rate (0.0 to 1.0) assuming infinite cache"
    )
    min_isl: int = Field(description="Minimum input sequence length")
    max_isl: int = Field(description="Maximum input sequence length")
    avg_isl: float = Field(description="Average input sequence length")
    min_osl: int = Field(description="Minimum output sequence length")
    max_osl: int = Field(description="Maximum output sequence length")
    avg_osl: float = Field(description="Average output sequence length")
    prefix_reuse_ratio: float = Field(
        description="Ratio of reused prefixes to total prefixes (0.0 to 1.0)"
    )


class SynthesisParams(AIPerfBaseModel):
    """Parameters for synthetic trace generation."""

    speedup_ratio: float = Field(
        default=1.0, ge=0.0, description="Multiplier for timestamp scaling"
    )
    prefix_len_multiplier: float = Field(
        default=1.0, ge=0.0, description="Multiplier for core prefix branch lengths"
    )
    prefix_root_multiplier: int = Field(
        default=1, ge=1, description="Number of times to replicate the radix tree"
    )
    prompt_len_multiplier: float = Field(
        default=1.0,
        ge=0.0,
        description="Multiplier for leaf path (unique prompt) lengths",
    )
    max_isl: int | None = Field(
        default=None, ge=1, description="Maximum input sequence length filter"
    )
    block_size: int = Field(default=512, ge=1, description="KV cache page size")
