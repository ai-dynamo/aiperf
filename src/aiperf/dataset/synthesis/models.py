# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pydantic models for synthesis and analysis data."""

from pydantic import Field
from typing_extensions import Self

from aiperf.common.models import AIPerfBaseModel
from aiperf.config.dataset.defaults import InputTokensDefaults
from aiperf.config.dataset.trace import SynthesisConfig


class MetricStats(AIPerfBaseModel):
    """Statistics for a single metric."""

    mean: float = Field(description="Mean value")
    std_dev: float = Field(description="Standard deviation")
    min: float = Field(description="Minimum value")
    p25: float = Field(description="25th percentile")
    median: float = Field(description="Median (50th percentile)")
    p75: float = Field(description="75th percentile")
    max: float = Field(description="Maximum value")


class AnalysisStats(AIPerfBaseModel):
    """Statistics extracted from trace analysis."""

    total_requests: int = Field(description="Total number of requests in trace")
    unique_prefixes: int = Field(description="Number of unique prefix patterns")
    num_prefix_groups: int = Field(
        ge=0,
        description="Number of distinct shared first blocks (prefix groups)",
    )
    cache_hit_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="Theoretical cache hit rate (0.0 to 1.0) assuming infinite cache",
    )
    min_isl: int = Field(ge=0, description="Minimum input sequence length")
    max_isl: int = Field(ge=0, description="Maximum input sequence length")
    avg_isl: float = Field(ge=0, description="Average input sequence length")
    min_osl: int = Field(ge=0, description="Minimum output sequence length")
    max_osl: int = Field(ge=0, description="Maximum output sequence length")
    avg_osl: float = Field(ge=0, description="Average output sequence length")
    prefix_reuse_ratio: float = Field(
        ge=0.0,
        le=1.0,
        description="Ratio of reused prefixes to total prefixes (0.0 to 1.0)",
    )
    # Extended statistics matching prefix_data_generator output
    isl_stats: MetricStats | None = Field(
        default=None, description="Full statistics for input sequence length"
    )
    osl_stats: MetricStats | None = Field(
        default=None, description="Full statistics for output sequence length"
    )
    context_length_stats: MetricStats | None = Field(
        default=None, description="Full statistics for context (shared prefix) length"
    )
    unique_prompt_length_stats: MetricStats | None = Field(
        default=None, description="Full statistics for unique prompt length"
    )
    hit_rate_stats: MetricStats | None = Field(
        default=None, description="Full statistics for per-request cache hit rates"
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
        default=1,
        ge=1,
        description="Number of independent radix trees to distribute traces across",
    )
    prompt_len_multiplier: float = Field(
        default=1.0,
        ge=0.0,
        description="Multiplier for leaf path (unique prompt) lengths",
    )
    output_len_multiplier: float = Field(
        default=1.0, ge=0.0, description="Multiplier for output lengths"
    )
    max_isl: int | None = Field(
        default=None, ge=1, description="Maximum input sequence length filter"
    )
    max_osl: int | None = Field(
        default=None, ge=1, description="Maximum output sequence length cap"
    )
    block_size: int = Field(
        default=InputTokensDefaults.BLOCK_SIZE, ge=1, description="KV cache page size"
    )
    # Only used for round-trip testing. Typically leave False as rehashing is
    # expensive and unnecessary - the output hash_ids are already valid.
    renormalize_hash_ids: bool = Field(
        default=False,
        description="Whether to apply rolling hash renormalization to output hash_ids",
    )

    @classmethod
    def from_synthesis_config(
        cls, config: SynthesisConfig, block_size: int = InputTokensDefaults.BLOCK_SIZE
    ) -> Self:
        """Create SynthesisParams from a SynthesisConfig and block size.

        Args:
            config: SynthesisConfig from user configuration.
            block_size: KV cache page size for block-aligned calculations.

        Returns:
            SynthesisParams instance with matching values.
        """
        return cls(
            speedup_ratio=config.speedup_ratio,
            prefix_len_multiplier=config.prefix_len_multiplier,
            prefix_root_multiplier=config.prefix_root_multiplier,
            prompt_len_multiplier=config.prompt_len_multiplier,
            output_len_multiplier=config.output_len_multiplier,
            max_isl=config.max_isl,
            max_osl=config.max_osl,
            block_size=block_size,
        )
