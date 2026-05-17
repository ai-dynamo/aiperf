# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Sequence length distribution models for AIPerf benchmarking.

This module provides data models and parsers for specifying distributions of input sequence
length (ISL) and output sequence length (OSL) pairs with optional standard deviations,
allowing for more realistic LLM benchmarking scenarios.

The sequence distribution feature allows users to specify multiple ISL/OSL pairs with
different probabilities, enabling simulation of mixed workloads that better represent
production traffic patterns.

        Supported formats (probabilities must be percentages 0-100):
        - Semicolon: "256,128:40;512,256:60" or "256|10,128|5:40;512|20,256|10:60"
        - Bracket: "[(256,128):40,(512,256):60]" or "[(256|10,128|5):40,(512|20,256|10):60]"
        - JSON: '{"pairs": [{"isl": 256, "isl_stddev": 10, "osl": 128, "osl_stddev": 5, "prob": 40}, ...]}'

Note: Probabilities must be specified as percentages (0-100), not fractions (0-1).
This prevents common errors from mixing different probability formats.

Examples:
    Basic usage:
        >>> from aiperf.common.models.sequence_distribution import DistributionParser
        >>> dist = DistributionParser.parse("256,128:60;512,256:40")
        >>> isl, osl = dist.sample()

    With standard deviations:
        >>> dist = DistributionParser.parse("256|10,128|5:60;512|20,256|10:40")
        >>> isl, osl = dist.sample()  # Will vary around means based on stddev
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.models.sequence_distribution_core import (
    SequenceLengthDistribution,
    SequenceLengthPair,
)
from aiperf.common.models.sequence_distribution_parser import (
    DistributionParser as DistributionParser,
)

if TYPE_CHECKING:
    pass

logger = AIPerfLogger(__name__)


def _validate_probability_sum(pairs: list[SequenceLengthPair]) -> None:
    """
    Validate that probabilities sum to approximately 100.0.

    This is a module-level helper used by both SequenceLengthDistribution
    and DistributionParser to avoid code duplication.

    Args:
        pairs: List of SequenceLengthPair objects to validate

    Raises:
        ValueError: If probabilities don't sum to 100.0 (within floating-point tolerance)
    """
    total_prob = sum(pair.probability for pair in pairs)

    # Allow small floating-point errors
    if not np.isclose(total_prob, 100.0, rtol=1e-6, atol=1e-6):
        raise ValueError(
            f"Probabilities must sum to 100.0, got {total_prob:.6f}. "
            f"Pairs: {[str(p) for p in pairs]}"
        )


def create_uniform_distribution(isl: int, osl: int) -> SequenceLengthDistribution:
    """
    Create a uniform distribution with a single ISL/OSL pair.

    Args:
        isl: Input sequence length
        osl: Output sequence length

    Returns:
        SequenceLengthDistribution with single pair at 100% probability
    """
    return SequenceLengthDistribution([SequenceLengthPair(isl, osl, 100.0)])


def create_balanced_distribution(
    pairs: list[tuple[int, int]],
) -> SequenceLengthDistribution:
    """
    Create a balanced distribution where all pairs have equal probability.

    Args:
        pairs: List of (isl, osl) tuples

    Returns:
        SequenceLengthDistribution with equal probabilities
    """
    if not pairs:
        raise ValueError("Cannot create distribution from empty pairs list")

    prob_per_pair = 100.0 / len(pairs)
    seq_pairs = [SequenceLengthPair(isl, osl, prob_per_pair) for isl, osl in pairs]

    return SequenceLengthDistribution(seq_pairs)
