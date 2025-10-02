# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Sequence Length Distribution Module

Provides configurable distributions of input/output sequence length pairs
for realistic LLM benchmarking workloads.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SequenceLengthPair:
    """Immutable representation of an ISL/OSL pair with probability weight."""

    input_seq_len: int
    output_seq_len: int
    probability: float

    def __post_init__(self) -> None:
        """Validate sequence lengths and probability on construction."""
        if self.input_seq_len <= 0:
            raise ValueError(
                f"Input sequence length must be positive, got {self.input_seq_len}"
            )
        if self.output_seq_len <= 0:
            raise ValueError(
                f"Output sequence length must be positive, got {self.output_seq_len}"
            )
        if not 0.0 <= self.probability <= 100.0:
            raise ValueError(f"Probability must be in [0,100], got {self.probability}")

    def __str__(self) -> str:
        return f"({self.input_seq_len},{self.output_seq_len}):{self.probability}%"


class SequenceLengthDistribution:
    """
    Manages probability distributions of ISL/OSL pairs for benchmark sampling.

    Supports efficient O(log n) sampling using binary search on cumulative
    probability distribution.
    """

    def __init__(self, pairs: list[SequenceLengthPair]) -> None:
        """
        Initialize distribution from list of sequence length pairs.

        Args:
            pairs: List of SequenceLengthPair objects. Probabilities must sum to 1.0.

        Raises:
            ValueError: If pairs is empty or probabilities don't sum to 1.0.
        """
        if not pairs:
            raise ValueError(
                "Distribution must contain at least one sequence length pair"
            )

        self._pairs = tuple(pairs)  # Immutable copy
        self._validate_probabilities()
        self._cumulative_probs = self._compute_cumulative_probabilities()

        logger.debug(f"Created distribution with {len(self._pairs)} pairs: {self}")

    def _validate_probabilities(self) -> None:
        """Validate that probabilities sum to approximately 100.0."""
        total_prob = sum(pair.probability for pair in self._pairs)

        # Allow small floating-point errors
        if not np.isclose(total_prob, 100.0, rtol=1e-6, atol=1e-6):
            raise ValueError(
                f"Probabilities must sum to 100.0, got {total_prob:.6f}. "
                f"Pairs: {[str(p) for p in self._pairs]}"
            )

    def _compute_cumulative_probabilities(self) -> np.ndarray:
        """Compute cumulative probability distribution for efficient sampling."""
        # Convert percentages to fractions for internal calculation
        probs = [pair.probability / 100.0 for pair in self._pairs]
        return np.cumsum(probs, dtype=np.float64)

    def sample(
        self, random_state: int | np.random.Generator | None = None
    ) -> tuple[int, int]:
        """
        Sample an (ISL, OSL) pair according to the distribution.

        Args:
            random_state: Random number generator or seed for reproducible sampling.

        Returns:
            Tuple of (input_seq_len, output_seq_len)
        """
        if isinstance(random_state, int):
            rng = np.random.default_rng(random_state)
        elif isinstance(random_state, np.random.Generator):
            rng = random_state
        else:
            rng = np.random.default_rng()

        rand_val = rng.random()

        # Binary search for efficiency with large distributions
        idx = np.searchsorted(self._cumulative_probs, rand_val, side="right")
        idx = min(idx, len(self._pairs) - 1)  # Handle edge case

        pair = self._pairs[idx]
        return (pair.input_seq_len, pair.output_seq_len)

    def sample_batch(
        self, batch_size: int, random_state: int | np.random.Generator | None = None
    ) -> list[tuple[int, int]]:
        """
        Sample multiple (ISL, OSL) pairs efficiently.

        Args:
            batch_size: Number of pairs to sample
            random_state: Random number generator or seed

        Returns:
            List of (input_seq_len, output_seq_len) tuples
        """
        if batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {batch_size}")

        if isinstance(random_state, int):
            rng = np.random.default_rng(random_state)
        elif isinstance(random_state, np.random.Generator):
            rng = random_state
        else:
            rng = np.random.default_rng()

        rand_vals = rng.random(batch_size)
        indices = np.searchsorted(self._cumulative_probs, rand_vals, side="right")
        indices = np.clip(indices, 0, len(self._pairs) - 1)

        return [
            (self._pairs[idx].input_seq_len, self._pairs[idx].output_seq_len)
            for idx in indices
        ]

    @property
    def pairs(self) -> tuple[SequenceLengthPair, ...]:
        """Get immutable view of sequence length pairs."""
        return self._pairs

    def get_statistics(self) -> dict[str, int | float | list[tuple[int, int, float]]]:
        """
        Get comprehensive statistics about the distribution.

        Returns:
            Dictionary with distribution statistics including expected values,
            variance, and individual pair information.
        """
        # Expected values (convert percentages to fractions for calculation)
        exp_isl = sum(p.input_seq_len * (p.probability / 100.0) for p in self._pairs)
        exp_osl = sum(p.output_seq_len * (p.probability / 100.0) for p in self._pairs)

        # Variance calculations
        var_isl = sum(
            (p.probability / 100.0) * (p.input_seq_len - exp_isl) ** 2
            for p in self._pairs
        )
        var_osl = sum(
            (p.probability / 100.0) * (p.output_seq_len - exp_osl) ** 2
            for p in self._pairs
        )

        return {
            "num_pairs": len(self._pairs),
            "expected_isl": exp_isl,
            "expected_osl": exp_osl,
            "variance_isl": var_isl,
            "variance_osl": var_osl,
            "std_isl": np.sqrt(var_isl),
            "std_osl": np.sqrt(var_osl),
            "pairs": [
                (p.input_seq_len, p.output_seq_len, p.probability) for p in self._pairs
            ],
            "total_probability": sum(p.probability for p in self._pairs),
        }

    def __str__(self) -> str:
        """String representation showing all pairs."""
        pairs_str = ";".join(str(pair) for pair in self._pairs)
        return f"SequenceLengthDistribution[{pairs_str}]"

    def __repr__(self) -> str:
        return f"SequenceLengthDistribution({list(self._pairs)})"


def _normalize_probability(prob: float) -> float:
    """
    Normalize probability to percentage format (0-100).

    Supports backward compatibility:
    - If prob <= 1.0, treat as fraction and convert to percentage
    - If prob > 1.0, treat as percentage
    """
    if prob <= 1.0:
        # Backward compatibility: treat as fraction (0-1) and convert to percentage
        return prob * 100.0
    else:
        # New format: already a percentage
        return prob


class DistributionParser:
    """Parser for various sequence length distribution string formats."""

    # Regex patterns for different formats (allow whitespace)
    SEMICOLON_PATTERN = re.compile(r"(\d+)\s*,\s*(\d+)\s*:\s*([0-9]*\.?[0-9]+)")
    BRACKET_PATTERN = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*:\s*([0-9]*\.?[0-9]+)")

    @classmethod
    def parse(cls, dist_str: str) -> SequenceLengthDistribution:
        """
        Parse distribution string in various supported formats.

        Supported formats:
        - Semicolon: "256,128:40;512,256:60" (percentages) or "256,128:0.4;512,256:0.6" (fractions)
        - Bracket: "[(256,128):40,(512,256):60]" (percentages) or "[(256,128):0.4,(512,256):0.6]" (fractions)
        - JSON: '{"pairs": [{"isl": 256, "osl": 128, "prob": 40}, ...]}' (percentages)

        Args:
            dist_str: Distribution specification string

        Returns:
            SequenceLengthDistribution object

        Raises:
            ValueError: If string format is invalid or unrecognized
        """
        if not isinstance(dist_str, str) or not dist_str.strip():
            raise ValueError("Distribution string cannot be empty")

        dist_str = dist_str.strip()

        try:
            # Try JSON format first
            if dist_str.startswith("{"):
                return cls._parse_json_format(dist_str)

            # Try bracket format
            if dist_str.startswith("[") and dist_str.endswith("]"):
                return cls._parse_bracket_format(dist_str[1:-1])

            # Default to semicolon format
            return cls._parse_semicolon_format(dist_str)

        except Exception as e:
            raise ValueError(
                f"Failed to parse distribution string '{dist_str}': {e}"
            ) from e

    @classmethod
    def _parse_json_format(cls, json_str: str) -> SequenceLengthDistribution:
        """Parse JSON format: {"pairs": [{"isl": 256, "osl": 128, "prob": 0.4}, ...]}"""
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}") from None

        # Validate structure outside the JSON parsing try-catch
        if "pairs" not in data:
            raise ValueError("JSON format must contain 'pairs' key")

        try:
            pairs = []
            for i, pair_data in enumerate(data["pairs"]):
                required_keys = {"isl", "osl", "prob"}
                if not required_keys.issubset(pair_data.keys()):
                    missing = required_keys - pair_data.keys()
                    raise ValueError(f"Pair {i} missing required keys: {missing}")

                pairs.append(
                    SequenceLengthPair(
                        input_seq_len=int(pair_data["isl"]),
                        output_seq_len=int(pair_data["osl"]),
                        probability=_normalize_probability(float(pair_data["prob"])),
                    )
                )

            return SequenceLengthDistribution(pairs)

        except (KeyError, TypeError) as e:
            raise ValueError(f"Invalid JSON structure: {e}") from e

    @classmethod
    def _parse_bracket_format(cls, content: str) -> SequenceLengthDistribution:
        """Parse bracket format: (256,128):0.4,(512,256):0.6"""
        pairs = []

        for match in cls.BRACKET_PATTERN.finditer(content):
            isl, osl, prob = match.groups()
            pairs.append(
                SequenceLengthPair(
                    input_seq_len=int(isl),
                    output_seq_len=int(osl),
                    probability=_normalize_probability(float(prob)),
                )
            )

        if not pairs:
            raise ValueError("No valid pairs found in bracket format")

        return SequenceLengthDistribution(pairs)

    @classmethod
    def _parse_semicolon_format(cls, dist_str: str) -> SequenceLengthDistribution:
        """Parse semicolon format: 256,128:0.4;512,256:0.6"""
        pairs = []

        for pair_str in dist_str.split(";"):
            pair_str = pair_str.strip()
            if not pair_str:
                continue

            match = cls.SEMICOLON_PATTERN.fullmatch(pair_str)
            if not match:
                raise ValueError(
                    f"Invalid pair format: '{pair_str}'. Expected 'ISL,OSL:PROB'"
                )

            isl, osl, prob = match.groups()
            pairs.append(
                SequenceLengthPair(
                    input_seq_len=int(isl),
                    output_seq_len=int(osl),
                    probability=_normalize_probability(float(prob)),
                )
            )

        if not pairs:
            raise ValueError("No valid pairs found in semicolon format")

        return SequenceLengthDistribution(pairs)


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
