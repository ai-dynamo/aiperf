# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Core data models for sequence length distributions.

Split out of ``sequence_distribution`` to keep that module under the
ergonomics file-size limit. Public API is re-exported from
``aiperf.common.models.sequence_distribution`` for backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aiperf.common import random_generator as rng
from aiperf.common.aiperf_logger import AIPerfLogger

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


@dataclass(frozen=True)
class SequenceLengthPair:
    """Immutable representation of an ISL/OSL pair with probability weight and optional stddevs."""

    input_seq_len: int
    """Mean input sequence length (must be positive)."""

    output_seq_len: int
    """Mean output sequence length (must be positive)."""

    probability: float
    """Selection probability as a percentage in [0, 100]."""

    input_seq_len_stddev: float = 0.0
    """Standard deviation for input sequence length sampling (0 = fixed)."""

    output_seq_len_stddev: float = 0.0
    """Standard deviation for output sequence length sampling (0 = fixed)."""

    def __post_init__(self) -> None:
        """Validate sequence lengths, standard deviations, and probability on construction."""
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
        if self.input_seq_len_stddev < 0.0:
            raise ValueError(
                f"Input sequence length stddev must be non-negative, got {self.input_seq_len_stddev}"
            )
        if self.output_seq_len_stddev < 0.0:
            raise ValueError(
                f"Output sequence length stddev must be non-negative, got {self.output_seq_len_stddev}"
            )

    def __str__(self) -> str:
        if self.input_seq_len_stddev > 0 or self.output_seq_len_stddev > 0:
            return f"({self.input_seq_len}|{self.input_seq_len_stddev},{self.output_seq_len}|{self.output_seq_len_stddev}):{self.probability}%"
        else:
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

        # Lazily derive the RNG on first use. The CLI converter constructs
        # SequenceLengthDistribution during option parsing, which runs before
        # rng.init() in the bootstrap flow.
        self._rng_instance = None
        self._pairs = tuple(pairs)  # Immutable copy
        _validate_probability_sum(list(self._pairs))
        self._cumulative_probs = self._compute_cumulative_probabilities()

        logger.debug(f"Created distribution with {len(self._pairs)} pairs: {self}")

    @property
    def _rng(self):
        if self._rng_instance is None:
            self._rng_instance = rng.derive("models.sequence.distribution")
        return self._rng_instance

    def _compute_cumulative_probabilities(self) -> np.ndarray:
        """Compute cumulative probability distribution for efficient sampling."""
        # Convert percentages to fractions for internal calculation
        probs = [pair.probability / 100.0 for pair in self._pairs]
        return np.cumsum(probs, dtype=np.float64)

    def sample(self) -> tuple[int, int]:
        """
        Sample an (ISL, OSL) pair according to the distribution.

        Returns:
            Tuple of (input_seq_len, output_seq_len)
        """
        rand_val = self._rng.random()

        # Binary search for efficiency with large distributions
        idx = np.searchsorted(self._cumulative_probs, rand_val, side="right")
        idx = min(idx, len(self._pairs) - 1)  # Handle edge case

        pair = self._pairs[idx]

        # Sample from normal distribution if stddev is specified
        if pair.input_seq_len_stddev > 0:
            isl = self._rng.sample_positive_normal_integer(
                pair.input_seq_len, pair.input_seq_len_stddev
            )
        else:
            isl = pair.input_seq_len

        if pair.output_seq_len_stddev > 0:
            osl = self._rng.sample_positive_normal_integer(
                pair.output_seq_len, pair.output_seq_len_stddev
            )
        else:
            osl = pair.output_seq_len

        return (isl, osl)

    def sample_batch(self, batch_size: int) -> list[tuple[int, int]]:
        """
        Sample multiple (ISL, OSL) pairs efficiently.

        Args:
            batch_size: Number of pairs to sample

        Returns:
            List of (input_seq_len, output_seq_len) tuples
        """
        if batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {batch_size}")

        rand_vals = self._rng.random_batch(batch_size)
        indices = np.searchsorted(self._cumulative_probs, rand_vals, side="right")
        indices = np.clip(indices, 0, len(self._pairs) - 1)

        samples: list[tuple[int, int]] = []
        for idx in indices:
            pair = self._pairs[idx]
            if pair.input_seq_len_stddev > 0:
                isl = self._rng.sample_positive_normal_integer(
                    pair.input_seq_len, pair.input_seq_len_stddev
                )
            else:
                isl = pair.input_seq_len

            if pair.output_seq_len_stddev > 0:
                osl = self._rng.sample_positive_normal_integer(
                    pair.output_seq_len, pair.output_seq_len_stddev
                )
            else:
                osl = pair.output_seq_len

            samples.append((isl, osl))

        return samples

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
