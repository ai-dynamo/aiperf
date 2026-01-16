# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Empirical distribution sampler for drawing from observed data distributions."""

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class EmpiricalSamplerStats:
    """Statistics about the learned empirical distribution.

    Attributes:
        min: Minimum value in original data.
        max: Maximum value in original data.
        mean: Mean of original data.
        median: Median of original data.
        num_unique: Number of unique values in distribution.
    """

    min: float
    max: float
    mean: float
    median: float
    num_unique: int


class EmpiricalSampler:
    """Samples values from an empirical distribution learned from data.

    Learns the cumulative distribution function (CDF) from input data and
    uses it to generate samples that match the empirical distribution.
    """

    def __init__(self, data: list[int] | list[float]) -> None:
        """Initialize sampler from observed data.

        Args:
            data: List of observed values to learn distribution from.
        """
        if not data:
            self._original_data = np.array([0])
            self._values = np.array([0])
            self._cdf = np.array([1.0])
            return

        # Store original data for statistics
        self._original_data = np.array(data)

        # Sort unique values and compute CDF
        sorted_data = np.sort(self._original_data)
        self._values, counts = np.unique(sorted_data, return_counts=True)
        self._cdf = np.cumsum(counts) / len(sorted_data)

    def sample(self, rng: np.random.Generator | None = None) -> int | float:
        """Draw a single sample from the learned distribution.

        Args:
            rng: Optional numpy random generator. If None, uses default random state.

        Returns:
            A value sampled from the empirical distribution, preserving original type.
        """
        if rng is None:
            rng = np.random.default_rng()

        # Sample uniform random value and map through CDF
        u = rng.uniform(0, 1)
        idx = np.searchsorted(self._cdf, u)
        idx = min(idx, len(self._values) - 1)

        # Convert numpy scalar to Python type
        return self._values[idx].item()

    def sample_batch(
        self, size: int, rng: np.random.Generator | None = None
    ) -> list[int | float]:
        """Draw multiple samples from the learned distribution.

        Args:
            size: Number of samples to draw.
            rng: Optional numpy random generator.

        Returns:
            List of sampled values.
        """
        return [self.sample(rng) for _ in range(size)]

    def get_stats(self) -> EmpiricalSamplerStats:
        """Get statistics about the learned distribution.

        Returns:
            EmpiricalSamplerStats with distribution statistics computed from original data.
        """
        return EmpiricalSamplerStats(
            min=float(np.min(self._original_data)),
            max=float(np.max(self._original_data)),
            mean=float(np.mean(self._original_data)),
            median=float(np.median(self._original_data)),
            num_unique=len(self._values),
        )
