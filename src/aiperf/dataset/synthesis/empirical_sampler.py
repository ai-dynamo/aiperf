# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Empirical distribution sampler for drawing from observed data distributions."""

from typing import Any

import numpy as np


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
            self._values = np.array([0])
            self._cdf = np.array([1.0])
            return

        # Sort unique values and compute CDF
        sorted_data = np.sort(np.array(data))
        self._values, counts = np.unique(sorted_data, return_counts=True)
        self._cdf = np.cumsum(counts) / len(sorted_data)

    def sample(self, rng: np.random.Generator | None = None) -> Any:
        """Draw a single sample from the learned distribution.

        Args:
            rng: Optional numpy random generator. If None, uses default random state.

        Returns:
            A value sampled from the empirical distribution.
        """
        if rng is None:
            rng = np.random.default_rng()

        # Sample uniform random value and map through CDF
        u = rng.uniform(0, 1)
        idx = np.searchsorted(self._cdf, u)
        idx = min(idx, len(self._values) - 1)

        return int(self._values[idx])

    def sample_batch(
        self, size: int, rng: np.random.Generator | None = None
    ) -> list[Any]:
        """Draw multiple samples from the learned distribution.

        Args:
            size: Number of samples to draw.
            rng: Optional numpy random generator.

        Returns:
            List of sampled values.
        """
        return [self.sample(rng) for _ in range(size)]

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the learned distribution.

        Returns:
            Dictionary with distribution statistics.
        """
        return {
            "min": int(self._values[0]),
            "max": int(self._values[-1]),
            "mean": float(np.mean(self._values)),
            "median": float(np.median(self._values)),
            "num_unique": len(self._values),
        }
