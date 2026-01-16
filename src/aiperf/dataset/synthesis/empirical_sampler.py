# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Empirical distribution sampler for drawing from observed data distributions."""

from dataclasses import dataclass
from typing import cast

import numpy as np

from aiperf.common import random_generator as rng


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
    """Samples values from an empirical distribution learned from data."""

    def __init__(self, data: list[int] | list[float]) -> None:
        """Initialize sampler from observed data.

        Args:
            data: List of observed values to learn distribution from.
        """
        if not data:
            self._values = np.array([0])
            self._probs = np.array([1.0])
        else:
            self._values, counts = np.unique(data, return_counts=True)
            self._probs = counts / counts.sum()
        self._rng = rng.derive("dataset.synthesis.empirical_sampler")

    def sample(self) -> int | float:
        """Draw a single sample from the learned distribution.

        Returns:
            A value sampled from the empirical distribution.
        """
        # Cast needed: numpy's choice() return type varies by size param, but
        # type stubs don't capture this overload (returns scalar when size=None)
        result = cast(np.intp, self._rng.numpy_choice(self._values, p=self._probs))
        return result.item()

    def sample_batch(self, size: int) -> list[int | float]:
        """Draw multiple samples from the learned distribution.

        Args:
            size: Number of samples to draw.

        Returns:
            List of sampled values.
        """
        # Cast needed: numpy's choice() return type varies by size param, but
        # type stubs don't capture this overload (returns ndarray when size given)
        result = cast(
            np.ndarray, self._rng.numpy_choice(self._values, size=size, p=self._probs)
        )
        return result.tolist()

    def get_stats(self) -> EmpiricalSamplerStats:
        """Get statistics about the learned distribution.

        Returns:
            EmpiricalSamplerStats with distribution statistics.
        """
        return EmpiricalSamplerStats(
            min=float(self._values.min()),
            max=float(self._values.max()),
            mean=float(np.average(self._values, weights=self._probs)),
            median=float(np.median(self._values)),
            num_unique=len(self._values),
        )
