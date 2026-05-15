# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RandomGenerator implementation.

Internal module. Import ``RandomGenerator`` from
``aiperf.common.random_generator`` instead of this module.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import Any

import numpy as np

from aiperf.common.aiperf_logger import AIPerfLogger

_logger = AIPerfLogger(__name__)


class RandomGenerator:
    """Unified random number generator that encapsulates both Python random and NumPy RNG.

    This class provides a consistent interface for random operations using both
    Python's random.Random and NumPy's Generator for optimal performance.

    Instances should be obtained via rng.derive() with a unique identifier rather
    than constructed directly. This ensures deterministic seed derivation and
    reproducible random sequences.

    Note:
        Instances are NOT thread-safe. Each instance maintains independent mutable
        state and should not be shared across threads or concurrent async tasks.
    """

    def __init__(self, seed: int | None = None, *, _internal: bool = False):
        """Initialize random generator with optional seed.

        Note:
            Do not construct RandomGenerator directly. Use rng.derive(identifier)
            to obtain instances through the managed derivation system.

        Args:
            seed: Optional random seed (0 to 2^64-1). If None, generator uses
                  non-deterministic entropy from OS. The same seed guarantees
                  identical random sequences across program runs.
            _internal: Internal flag - must be True to construct. This prevents
                      direct construction and enforces use of rng.derive().

        Raises:
            RuntimeError: If _internal is False (direct construction attempt).

        Note:
            Instances are NOT thread-safe. Do not share across threads/async tasks.
            Each instance maintains independent mutable state.
        """
        if not _internal:
            raise RuntimeError(
                "RandomGenerator should not be constructed directly. "
                "Use rng.derive('your.identifier') to obtain instances with "
                "properly derived seeds for reproducibility."
            )

        self._seed = seed
        self._python_rng = random.Random(seed)
        self._numpy_rng = np.random.default_rng(seed)

    def __repr__(self) -> str:
        """Return string representation for debugging."""
        return f"RandomGenerator(seed={self._seed})"

    @property
    def seed(self) -> int | None:
        """Get the seed used to initialize this generator."""
        return self._seed

    def integers(
        self,
        low: int,
        high: int | None = None,
        size: int | tuple[int, ...] | None = None,
        dtype: type = np.int64,
    ) -> Any:
        """Generate random integers from [low, high) using NumPy.

        Args:
            low: Lowest integer (inclusive), or if high is None, then [0, low)
            high: Highest integer (exclusive), optional
            size: Output shape, optional
            dtype: Desired NumPy dtype for the result (default: np.int64)

        Returns:
            Random integer or array of integers
        """
        return self._numpy_rng.integers(low, high, size=size, dtype=dtype)

    def choice(self, seq: Sequence[Any]) -> Any:
        """Select random element from non-empty sequence.

        Args:
            seq: Non-empty sequence to choose from

        Returns:
            Randomly selected element

        Raises:
            IndexError: If sequence is empty
        """
        return self._python_rng.choice(seq)

    def randrange(self, *args: int) -> int:
        """Generate random integer from range (start, stop[, step]).

        Args:
            *args: Same as range() - (stop) or (start, stop) or (start, stop, step)

        Returns:
            Random integer from specified range
        """
        return self._python_rng.randrange(*args)

    def randint(self, a: int, b: int) -> int:
        """Generate random integer N such that a <= N <= b (inclusive).

        Args:
            a: Lower bound (inclusive)
            b: Upper bound (inclusive)

        Returns:
            Random integer in [a, b]

        Note:
            Unlike randrange, this includes the upper bound.
        """
        return self._python_rng.randint(a, b)

    def uniform(self, a: float, b: float) -> float:
        """Generate random float N such that a <= N <= b.

        Args:
            a: Lower bound
            b: Upper bound

        Returns:
            Random float in [a, b] or [b, a] if b < a
        """
        return self._python_rng.uniform(a, b)

    def choices(self, population: Sequence[Any], k: int) -> list[Any]:
        """Select k elements with replacement.

        Args:
            population: Sequence to sample from
            k: Number of elements to select

        Returns:
            List of k elements (with replacement)
        """
        return self._python_rng.choices(population, k=k)

    def sample(self, population: Sequence[Any], k: int) -> list[Any]:
        """Select k unique elements without replacement.

        Args:
            population: Sequence to sample from (must have len >= k)
            k: Number of unique elements to select

        Returns:
            List of k unique elements

        Raises:
            ValueError: If k > len(population)
        """
        return self._python_rng.sample(population, k=k)

    def numpy_choice(
        self,
        a: Any,
        size: int | tuple[int, ...] | None = None,
        p: np.ndarray | None = None,
        replace: bool = True,
    ) -> Any:
        """NumPy random choice from array.

        Args:
            a: Array-like or int (if int, choose from range(a))
            size: Output shape, optional
            p: Probabilities for each entry in a, optional
            replace: Whether to sample with replacement, default True

        Returns:
            Random sample(s) from array
        """
        return self._numpy_rng.choice(a, size=size, p=p, replace=replace)

    def normal(
        self,
        loc: float = 0.0,
        scale: float = 1.0,
        size: int | tuple[int, ...] | None = None,
    ) -> Any:
        """Draw samples from normal (Gaussian) distribution.

        Args:
            loc: Mean ("center") of distribution, default 0.0
            scale: Standard deviation, default 1.0
            size: Output shape, optional

        Returns:
            Random sample(s) from normal distribution
        """
        return self._numpy_rng.normal(loc, scale, size)

    def sample_normal(
        self, mean: float, stddev: float, lower: float = -np.inf, upper: float = np.inf
    ) -> float:
        """Sample from bounded normal distribution using rejection sampling.

        Args:
            mean: Mean of the normal distribution
            stddev: Standard deviation of the normal distribution
            lower: Lower bound (inclusive), default -inf
            upper: Upper bound (inclusive), default +inf

        Returns:
            Sample from normal distribution clamped to [lower, upper]

        Raises:
            ValueError: If lower > upper (impossible constraint)

        Note:
            Uses rejection sampling with max 10,000 iterations. If bounds are
            unreachable (e.g., >10 stddevs from mean), falls back to clamped mean.
            Uses Python's gauss() for optimal scalar performance (~6x faster than NumPy).
        """
        if lower > upper:
            raise ValueError(
                f"Invalid bounds: lower ({lower}) > upper ({upper}). "
                "Bounds must satisfy lower <= upper."
            )

        # Rejection sampling with iteration limit to prevent infinite loops
        # Use Python's gauss() for scalar sampling (~6x faster than NumPy's normal())
        max_iterations = 10000
        for _ in range(max_iterations):
            n = self._python_rng.gauss(mean, stddev)
            if lower <= n <= upper:
                return n

        # Fallback if rejection sampling fails (bounds unreachable)
        _logger.warning(
            f"Rejection sampling failed for normal distribution with mean {mean} and stddev {stddev}. "
            f"Falling back to clamped mean {mean}."
        )
        return max(lower, min(upper, mean))

    def sample_positive_normal(self, mean: float, stddev: float) -> float:
        """Sample positive value from normal distribution (lower bound = 0)."""
        if mean < 0:
            raise ValueError(f"Mean value ({mean}) should be greater than 0")
        return self.sample_normal(mean, stddev, lower=0)

    def sample_positive_normal_integer(self, mean: float, stddev: float) -> int:
        """Sample positive integer from normal distribution (minimum 1).

        Args:
            mean: Mean of the normal distribution
            stddev: Standard deviation. If <= 0, returns mean as integer (min 1).

        Returns:
            Positive integer >= 1 sampled from normal distribution

        Note:
            Uses ceiling to ensure result is always >= 1 even when sample
            approaches 0. For stddev <= 0, returns max(1, round(mean)).
        """
        if stddev <= 0:
            return max(1, round(mean))
        return max(1, math.ceil(self.sample_positive_normal(mean, stddev)))

    def expovariate(self, lambd: float) -> float:
        """Generate exponentially distributed random number.

        Args:
            lambd: Lambda parameter (lambd = 1.0 / desired mean)

        Returns:
            Random float from exponential distribution

        Note:
            For desired mean of X, use lambd = 1.0 / X
        """
        return self._python_rng.expovariate(lambd)

    def gammavariate(self, alpha: float, beta: float) -> float:
        """Generate gamma distributed random number.

        Args:
            alpha: Shape parameter (must be > 0). Controls the distribution shape:
                   - alpha = 1.0: Exponential distribution (equivalent to expovariate)
                   - alpha < 1.0: More bursty/clustered arrivals
                   - alpha > 1.0: More regular/smooth arrivals
            beta: Scale parameter (must be > 0). For rate-based arrivals,
                  use beta = 1.0 / (rate * alpha) to maintain the target mean.

        Returns:
            Random float from gamma distribution

        Note:
            The mean of the distribution is alpha * beta.
            For arrival intervals at a given rate with tunable smoothness:
                interval = gammavariate(smoothness, 1.0 / (rate * smoothness))
        """
        return self._python_rng.gammavariate(alpha, beta)

    def zipf(self, alpha: float) -> int:
        """Generate Zipf-distributed random integer >= 1.

        Args:
            alpha: Exponent parameter (must be > 1). Higher values concentrate
                   more probability on small ranks (value 1 dominates).

        Returns:
            Random positive integer from Zipf distribution.
        """
        return int(self._numpy_rng.zipf(alpha))

    def random(self) -> float:
        """Generate random float in [0.0, 1.0).

        Returns:
            Random float from uniform distribution
        """
        return self._python_rng.random()

    def shuffle(self, x: list) -> None:
        """Shuffle list in-place using Fisher-Yates algorithm.

        Args:
            x: List to shuffle (modified in-place)

        Note:
            Modifies the input list directly, returns None.
            Uses NumPy's shuffle for ~6x better performance.
        """
        self._numpy_rng.shuffle(x)

    def random_batch(self, size: int) -> np.ndarray:
        """Generate array of random floats in [0.0, 1.0) using NumPy.

        Args:
            size: Number of random floats to generate

        Returns:
            NumPy array of random floats
        """
        return self._numpy_rng.random(size)
