# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified random number generation for AIPerf.

This module provides RandomGenerator, a unified interface for random operations
that encapsulates both Python's random.Random and NumPy's Generator for optimal
performance and perfect reproducibility.

Architecture:
- RandomGenerator: Pure RNG class for random operations
- _RNGManager: Internal manager for deterministic seed derivation
- Module functions: Clean API for initialization and RNG derivation

Key features:
- Hash-based deterministic seed derivation
- Order-independent child RNG creation
- Support for both deterministic and non-deterministic modes
- Cross-run stability and reproducibility

Why Both Python and NumPy RNG?
The dual backend design provides optimal performance:
- Python's random.Random: Efficient for scalar ops (choice, randint, gauss, etc.)
- NumPy's Generator: Efficient for array ops (normal, shuffle, batch generation)

**Thread Safety:**
RandomGenerator instances are NOT thread-safe. Each maintains mutable state and
should not be shared across threads or async tasks. Obtain independent instances
via rng.derive() for each component.

Usage:
    >>> from aiperf.common import random_generator as rng
    >>>
    >>> # Initialize once at startup
    >>> rng.init(42)  # or None for non-deterministic
    >>>
    >>> # Derive child RNGs in component __init__
    >>> class MyComponent:
    ...     def __init__(self):
    ...         self._rng = rng.derive("my_module.my_component")
    ...
    ...     def do_something(self):
    ...         value = self._rng.choice([1, 2, 3, 4, 5])
    ...         sample = self._rng.sample_positive_normal_integer(100, 10)
"""

import hashlib
import math
import random

import numpy as np

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.exceptions import InvalidStateError

__all__ = [
    "RandomGenerator",
    "derive",
    "init",
    "reset",
]

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

    def integers(self, low: int, high: int | None = None, size=None):
        """Generate random integers from [low, high) using NumPy.

        Args:
            low: Lowest integer (inclusive), or if high is None, then [0, low)
            high: Highest integer (exclusive), optional
            size: Output shape, optional

        Returns:
            Random integer or array of integers
        """
        return self._numpy_rng.integers(low, high, size=size)

    def choice(self, seq):
        """Select random element from non-empty sequence.

        Args:
            seq: Non-empty sequence to choose from

        Returns:
            Randomly selected element

        Raises:
            IndexError: If sequence is empty
        """
        return self._python_rng.choice(seq)

    def randrange(self, *args):
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

    def choices(self, population, k: int):
        """Select k elements with replacement.

        Args:
            population: Sequence to sample from
            k: Number of elements to select

        Returns:
            List of k elements (with replacement)
        """
        return self._python_rng.choices(population, k=k)

    def sample(self, population, k: int):
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

    def numpy_choice(self, a, size=None):
        """NumPy random choice from array.

        Args:
            a: Array-like or int (if int, choose from range(a))
            size: Output shape, optional

        Returns:
            Random sample(s) from array
        """
        return self._numpy_rng.choice(a, size=size)

    def normal(self, loc=0.0, scale=1.0, size=None):
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


class _RNGManager:
    """Internal manager for RNG seed derivation.

    Handles deterministic seed derivation from a root seed, allowing
    hierarchical child RNG creation with reproducible seeds.
    """

    def __init__(self, root_seed: int | None):
        """Initialize the RNG manager.

        Args:
            root_seed: Root seed for derivation. If None, all derived RNGs
                      will be non-deterministic (seeded with None).
        """
        self._root_seed = root_seed

    def derive(self, identifier: str) -> RandomGenerator:
        """Derive a child RNG with deterministic seed from identifier.

        Args:
            identifier: Unique dotted identifier (e.g., "dataset.loader").

        Returns:
            New RandomGenerator with derived seed (or None if root is None).

        Note:
            Same identifier always produces same derived seed, ensuring
            reproducible sequences. Uses SHA-256 for stable hashing.
        """
        if self._root_seed is not None:
            # Deterministic: derive seed from root + identifier
            seed_string = f"{self._root_seed}:{identifier}"
            hash_bytes = hashlib.sha256(seed_string.encode("utf-8")).digest()
            child_seed = int.from_bytes(hash_bytes[:8], byteorder="big")
            return RandomGenerator(child_seed, _internal=True)
        else:
            # Non-deterministic: pass through None
            return RandomGenerator(None, _internal=True)


# Global RNG manager instance
_manager: _RNGManager | None = None


def init(seed: int | None) -> None:
    """Initialize global RNG manager. Called once at startup (bootstrap.py).

    Args:
        seed: Root seed (0 to 2^64-1) for deterministic behavior, or None
              for non-deterministic behavior. All derived RNGs will inherit
              this deterministic/non-deterministic property.

    Raises:
        InvalidStateError: If global RNG manager has already been initialized.

    Note:
        Also sets global random seeds for Python's random and NumPy as a defensive
        measure. This ensures reproducibility even if third-party libraries or
        future code inadvertently uses global random state.
    """
    global _manager
    if _manager is not None:
        raise InvalidStateError(
            "Global RNG manager has already been initialized. Call rng.reset() first."
        )

    # Set global seeds defensively for reproducibility
    # This protects against third-party code or future changes that might use global state
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    _manager = _RNGManager(seed)


def derive(identifier: str) -> RandomGenerator:
    """Derive a child RNG with deterministic seed from the identifier.

    This is the primary way to obtain RandomGenerator instances in your code.
    Store the result in __init__ and reuse it for all random operations.

    Args:
        identifier: Unique dotted identifier for this component (e.g., "dataset.loader").
                    Use hierarchical naming matching your component structure.

    Returns:
        New child RandomGenerator with deterministic seed derived from identifier.

    Raises:
        InvalidStateError: If global RNG manager has not been initialized.

    Example:
        >>> from aiperf.common import random_generator as rng
        >>>
        >>> class MyComponent:
        ...     def __init__(self):
        ...         self._rng = rng.derive("my_module.my_component")
        ...
        ...     def process(self):
        ...         return self._rng.choice([1, 2, 3])

    Note:
        The same identifier always produces the same seed, ensuring reproducible
        random sequences across runs when using the same global seed.
    """
    if _manager is None:
        raise InvalidStateError(
            "Global RNG manager has not been initialized. Call rng.init() first."
        )

    return _manager.derive(identifier)


def reset() -> None:
    """Reset global RNG manager to None.

    This is intended for testing and bootstrap.py only. After calling this,
    you must call rng.init() before using rng.derive() again.

    Note:
        This does not affect existing child RNG instances - they continue to
        function independently with their own state.
    """
    global _manager
    _manager = None
