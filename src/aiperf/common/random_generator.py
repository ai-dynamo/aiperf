# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified random number generation for AIPerf.

This module provides RandomGenerator, a unified interface for random operations
that encapsulates both Python's random.Random and NumPy's Generator for optimal
performance and perfect reproducibility.

Architecture:
- RandomGenerator: Pure RNG class for random operations (see _random_generator_impl)
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

from __future__ import annotations

import hashlib
import random

import numpy as np

from aiperf.common._random_generator_impl import RandomGenerator
from aiperf.common.exceptions import InvalidStateError

__all__ = [
    "RandomGenerator",
    "derive",
    "init",
    "reset",
]


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


# Global RNG manager singleton. Module-level state is intentional: this is the
# process-wide root of deterministic seed derivation, initialized once from
# bootstrap.py and reset only in tests via reset().
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
        # Normalize seed to numpy's 32-bit range by folding high and low bits
        np_seed = (seed ^ (seed >> 32)) & 0xFFFFFFFF
        np.random.seed(np_seed)

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
