# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hash-ID-based random generator for parallel processing with reproducibility.

Enables parallel processing of traces with hash_ids while maintaining
reproducibility. Each (trace_id, hash_id) pair produces a deterministic random
sequence regardless of worker count or processing order.

Architecture:
    Global Seed -> Base RNG -> (trace_id, hash_id) -> Deterministic tokens

The trace_id (typically a content hash of the trace file) ensures that different
trace files with overlapping hash_id values produce different content, while the
same trace file always produces identical results.
"""

import hashlib

from aiperf.common.random_generator import RandomGenerator

__all__ = ["HashIdRandomGenerator"]


class _DisabledNumpyRNG:
    """Raises on any attribute access to prevent NumPy RNG usage."""

    def __getattr__(self, name):
        raise RuntimeError(
            "HashIdRandomGenerator does not support NumPy RNG operations. "
            "Use Python RNG methods (randrange, choice, etc.) instead."
        )


class HashIdRandomGenerator(RandomGenerator):
    """RandomGenerator that re-seeds deterministically per (trace_id, hash_id).

    Designed for parallel processing where multiple workers need to generate
    identical content for the same hash_id within a trace file.

    Thread Safety:
        NOT thread-safe. Each worker process must have its own instance.
    """

    @classmethod
    def from_base_rng(cls, base_rng: RandomGenerator) -> "HashIdRandomGenerator":
        """Create from a base RandomGenerator (typically from rng.derive())."""
        base_seed = base_rng.seed or base_rng.randrange(0, 2**64)
        return cls(base_seed, _internal=True)

    def __init__(self, base_seed: int, *, _internal: bool = False):
        super().__init__(base_seed, _internal=_internal)
        self._numpy_rng = _DisabledNumpyRNG()
        self._trace_id: str = ""

    def set_trace_id(self, trace_id: str) -> None:
        """Set trace identifier to scope hash_ids to a specific trace file.

        Args:
            trace_id: Content hash or unique identifier for the trace file.
                      Different trace files must use different trace_ids.
        """
        self._trace_id = trace_id

    def reseed_for_hash_id(self, hash_id: int) -> None:
        """Re-seed RNG deterministically for a specific hash_id.

        After calling, all random operations use the derived seed until
        the next reseed_for_hash_id call.

        Args:
            hash_id: KV block hash ID from trace data.
        """
        seed_bytes = hashlib.sha256(
            f"{self.seed}:{self._trace_id}:{hash_id}".encode()
        ).digest()
        self._python_rng.seed(int.from_bytes(seed_bytes[:8], "big"))
