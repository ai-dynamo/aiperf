# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hash-ID-scoped RNG for order-independent parallel trace synthesis.

Ported from ``rust/aiperf/src/rng/hash_id.rs``. A base seed is preserved without
consuming state when present (seed ``0`` is legal), and each ``(trace_id, hash_id)`` pair
deterministically reseeds the inner generator via the BLAKE3 algebra so worker scheduling
cannot perturb generated content.
"""

from __future__ import annotations

from aiperf.common.rng_parity.derive import derive_seed_parts
from aiperf.common.rng_parity.generator import ParityRandomGenerator

__all__ = ["HashIdRandomGenerator"]


class HashIdRandomGenerator:
    """Random generator that reseeds per ``(trace_id, hash_id)`` scope."""

    __slots__ = ("_base_seed", "_trace_id", "_generator")

    def __init__(
        self, base_seed: int, trace_id: str, generator: ParityRandomGenerator
    ) -> None:
        self._base_seed = base_seed
        self._trace_id = trace_id
        self._generator = generator

    @classmethod
    def from_base(cls, base: ParityRandomGenerator) -> HashIdRandomGenerator:
        """Build from a base generator (``hash_id.rs`` ``from_base``).

        A seeded base's seed (including ``0``) is read without consuming state; a seedless
        base consumes one ``u64`` fallback and becomes deterministic from there.
        """
        base_seed = base.seed if base.seed is not None else base.random_u64()
        return cls(
            base_seed,
            "",
            ParityRandomGenerator.from_seed(base_seed),
        )

    @property
    def base_seed(self) -> int:
        """Return the base seed used in hash-id derivation."""
        return self._base_seed

    @property
    def trace_id(self) -> str:
        """Return the current instance trace scope."""
        return self._trace_id

    def set_trace_id(self, trace_id: str) -> None:
        """Set the instance trace scope used when no override is passed."""
        self._trace_id = trace_id

    def reseed_for_hash_id(
        self, hash_id: int, trace_id_override: str | None = None
    ) -> None:
        """Reseed the inner generator for ``hash_id`` in the selected trace scope.

        ``trace_id_override`` applies only to this call; ``None`` uses the instance scope
        (default empty string = the content-global namespace).
        """
        scope = trace_id_override if trace_id_override is not None else self._trace_id
        seed = derive_seed_parts(
            [
                str(self._base_seed).encode("utf-8"),
                b":",
                scope.encode("utf-8"),
                b":",
                str(hash_id).encode("utf-8"),
            ]
        )
        self._generator.reseed(seed)

    @property
    def generator(self) -> ParityRandomGenerator:
        """Borrow the inner generator."""
        return self._generator

    def __getattr__(self, name):
        # Deref to the inner generator (mirrors Rust Deref/DerefMut).
        return getattr(self._generator, name)
