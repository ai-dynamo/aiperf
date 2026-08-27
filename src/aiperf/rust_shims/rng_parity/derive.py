# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BLAKE3 seed algebra for order-independent random streams.

Ported from ``rust/aiperf/src/rng/derive.rs``. The stable contract is BLAKE3's first
eight digest bytes interpreted as a big-endian ``u64``. A component names its stream and
the child seed depends only on ``(root_seed, identifier)``, so derivation order never
perturbs any stream.

The Rust root formats the ``u64`` seed as canonical decimal (``itoa``) before hashing;
Python ``str(int)`` produces the identical bytes.
"""

from __future__ import annotations

import blake3

__all__ = [
    "RngRoot",
    "derive_seed_parts",
    "derive_seed_u64",
]

_M64 = (1 << 64) - 1


def derive_seed_parts(parts: list[bytes]) -> int:
    """BLAKE3 over the concatenated parts; first 8 digest bytes as big-endian ``u64``.

    Ported from ``derive.rs`` ``derive_seed_parts``. Concatenating the parts must
    produce exactly the same bytes as a single key passed to ``derive_seed_u64``.
    """
    hasher = blake3.blake3()
    for part in parts:
        hasher.update(part)
    digest = hasher.digest()
    return int.from_bytes(digest[:8], "big")


def derive_seed_u64(key: str) -> int:
    """Derive a ``u64`` seed from one UTF-8 key (``derive.rs`` ``derive_seed_u64``)."""
    return derive_seed_parts([key.encode("utf-8")])


class RngRoot:
    """Root seed for a reproducible run (``derive.rs`` ``RngRoot``).

    ``RngRoot(seed)`` with an ``int`` produces deterministic child streams; ``RngRoot(None)``
    makes every derived generator seed from OS entropy.
    """

    __slots__ = ("_seed",)

    def __init__(self, seed: int | None) -> None:
        self._seed = seed & _M64 if seed is not None else None

    @property
    def seed(self) -> int | None:
        """Return the underlying optional root seed."""
        return self._seed

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RngRoot) and other._seed == self._seed

    def __repr__(self) -> str:
        return f"RngRoot({self._seed!r})"

    def derive_seed(self, identifier: str) -> int | None:
        """Deterministic child seed for ``identifier`` (``derive.rs`` ``derive_seed``).

        Hashes ``f"{root}:{identifier}"`` bytes; ``None`` for a seedless root.
        """
        if self._seed is None:
            return None
        return derive_seed_parts(
            [str(self._seed).encode("utf-8"), b":", identifier.encode("utf-8")]
        )

    def derive_indexed_seed(self, identifier: str, index: int) -> int | None:
        """Deterministic seed for one indexed instance (``derive.rs``).

        Hashes ``f"{root}:{identifier}:{index}"``; ``None`` for a seedless root.
        """
        if self._seed is None:
            return None
        return derive_seed_parts(
            [
                str(self._seed).encode("utf-8"),
                b":",
                identifier.encode("utf-8"),
                b":",
                str(index).encode("utf-8"),
            ]
        )

    def derive_variation_seed(self, label: str) -> int | None:
        """Adaptive-sweep variation seed for ``label`` (``derive.rs``).

        Hashes ``f"{root}:variation:{label}"``.
        """
        if self._seed is None:
            return None
        return derive_seed_parts(
            [str(self._seed).encode("utf-8"), b":variation:", label.encode("utf-8")]
        )

    def derive_root(self, identifier: str) -> RngRoot:
        """Derive a child root for a named subsystem (seedless stays seedless)."""
        return RngRoot(self.derive_seed(identifier))

    def derive_indexed_root(self, identifier: str, index: int) -> RngRoot:
        """Derive a child root for one indexed instance of a named subsystem."""
        return RngRoot(self.derive_indexed_seed(identifier, index))

    def derive(self, identifier: str):
        """Derive an owned generator for ``identifier``.

        Imported lazily to avoid a module import cycle with ``generator``.
        """
        from aiperf.rust_shims.rng_parity.generator import ParityRandomGenerator

        return ParityRandomGenerator.from_seed(self.derive_seed(identifier))

    def derive_seed_or_entropy(self, identifier: str) -> int:
        """Return a deterministic derived seed or one fresh entropy-backed value."""
        seed = self.derive_seed(identifier)
        if seed is not None:
            return seed
        return self.derive(identifier).random_u64()
