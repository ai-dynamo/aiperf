# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Byte-exact pure-Python port of ``rand_pcg::Pcg64`` (``Lcg128Xsl64``).

Ported from:
- ``rand_pcg-0.9.0/src/pcg128.rs`` (Lcg128Xsl64: ``from_seed``, ``from_state_incr``,
  ``step``, ``next_u64``, ``output_xsl_rr``).
- ``rand_core-0.9.5/src/lib.rs:466`` (``SeedableRng::seed_from_u64`` default: PCG32
  expansion of a ``u64`` into the 32-byte seed).
- ``rand_core::impls::fill_bytes_via_next`` (``fill_bytes``).

All 128-/64-bit arithmetic is done on Python ``int`` and masked back to width, exactly
reproducing Rust's ``wrapping_*`` semantics. This is the single source of raw ``u64``
draws for the parity generator; every higher-level draw is built on ``next_u64``.
"""

from __future__ import annotations

__all__ = ["Pcg64"]

# Default PCG multiplier for 128-bit state (pcg128.rs `MULTIPLIER`).
_MULTIPLIER = 0x2360ED051FC65DA44385DF649FCCF645
_M128 = (1 << 128) - 1
_M64 = (1 << 64) - 1
_M32 = (1 << 32) - 1

# Constants from `seed_from_u64`'s inner PCG32 (rand_core lib.rs:469-470).
_SEED_MUL = 6364136223846793005
_SEED_INC = 11634580027462260723


def _rotr32(value: int, rot: int) -> int:
    """32-bit rotate-right (``u32::rotate_right``)."""
    rot &= 31
    return ((value >> rot) | (value << (32 - rot))) & _M32 if rot else value & _M32


def _rotr64(value: int, rot: int) -> int:
    """64-bit rotate-right (``u64::rotate_right``)."""
    rot &= 63
    return ((value >> rot) | (value << (64 - rot))) & _M64 if rot else value & _M64


def seed_from_u64(state: int) -> bytes:
    """Expand a ``u64`` into a 32-byte seed via PCG32 (rand_core lib.rs:466).

    Reproduces the default ``SeedableRng::seed_from_u64``: eight little-endian ``u32``
    chunks, each advancing an internal PCG32 state.
    """
    state &= _M64
    out = bytearray()
    for _ in range(8):  # 32 bytes / 4 bytes per u32 chunk
        state = (state * _SEED_MUL + _SEED_INC) & _M64
        xorshifted = (((state >> 18) ^ state) >> 27) & _M32
        rot = (state >> 59) & _M32
        x = _rotr32(xorshifted, rot)
        out += x.to_bytes(4, "little")
    return bytes(out)


def output_xsl_rr(state: int) -> int:
    """PCG-XSL-RR output function (pcg128.rs ``output_xsl_rr``)."""
    rot = (state >> 122) & _M32
    xsl = ((state >> 64) ^ state) & _M64
    return _rotr64(xsl, rot)


class Pcg64:
    """PCG64 (``Lcg128Xsl64``): 128-bit LCG with the XSL-RR 64-bit output function."""

    __slots__ = ("_state", "_increment")

    def __init__(self, state: int, increment: int) -> None:
        self._state = state & _M128
        self._increment = increment & _M128

    @classmethod
    def from_seed(cls, seed: bytes) -> Pcg64:
        """Construct from a 32-byte seed (pcg128.rs ``from_seed``)."""
        if len(seed) != 32:
            raise ValueError("Pcg64 seed must be exactly 32 bytes")
        s0 = int.from_bytes(seed[0:8], "little")
        s1 = int.from_bytes(seed[8:16], "little")
        s2 = int.from_bytes(seed[16:24], "little")
        s3 = int.from_bytes(seed[24:32], "little")
        state = s0 | (s1 << 64)
        incr = s2 | (s3 << 64)
        return cls._from_state_incr(state, incr | 1)

    @classmethod
    def from_u64_seed(cls, seed: int) -> Pcg64:
        """Construct as ``Pcg64::seed_from_u64(seed)`` does (the AIPerf seeding path)."""
        return cls.from_seed(seed_from_u64(seed))

    @classmethod
    def _from_state_incr(cls, state: int, increment: int) -> Pcg64:
        pcg = cls(0, increment)
        # Move away from the initial value (pcg128.rs `from_state_incr`).
        pcg._state = (state + increment) & _M128
        pcg._step()
        return pcg

    def _step(self) -> None:
        self._state = (self._state * _MULTIPLIER + self._increment) & _M128

    def next_u64(self) -> int:
        """Advance one step and return the 64-bit output (pcg128.rs ``next_u64``)."""
        self._step()
        return output_xsl_rr(self._state)

    def fill_bytes(self, length: int) -> bytes:
        """Fill ``length`` bytes from successive ``next_u64`` draws (little-endian).

        Reproduces ``rand_core::impls::fill_bytes_via_next``: full 8-byte chunks then a
        truncated tail chunk.
        """
        out = bytearray()
        remaining = length
        while remaining >= 8:
            out += self.next_u64().to_bytes(8, "little")
            remaining -= 8
        if remaining > 0:
            out += self.next_u64().to_bytes(8, "little")[:remaining]
        return bytes(out)

    def clone(self) -> Pcg64:
        """Return an independent copy with identical internal state."""
        return Pcg64(self._state, self._increment)
