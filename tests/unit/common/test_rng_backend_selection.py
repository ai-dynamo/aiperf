# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Env-var backend selection for the AIPerf RNG (``AIPERF_RNG_BACKEND``).

Verifies that ``rng.init``/``rng.derive`` honor the configured backend: ``legacy``
(default) yields the Python MT + NumPy ``RandomGenerator`` with SHA-256 derivation, while
``rust_parity`` yields the byte-exact ``ParityRandomGenerator`` with BLAKE3 derivation.

The backend is read from ``Environment.RNG.BACKEND`` at ``init`` time, so these tests
build a manager with the desired backend directly to avoid mutating the process-wide
``Environment`` singleton.
"""

from __future__ import annotations

import hashlib

import pytest

from aiperf.common import random_generator as rng
from aiperf.common.random_generator import RandomGenerator, _RNGManager
from aiperf.common.rng_parity import ParityRandomGenerator, RngRoot


@pytest.fixture(autouse=True)
def _reset_manager():
    rng.reset()
    yield
    rng.reset()


def test_default_backend_is_legacy():
    manager = _RNGManager(42)
    generator = manager.derive("dataset.loader")
    assert isinstance(generator, RandomGenerator)
    # Legacy derivation is SHA-256 of "{root}:{identifier}".
    expected = int.from_bytes(hashlib.sha256(b"42:dataset.loader").digest()[:8], "big")
    assert generator.seed == expected


def test_rust_parity_backend_uses_pcg64_and_blake3():
    manager = _RNGManager(42, backend="rust_parity")
    generator = manager.derive("dataset.loader")
    assert isinstance(generator, ParityRandomGenerator)
    # Parity derivation is BLAKE3, matching the Rust aiperf::rng contract.
    assert generator.seed == RngRoot(42).derive_seed("dataset.loader")


def test_rust_parity_seedless_is_entropy_backed():
    manager = _RNGManager(None, backend="rust_parity")
    a = manager.derive("dataset.loader")
    b = manager.derive("dataset.loader")
    assert isinstance(a, ParityRandomGenerator)
    assert a.seed is None and b.seed is None
    # Independent entropy streams: two seedless derivations differ.
    assert [a.random_u64(), a.random_u64()] != [b.random_u64(), b.random_u64()]


def test_rust_parity_derivation_is_reproducible():
    first = _RNGManager(7, backend="rust_parity").derive("timing.request_rate")
    second = _RNGManager(7, backend="rust_parity").derive("timing.request_rate")
    assert [first.random_u64() for _ in range(8)] == [
        second.random_u64() for _ in range(8)
    ]
