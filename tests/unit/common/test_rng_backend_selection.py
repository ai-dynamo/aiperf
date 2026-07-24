# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AIPerf random-number backend selection tests."""

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


def test_default_backend_is_python():
    manager = _RNGManager(42)
    generator = manager.derive("dataset.loader")
    assert isinstance(generator, RandomGenerator)
    expected = int.from_bytes(hashlib.sha256(b"42:dataset.loader").digest()[:8], "big")
    assert generator.seed == expected


def test_rust_backend_uses_pcg64_and_blake3():
    manager = _RNGManager(42, backend="rust")
    generator = manager.derive("dataset.loader")
    assert isinstance(generator, ParityRandomGenerator)
    assert generator.seed == RngRoot(42).derive_seed("dataset.loader")


def test_rust_backend_seedless_is_entropy_backed():
    manager = _RNGManager(None, backend="rust")
    a = manager.derive("dataset.loader")
    b = manager.derive("dataset.loader")
    assert isinstance(a, ParityRandomGenerator)
    assert a.seed is None and b.seed is None
    assert [a.random_u64(), a.random_u64()] != [b.random_u64(), b.random_u64()]


def test_rust_backend_derivation_is_reproducible():
    first = _RNGManager(7, backend="rust").derive("timing.request_rate")
    second = _RNGManager(7, backend="rust").derive("timing.request_rate")
    assert [first.random_u64() for _ in range(8)] == [
        second.random_u64() for _ in range(8)
    ]
