# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HashIdRandomGenerator seed derivation (C6).

``from_base_rng`` must read ``base_rng.seed`` WITHOUT consuming RNG state --
the weka content synthesizer relies on that stability contract. The old falsy
``or`` fallback conflated the legal seed 0 with "unset", silently drawing a
seed and mutating the base RNG.
"""

from __future__ import annotations

from aiperf.common.hash_id_random_generator import HashIdRandomGenerator
from aiperf.common.random_generator import RandomGenerator


def _base_rng(seed: int | None) -> RandomGenerator:
    return RandomGenerator(seed, _internal=True)


def test_from_base_rng_seed_zero_preserved() -> None:
    gen = HashIdRandomGenerator.from_base_rng(_base_rng(0))
    assert gen.seed == 0


def test_from_base_rng_seed_zero_does_not_consume_base_state() -> None:
    base = _base_rng(0)
    HashIdRandomGenerator.from_base_rng(base)
    fresh = _base_rng(0)
    assert base.randrange(0, 2**64) == fresh.randrange(0, 2**64)


def test_from_base_rng_seed_zero_produces_deterministic_output() -> None:
    gen_a = HashIdRandomGenerator.from_base_rng(_base_rng(0))
    gen_b = HashIdRandomGenerator.from_base_rng(_base_rng(0))
    gen_a.reseed_for_hash_id(7, trace_id="trace")
    gen_b.reseed_for_hash_id(7, trace_id="trace")
    assert [gen_a.randrange(0, 10**6) for _ in range(5)] == [
        gen_b.randrange(0, 10**6) for _ in range(5)
    ]


def test_from_base_rng_seedless_base_draws_fallback_seed() -> None:
    gen = HashIdRandomGenerator.from_base_rng(_base_rng(None))
    assert gen.seed is not None
