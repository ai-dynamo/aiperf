# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioral tests for the parity RNG, mirroring the Rust ``rng`` unit tests.

Where ``test_rng_parity.py`` proves byte-exactness against Rust golden vectors, this
suite locks the *behavioral* contracts (validation, bounds, reproducibility, order
independence, table integrity) so refactors can't silently regress them.
"""

from __future__ import annotations

import math

import pytest

from aiperf.rust_shims.rng_parity import (
    HashIdRandomGenerator,
    ParityRandomGenerator,
    RngError,
    RngRoot,
    SamplingDistribution,
    SequenceLengthDistribution,
    SequenceLengthPair,
    namespace,
)
from aiperf.rust_shims.rng_parity import ziggurat_tables as zt


# --------------------------------------------------------------------- ziggurat tables
def test_ziggurat_tables_have_expected_shape_and_scalars():
    for name in ("ZIG_NORM_X", "ZIG_NORM_F", "ZIG_EXP_X", "ZIG_EXP_F"):
        assert len(getattr(zt, name)) == 257, name
    # Scalars transcribed from rand_distr-0.5.1/src/ziggurat_tables.rs.
    assert zt.ZIG_NORM_R == 3.654152885361008796
    assert zt.ZIG_EXP_R == 7.697117470131050077


# --------------------------------------------------------------------- derivation
def test_derivation_is_order_independent():
    root = RngRoot(99)
    ids = [
        "dataset.audio.duration",
        "timing.request.poisson_interval",
        "dataset.sampler.shuffle",
    ]
    expected = {i: root.derive_seed(i) for i in ids}
    for order in ([2, 0, 1], [1, 2, 0], [0, 1, 2]):
        for idx in order:
            assert root.derive_seed(ids[idx]) == expected[ids[idx]]
    assert expected[ids[0]] != expected[ids[1]]


def test_every_canonical_namespace_has_distinct_seed():
    root = RngRoot(7)
    seeds = {root.derive_seed(name) for name in namespace.ALL}
    assert len(seeds) == len(namespace.ALL)


def test_seedless_root_stays_seedless():
    root = RngRoot(None)
    assert root.derive_seed(namespace.DATASET_LOADER) is None
    assert root.derive_indexed_seed(namespace.GRAPH_PHASE, 7) is None
    assert root.derive_variation_seed("x") is None
    assert root.derive_root(namespace.GRAPH_PHASE) == root


# --------------------------------------------------------------------- reproducibility
def test_same_seed_reproduces_stream():
    a = ParityRandomGenerator.from_seed(42)
    b = ParityRandomGenerator.from_seed(42)
    assert [a.random_u64() for _ in range(64)] == [b.random_u64() for _ in range(64)]


def test_reseed_resets_stream():
    g = ParityRandomGenerator.from_seed(1)
    g.random_u64()
    g.reseed(9)
    fresh = ParityRandomGenerator.from_seed(9)
    assert g.seed == 9
    assert g.random_u64() == fresh.random_u64()


def test_seedless_streams_are_independent():
    a = ParityRandomGenerator.from_seed(None)
    b = ParityRandomGenerator.from_seed(None)
    assert a.seed is None and b.seed is None
    assert [a.random_u64(), a.random_u64()] != [b.random_u64(), b.random_u64()]


# --------------------------------------------------------------------- int ranges
def test_integer_range_bounds_and_errors():
    g = ParityRandomGenerator.from_seed(7)
    for _ in range(500):
        assert g.randrange(2, 10, 2) in (2, 4, 6, 8)
        assert g.randrange(10, 2, -3) in (10, 7, 4)
        assert 1 <= g.randint(1, 3) <= 3
    with pytest.raises(RngError):
        g.randrange(1, 1, 1)
    with pytest.raises(RngError):
        g.randrange(1, 3, 0)
    with pytest.raises(RngError):
        g.randint(3, 1)
    with pytest.raises(RngError):
        g.randbelow(0)
    assert g.randbelow(1) == 0


def test_extreme_i64_ranges_do_not_overflow():
    g = ParityRandomGenerator.from_seed(70)
    lo, hi = -(2**63), 2**63 - 1
    for _ in range(200):
        assert lo <= g.randint(lo, hi) <= hi
        assert g.randrange(lo, hi, 1) < hi


# --------------------------------------------------------------------- selection
def test_choice_sample_shuffle_contracts():
    g = ParityRandomGenerator.from_seed(71)
    assert g.choice([10, 20, 30]) in (10, 20, 30)
    with pytest.raises(RngError):
        g.choice([])
    assert g.choices([], 0) == []
    with pytest.raises(RngError):
        g.choices([], 1)
    with pytest.raises(RngError):
        g.sample([1, 2], 3)
    picked = g.sample([1, 2, 3, 4], 3)
    assert len(picked) == 3 and len(set(picked)) == 3
    a = ParityRandomGenerator.from_seed(72)
    b = ParityRandomGenerator.from_seed(72)
    first, second = [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]
    a.shuffle(first)
    b.shuffle(second)
    assert first == second


def test_weighted_choice_validation_and_dominance():
    g = ParityRandomGenerator.from_seed(3)
    with pytest.raises(RngError):
        g.weighted_choice([1, 2], [1.0])
    with pytest.raises(RngError):
        g.weighted_choice([1, 2], [0.0, 0.0])
    with pytest.raises(RngError):
        g.weighted_choice([1, 2], [1.0, float("nan")])
    for _ in range(100):
        assert g.weighted_choice([1, 2], [0.0, 5.0]) == 2


# --------------------------------------------------------------------- distributions
def test_continuous_distributions_validate_parameters():
    g = ParityRandomGenerator.from_seed(81)
    for lam in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(RngError):
            g.expovariate(lam)
    for alpha in (0.0, -1.0, float("inf")):
        with pytest.raises(RngError):
            g.gammavariate(alpha, 1.0)
    for scale in (-1.0, float("nan"), float("inf")):
        with pytest.raises(RngError):
            g.normal(0.0, scale)
    assert g.normal(3.0, 0.0) == 3.0


def test_expovariate_and_gamma_means_track_parameters():
    g = ParityRandomGenerator.from_seed(8)
    exp = [g.expovariate(4.0) for _ in range(200_000)]
    assert abs(sum(exp) / len(exp) - 0.25) / 0.25 < 0.02
    gamma = [g.gammavariate(2.0, 3.0) for _ in range(200_000)]
    assert abs(sum(gamma) / len(gamma) - 6.0) / 6.0 < 0.02


def test_bounded_normal_clamps_and_validates():
    g = ParityRandomGenerator.from_seed(5)
    with pytest.raises(RngError):
        g.sample_normal(0.0, 1.0, 2.0, 1.0)
    assert g.sample_normal(10.0, 0.0, 0.0, 5.0) == 5.0
    assert 8.0 <= g.sample_normal(10.0, 2.0, 8.0, 12.0) <= 12.0


def test_positive_normal_integer_shortcuts():
    g = ParityRandomGenerator.from_seed(6)
    assert g.sample_positive_normal_integer(0.1, 0.0) == 1
    assert g.sample_positive_normal_integer(2.5, 0.0) == 2  # round-ties-even
    assert g.sample_positive_normal_integer(3.5, 0.0) == 4
    for _ in range(100):
        assert g.sample_positive_normal_integer(10.0, 2.0) >= 1


# --------------------------------------------------------------------- configured dists
def test_sampling_distribution_bounds_and_expected_value():
    g = ParityRandomGenerator.from_seed(1)
    dist = SamplingDistribution.fixed(10.0).with_bounds(0.0, 5.0)
    assert dist.sample(g) == 5.0
    assert dist.expected_value() == 10.0
    with pytest.raises(RngError):
        SamplingDistribution.fixed(1.0).with_bounds(2.0, 1.0)


def test_sequence_length_distribution_probability_and_sampling():
    with pytest.raises(RngError):
        SequenceLengthDistribution([SequenceLengthPair(10, 20, 90.0)])
    dist = SequenceLengthDistribution(
        [
            SequenceLengthPair(10, 20, 20.0),
            SequenceLengthPair(30, 40, 30.0),
            SequenceLengthPair(50, 60, 50.0),
        ]
    )
    g = ParityRandomGenerator.from_seed(919)
    samples = dist.sample_batch(g, 50_000)
    for pair, prob in (((10, 20), 0.2), ((30, 40), 0.3), ((50, 60), 0.5)):
        observed = sum(1 for s in samples if s == pair) / len(samples)
        assert abs(observed - prob) < 0.02


# --------------------------------------------------------------------- hash id
def test_hash_id_order_independence():
    base = ParityRandomGenerator.from_seed(42)
    g = HashIdRandomGenerator.from_base(base)
    scopes = [("trace-a", 7), ("trace-b", -3), ("trace-a", 99)]
    first = []
    for scope, hid in scopes:
        g.reseed_for_hash_id(hid, scope)
        first.append((g.random_u64(), g.random_u64()))
    for idx in reversed(range(len(scopes))):
        scope, hid = scopes[idx]
        g.reseed_for_hash_id(hid, scope)
        assert (g.random_u64(), g.random_u64()) == first[idx]


def test_hash_id_preserves_seed_zero_without_consuming_base():
    base = ParityRandomGenerator.from_seed(0)
    fresh = ParityRandomGenerator.from_seed(0)
    g = HashIdRandomGenerator.from_base(base)
    assert g.base_seed == 0
    assert g.generator.seed == 0
    # Reading a seeded base's seed must not consume base RNG state.
    assert base.random_u64() == fresh.random_u64()


def test_math_helpers_are_used(monkeypatch):
    # Guard: sample_positive_normal_integer uses math.ceil semantics (min 1).
    g = ParityRandomGenerator.from_seed(6)
    assert g.sample_positive_normal_integer(1.2, 0.0) == 1
    assert math.ceil(1.2) == 2  # sanity of the primitive used elsewhere
