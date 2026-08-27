# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-language byte-exact parity tests for the parity RNG.

Replays the operation script from the Rust golden-vector generator
(``rust/aiperf/examples/rng_parity_vectors.rs``) through
:class:`aiperf.rust_shims.rng_parity.ParityRandomGenerator` and asserts every output matches
bit-for-bit. Floats are compared via their raw IEEE-754 ``u64`` bit pattern (the golden
stores ``f64::to_bits``), so the assertion is exact.

If this test fails after a deliberate Rust-side change, regenerate the golden:

    cargo run -p aiperf --example rng_parity_vectors \\
        > rust/aiperf/tests/data/rng_parity_vectors.json
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from aiperf.rust_shims.rng_parity import (
    HashIdRandomGenerator,
    ParityRandomGenerator,
    RngRoot,
    namespace,
)

_GOLDEN_PATH = (
    Path(__file__).resolve().parents[3]
    / "rust"
    / "aiperf"
    / "tests"
    / "data"
    / "rng_parity_vectors.json"
)


def _f64_bits(value: float) -> str:
    """Return the IEEE-754 ``u64`` bit pattern of ``value`` as a decimal string."""
    return str(struct.unpack("<Q", struct.pack("<d", value))[0])


@pytest.fixture(scope="module")
def golden() -> dict:
    if not _GOLDEN_PATH.exists():  # pragma: no cover - regeneration guidance
        pytest.skip(
            f"golden vectors missing: {_GOLDEN_PATH}; regenerate with "
            "`cargo run -p aiperf --example rng_parity_vectors`"
        )
    return json.loads(_GOLDEN_PATH.read_text())


def test_derive_vectors(golden: dict) -> None:
    derive = golden["derive"]
    root = RngRoot(42)
    for name in namespace.ALL:
        assert str(root.derive_seed(name)) == derive[name], name
    assert str(RngRoot(42).derive_seed("")) == derive["empty_id"]
    assert str(RngRoot(0).derive_seed("a")) == derive["root0_a"]
    assert (
        str(RngRoot(42).derive_variation_seed("concurrency=4")) == derive["variation"]
    )


def test_u64_stream(golden: dict) -> None:
    g = ParityRandomGenerator.from_seed(42)
    assert [str(g.random_u64()) for _ in range(12)] == golden["u64"]


def test_f64_stream(golden: dict) -> None:
    g = ParityRandomGenerator.from_seed(43)
    assert [_f64_bits(g.random()) for _ in range(12)] == golden["f64_bits"]


def test_integer_ranges(golden: dict) -> None:
    g = ParityRandomGenerator.from_seed(44)
    ranges = []
    ranges += [g.randrange(2, 10, 2) for _ in range(5)]
    ranges += [g.randrange(10, 2, -3) for _ in range(5)]
    ranges += [g.randint(1, 3) for _ in range(5)]
    ranges += [g.randbelow(100) for _ in range(5)]
    assert ranges == golden["randrange"]
    assert [str(g.randrange_u64(1000, 2000)) for _ in range(5)] == golden[
        "randrange_u64"
    ]


def test_choice(golden: dict) -> None:
    pool = list(range(10))
    g = ParityRandomGenerator.from_seed(45)
    assert [g.choice(pool) for _ in range(12)] == golden["choice"]


def test_shuffle(golden: dict) -> None:
    pool = list(range(10))
    g = ParityRandomGenerator.from_seed(46)
    g.shuffle(pool)
    assert pool == golden["shuffle"]


def test_sample(golden: dict) -> None:
    pool = list(range(10))
    g = ParityRandomGenerator.from_seed(47)
    assert g.sample(pool, 5) == golden["sample"]


def test_weighted_choice(golden: dict) -> None:
    vals = [0, 1, 2, 3]
    weights = [1.0, 2.0, 3.0, 4.0]
    g = ParityRandomGenerator.from_seed(48)
    assert [g.weighted_choice(vals, weights) for _ in range(12)] == golden[
        "weighted_choice"
    ]


def test_numpy_choice_replace(golden: dict) -> None:
    five = list(range(5))
    w5 = [1.0, 0.0, 2.0, 1.0, 4.0]
    g = ParityRandomGenerator.from_seed(49)
    assert g.numpy_choice(five, 8, w5, True) == golden["numpy_choice_replace"]


def test_numpy_choice_noreplace(golden: dict) -> None:
    five = list(range(5))
    w5 = [1.0, 0.0, 2.0, 1.0, 4.0]
    g = ParityRandomGenerator.from_seed(50)
    assert g.numpy_choice(five, 3, w5, False) == golden["numpy_choice_noreplace"]


def test_expovariate(golden: dict) -> None:
    g = ParityRandomGenerator.from_seed(51)
    assert [_f64_bits(g.expovariate(4.0)) for _ in range(12)] == golden[
        "expovariate_bits"
    ]


def test_gammavariate(golden: dict) -> None:
    g = ParityRandomGenerator.from_seed(52)
    got = []
    got += [_f64_bits(g.gammavariate(2.0, 3.0)) for _ in range(8)]
    got += [_f64_bits(g.gammavariate(0.5, 2.0)) for _ in range(8)]
    got += [_f64_bits(g.gammavariate(1.0, 2.0)) for _ in range(4)]
    assert got == golden["gammavariate_bits"]


def test_normal(golden: dict) -> None:
    g = ParityRandomGenerator.from_seed(53)
    assert [_f64_bits(g.normal(4.0, 2.0)) for _ in range(12)] == golden["normal_bits"]


def test_sample_normal(golden: dict) -> None:
    g = ParityRandomGenerator.from_seed(54)
    assert [
        _f64_bits(g.sample_normal(10.0, 2.0, 8.0, 12.0)) for _ in range(12)
    ] == golden["sample_normal_bits"]


def test_positive_normal_integer(golden: dict) -> None:
    g = ParityRandomGenerator.from_seed(55)
    assert [g.sample_positive_normal_integer(100.0, 10.0) for _ in range(12)] == golden[
        "positive_normal_int"
    ]


def test_hash_id_stream(golden: dict) -> None:
    base = ParityRandomGenerator.from_seed(42)
    hid = HashIdRandomGenerator.from_base(base)
    got = []
    for scope, hash_id in [("trace-a", 7), ("trace-b", -3), ("trace-a", 99)]:
        hid.reseed_for_hash_id(hash_id, scope)
        got.append(str(hid.random_u64()))
        got.append(str(hid.random_u64()))
    assert got == golden["hash_id_u64"]


def test_fill_bytes(golden: dict) -> None:
    g = ParityRandomGenerator.from_seed(56)
    assert g.fill_bytes(37).hex() == golden["fill_bytes_hex"]
