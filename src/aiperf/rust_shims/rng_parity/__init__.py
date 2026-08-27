# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Byte-exact pure-Python port of the Rust ``aiperf::rng`` module.

This subpackage reproduces ``rust/aiperf/src/rng/`` bit-for-bit against
``rand 0.9.4`` / ``rand_pcg 0.9.0`` / ``rand_distr 0.5.1``: BLAKE3 seed algebra
(:mod:`~aiperf.common.rng_parity.derive`), the ``Pcg64`` core
(:mod:`~aiperf.common.rng_parity.pcg64`), the uniform/int/float sampling surface
(:mod:`~aiperf.common.rng_parity.generator`), the ziggurat continuous distributions and
configured sampling distributions (:mod:`~aiperf.common.rng_parity.dist`), and the
hash-id generator (:mod:`~aiperf.common.rng_parity.hash_id`).

It is selected at runtime by ``AIPERF_RNG_BACKEND=rust`` so seeded Python and Rust
produce identical streams in tests. See
``~/.aiperf/docs/superpowers/specs/2026-07-14-python-parity-rng-design.md``.
"""

from __future__ import annotations

from aiperf.common.rng_parity import namespace
from aiperf.common.rng_parity.derive import (
    RngRoot,
    derive_seed_parts,
    derive_seed_u64,
)
from aiperf.common.rng_parity.dist import (
    EmpiricalDistribution,
    EmpiricalPoint,
    FixedDistribution,
    LogNormalDistribution,
    MultimodalDistribution,
    NormalDistribution,
    PeakEntry,
    SamplingDistribution,
    SequenceLengthDistribution,
    SequenceLengthPair,
)
from aiperf.common.rng_parity.errors import RngError
from aiperf.common.rng_parity.generator import ParityRandomGenerator
from aiperf.common.rng_parity.hash_id import HashIdRandomGenerator

__all__ = [
    "EmpiricalDistribution",
    "EmpiricalPoint",
    "FixedDistribution",
    "HashIdRandomGenerator",
    "LogNormalDistribution",
    "MultimodalDistribution",
    "NormalDistribution",
    "ParityRandomGenerator",
    "PeakEntry",
    "RngError",
    "RngRoot",
    "SamplingDistribution",
    "SequenceLengthDistribution",
    "SequenceLengthPair",
    "derive_seed_parts",
    "derive_seed_u64",
    "namespace",
]
