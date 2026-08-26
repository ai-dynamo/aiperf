// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hash-derived random-number substrate for AIPerf.
//!
//! Streams are derived from the run root seed and component identifier alone,
//! never from creation or draw order. The native backend
//! ([`RustRandomGenerator`]) derives with BLAKE3 and draws from `rand_pcg`'s
//! `Pcg64`; the Python-parity backend ([`PythonRandomGenerator`]) derives with
//! SHA-256 and draws from a CPython Mersenne Twister plus a numpy-compatible
//! PCG64, matching `src/aiperf/common/random_generator.py` byte-for-byte.

pub mod compat;
pub mod configured;
pub mod derive;
pub mod dist;
pub mod error;
pub mod generator;
pub mod hash_id;
pub mod namespace;
pub mod random_generator;

pub use compat::python_mt::PythonMt19937;
pub use compat::python_random::PythonRandomGenerator;
pub use configured::{
    ConfiguredRandomGenerator, RuntimeRngBackend, configured_runtime_rng_backend,
};
pub use derive::{DerivedRandomGenerator, RngRoot, derive_seed_parts, derive_seed_u64};
pub use dist::{
    DistributionSampler, EmpiricalDistribution, EmpiricalPoint, FixedDistribution,
    LogNormalDistribution, MultimodalDistribution, NormalDistribution, PeakEntry,
    SamplingDistribution, SamplingRng, SequenceLengthDistribution, SequenceLengthPair,
    SequenceSampler,
};
pub use error::{Result, RngError};
pub use generator::RustRandomGenerator;
pub use hash_id::HashIdRandomGenerator;
pub use random_generator::{RandomGenerator, RuntimeRandomGenerator};
