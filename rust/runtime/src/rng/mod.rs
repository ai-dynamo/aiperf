// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hash-derived random-number substrate for AIPerf.
//!
//! Streams are derived solely from the run root seed and component identifier
//! using BLAKE3, then consumed through `Pcg64` wrappers.

pub mod compat;
pub mod derive;
pub mod dist;
pub mod error;
pub mod generator;
pub mod hash_id;
pub mod namespace;
pub mod random_generator;

pub use compat::python_mt::PythonMt19937;
pub use compat::python_random::PythonRandomGenerator;
pub use derive::{RngRoot, derive_seed_parts, derive_seed_u64};
pub use dist::{
    DistributionSampler, EmpiricalDistribution, EmpiricalPoint, FixedDistribution,
    LogNormalDistribution, MultimodalDistribution, NormalDistribution, PeakEntry,
    SamplingDistribution, SamplingRng, SequenceLengthDistribution, SequenceLengthPair,
    SequenceSampler,
};
pub use error::{Result, RngError};
pub use generator::RustRandomGenerator;
pub use hash_id::HashIdRandomGenerator;
pub use random_generator::RandomGenerator;
