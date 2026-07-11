// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hash-derived random-number substrate for AIPerf.
//!
//! AIPerf's reproducibility contract is order-independent derivation: a component
//! names its stream, and the stream depends only on the run root seed and that
//! identifier. This crate owns that BLAKE3 seed algebra plus the Rust-native
//! `Pcg64` generator wrappers used by future dataset, scheduler, and graph
//! synthesis code.

pub mod derive;
pub mod error;
pub mod generator;

pub use derive::{RngRoot, derive_seed_parts, derive_seed_u64};
pub use error::{Result, RngError};
pub use generator::RandomGenerator;
