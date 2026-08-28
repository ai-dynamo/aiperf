// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Performance-test boundary for native AIPerf plugins.
//!
//! Four modules implement the Task-38 distribution parity gate:
//! - [`comparator`]: validates that the static comparator build is
//!   byte-for-byte identical to the dynamic candidate in every identity
//!   dimension that could affect measured performance.
//! - [`experiment`]: state machine for a balanced AB/BA paired experiment.
//! - [`stats`]: deterministic statistical engine (Hyndman-Fan type-7 quantile,
//!   coefficient of variation, paired bootstrap).
//! - [`report`]: the self-contained result document the gate publishes.
//!
//! The `parity` binary composes the four into a runnable gate. This phase
//! builds the harness only; the measurements it consumes arrive with the
//! plugin-backed builds it will compare.

pub mod comparator;
pub mod experiment;
pub mod report;
pub mod stats;

/// The source API version exposed by provisional plugin crate shells.
pub const PLUGIN_SOURCE_API_VERSION: &str = "1.0.0";
