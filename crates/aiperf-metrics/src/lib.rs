// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native Rust AIPerf metrics primitives.
//!
//! This crate is the IO-free metrics plane described by
//! `specs/2026-07-10-aiperf-rust-metrics-accumulator-sweepline-design.md`.
//! Task 1 establishes the public boundary value type, unit vocabulary, and
//! distribution kernels that later accumulator, catalog, and report code consume.

pub mod kernel;
pub mod units;
pub mod value;

pub use kernel::{linear_distribution, nearest_distribution, DistributionStats, PERCENTILES};
pub use units::{MetricValueType, Unit, UnitConversionError};
pub use value::MetricValue;
