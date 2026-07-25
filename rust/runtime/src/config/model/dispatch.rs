// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Admission-strategy selector shared by the typed config model and runtime.
//!
//! `DispatchMode` is the `runtime.dispatch` selector for `workers>1` scheduled
//! execution. It is defined here in the leaf config crate so both the typed
//! Config-v2 model and `aiperf-runtime` share one serde-stable enum without a
//! dependency cycle.

use serde::{Deserialize, Serialize};

/// Admission strategy for `workers>1` scheduled execution.
///
/// - `Sharded` statically partitions request budget, concurrency, and rate
///   `1/workers`-ways up front, per worker thread.
/// - `Global` (default) admits from one shared per-cell slot pool / rate gate,
///   so aggregate concurrency and rate across all worker threads is byte-exact
///   against a single global limiter.
/// - `GlobalHop` additionally routes every individual request through one
///   coordinator-owned dispatcher, for exact request-to-thread assignment order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DispatchMode {
    Sharded,
    #[default]
    Global,
    GlobalHop,
}
