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

/// Worker-assignment policy applied at the single [`DispatchMode::GlobalHop`]
/// pick site (`ThreadPerCoreExecutor::execute_command`) when `workers > 1`.
///
/// The hop only chooses *which worker executes an already-issued request*; every
/// global-hop guarantee (exactly-once, deterministic merged record order,
/// aggregate concurrency/rate/arrival pattern) is coordinator-side and unaffected
/// by this choice, so the policy is free to trade placement determinism for
/// per-session connection reuse.
///
/// - `RoundRobin` (default) hops each issued turn to worker `i % workers` in
///   issuance order — deterministic and load-even, but it fragments a session's
///   worker-local sticky connection pool across workers.
/// - `Sticky` maps every turn of a conversation to one worker via a fixed
///   seed-free hash of its `correlation_id`, so the worker-local sticky pool
///   reuses one connection per session; a turn with no `correlation_id` falls
///   back to round-robin.
/// - `LeastLoaded` sends a new session to the worker with the shallowest in-flight
///   count, then binds that `correlation_id` to the chosen worker so its
///   continuations stay sticky.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum HopRouting {
    #[default]
    RoundRobin,
    Sticky,
    LeastLoaded,
}
