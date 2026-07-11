// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Clock-native load-generation timing plane.
//!
//! The Rust home for the Python `src/aiperf/timing/` subsystem, built clock-first
//! rather than retrofitted. Time is **injected as nanoseconds** by the caller (from
//! an `aiperf-clock::Clock`) — nothing here reads a wall clock — so the identical
//! policy drives a `RealClock` (online) or a `SimClock` (offline) run. That is why
//! this crate carries no clock dependency of its own: it is a leaf both the online
//! CLI ([`aiperf`](../aiperf/index.html)) and the graph runtime
//! ([`aiperf_graph`](../aiperf_graph/index.html)) depend on, so the scheduling
//! policy lives in exactly one place and cannot drift between the two paths.
//!
//! The seam is four trait families, each with at least one concrete impl:
//! - [`intervals`] — inter-arrival distribution ([`IntervalGenerator`]),
//! - [`slots`] — concurrency admission ([`SlotPool`], debt-drain-capable),
//! - [`stop`] — run-termination bounds ([`StopCondition`] / [`StopChecker`]),
//! - [`user_centric`] — per-user session schedule math ([`plan_user_centric`]).

pub mod intervals;
pub mod slots;
pub mod stop;
pub mod user_centric;

pub use intervals::{ArrivalPattern, IntervalGenerator, make_interval_generator};
pub use slots::{ConcurrencyManager, ConcurrencyStats, SlotGuard, SlotPool};
pub use stop::{RunState, StopChecker, StopCondition, StopConfig};
pub use user_centric::{InitialUser, UserCentricPlan, plan_user_centric};
