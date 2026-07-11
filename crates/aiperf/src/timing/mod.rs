// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Clock-native load-generation timing plane.
//!
//! Everything here sources time from the injected `aiperf-clock::Clock` — never a
//! wall clock — so the same workload code runs on a `RealClock` (online) or a
//! `VirtualClock` (offline). This module is the Rust home for the Python
//! `src/aiperf/timing/` subsystem, built clock-first rather than retrofitted.
//!
//! First increment: inter-arrival [`intervals`] generators (the open-loop pacer's
//! distribution source). Slots, ramps, stop conditions, and the per-mode workload
//! generators land on top of this seam.

pub mod intervals;

pub use intervals::{ArrivalPattern, IntervalGenerator, make_interval_generator};
