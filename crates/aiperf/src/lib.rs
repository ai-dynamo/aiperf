// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rust-native AIPerf CLI (slim).
//!
//! Measurement wire types + the collector observer live in [`aiperf_core`]; the
//! Graph-IR engine lives in `aiperf_graph`; the clock-native scheduling policy
//! (arrivals, slots, stop conditions, ramps, cancellation, and URL selection)
//! lives in shared [`aiperf_timing`]. This crate owns the CLI/runtime composition:
//! the online HTTP sink over `aiperf-transport` ([`http`]), ancillary policy
//! wiring ([`ancillary`]), phased scheduled execution ([`phase_runtime`]),
//! workload shaping ([`workload`]), the online run loop ([`run`]), reporting
//! ([`report`]), and logging setup ([`logging`]). Named
//! compile-time extension composition lives in `aiperf_extensions` so extension
//! crates never need a dependency cycle through this application crate.

pub mod accuracy;
pub mod adaptive;
pub mod agentic;
pub mod ancillary;
pub mod fixed_schedule;
pub mod http;
pub mod logging;
pub mod metrics;
pub mod multiturn;
pub mod phase_runtime;
pub mod report;
pub mod run;
pub mod scheduled;
pub mod scheduler;
pub mod user_centric;
pub mod workload;

#[cfg(test)]
mod test_util;
