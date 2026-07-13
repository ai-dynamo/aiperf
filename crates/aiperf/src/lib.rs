// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rust-native AIPerf load-generation runtime.
//!
//! OpenAI SSE chunk types live in [`aiperf_transport_http`]`::sse`, the OpenAI
//! chat request-body builder in [`aiperf_endpoints`], and the `CollectorObserver`
//! recorder in [`loadgen_core`]`::observer`; the
//! Graph-IR engine lives in `aiperf_graph`; the clock-native scheduling policy
//! (arrivals, slots, stop conditions, ramps, cancellation, and URL selection)
//! lives in shared [`aiperf_timing`]. This library owns runtime composition used
//! by `aiperf-runner`: the online HTTP sink over `aiperf-transport-http` ([`http`]),
//! ancillary policy wiring ([`ancillary`]), phased scheduled execution
//! ([`phase_runtime`]), workload shaping ([`workload`]), the online run loop
//! ([`run`]), and reporting ([`report`]). Named compile-time extension
//! composition lives in `aiperf_extensions` and is owned by the runner, so
//! extension crates never need a dependency cycle through this runtime crate.
//! With the `dynosim` Cargo feature, [`dynosim`] composes the
//! same workloads and observers with `SimClock` plus Dynamo's passive mock
//! engine for deterministic, socket-free co-simulation.

pub mod adaptive;
pub mod ancillary;
#[cfg(feature = "dynamo-aic-forward-pass")]
pub mod aic_runtime;
#[cfg(feature = "dynosim")]
pub mod dynosim;
pub mod fixed_schedule;
pub mod grpc;
pub mod http;
pub mod metrics;
pub mod multiturn;
pub mod phase_runtime;
pub mod report;
pub mod request_rate;
pub mod run;
pub mod scheduled;
pub mod scheduler;
pub mod user_centric;
pub mod workload;

#[cfg(test)]
mod test_util;
