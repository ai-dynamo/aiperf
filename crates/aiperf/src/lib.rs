// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rust-native AIPerf load-generation runtime.
//!
//! Measurement wire types + the collector observer live in [`aiperf_core`]; the
//! Graph-IR engine lives in `aiperf_graph`; the clock-native scheduling policy
//! (arrivals, slots, stop conditions, ramps, cancellation, and URL selection)
//! lives in shared [`aiperf_timing`]. This library owns runtime composition used
//! by `aiperf-runner`: the online HTTP sink over `aiperf-transport-http` ([`http`]),
//! ancillary policy wiring ([`ancillary`]), phased scheduled execution
//! ([`phase_runtime`]), workload shaping ([`workload`]), the online run loop
//! ([`run`]), reporting ([`report`]), and canonical accuracy/agentic execution
//! seams. Named compile-time extension composition lives in `aiperf_extensions`
//! and is owned by the runner, so extension crates never need a dependency cycle
//! through this runtime crate.
//! With the `dynamo-offline` Cargo feature, [`dynamo_offline`] composes the
//! same workloads and observers with `SimClock` plus Dynamo's passive mock
//! engine for deterministic, socket-free co-simulation.

pub mod accuracy;
pub mod adaptive;
pub mod agentic;
pub mod agentic_gateway;
pub mod ancillary;
#[cfg(feature = "dynamo-offline")]
pub mod dynamo_offline;
pub mod evaluation;
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
