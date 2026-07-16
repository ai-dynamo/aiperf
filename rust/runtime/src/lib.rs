// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rust-native AIPerf load-generation runtime.
//!
//! OpenAI SSE chunk types live in [`crate::transport_http`]`::sse`, the OpenAI
//! chat request-body builder in [`crate::endpoints`], and the `CollectorObserver`
//! recorder in [`loadgen_core`]`::observer`; the
//! Graph-IR engine lives in `crate::graph`; the clock-native scheduling policy
//! (arrivals, slots, stop conditions, ramps, cancellation, and URL selection)
//! lives in shared [`crate::timing`]. This library owns runtime composition used
//! by `aiperf-runner`: the online HTTP sink over `aiperf-transport-http` ([`http`]),
//! ancillary policy wiring ([`ancillary`]), phased scheduled execution
//! ([`phase_runtime`]), workload shaping ([`workload`]), the online run loop
//! ([`run`]), reporting ([`report`]), and canonical static-accuracy execution
//! seams. Named compile-time extension composition lives in `crate::extensions`
//! and is owned by the runner, so extension crates never need a dependency cycle
//! through this runtime crate.
//! With the `dynosim` Cargo feature, [`dynosim`] composes the
//! same workloads and observers with `SimClock` plus Dynamo's passive mock
//! engine for deterministic, socket-free co-simulation.

pub mod accuracy;
pub mod adaptive;
#[cfg(feature = "dynamo-aic-forward-pass")]
pub mod aic_runtime;
pub mod ancillary;
#[cfg(feature = "dynosim")]
pub mod dynosim;
pub mod export;
pub mod failure;
pub mod fixed_schedule;
pub mod grpc;
pub mod http;
pub mod metrics;
pub mod multiturn;
pub mod phase_runtime;
pub mod realtime;
pub mod report;
pub mod request_rate;
pub mod run;
pub mod scheduled;
pub mod scheduler;
pub mod user_centric;
pub mod workload;

// The v2 protocol / registry / execution layer relocated out of `aiperf-runner`
// so there can eventually be one registry in `aiperf`. Gated by `runner-protocol`
// so `mock-server` and other library consumers skip it entirely.
#[cfg(feature = "runner-protocol")]
pub mod runner_protocol;

// Modules absorbed from the formerly-standalone aiperf-* library crates.
pub mod accuracy_core;
pub mod adaptive_core;
pub mod body_plan;
pub mod cellular;
pub mod clock;
pub mod content_server;
pub mod dataset;
pub mod endpoints;
pub mod extensions;
pub mod gpu_telemetry;
pub mod graph;
pub mod metrics_core;
pub mod network_latency;
pub mod rng;
pub mod server_metrics;
pub mod timing;
pub mod transport_grpc;
pub mod transport_http;

#[cfg(test)]
mod test_util;
