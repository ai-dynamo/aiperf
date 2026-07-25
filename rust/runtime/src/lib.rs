// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! AIPerf load-generation runtime.
//!
//! The crate composes endpoint, transport, scheduling, workload, reporting,
//! accuracy, and extension subsystems for the `aiperf` binary.
//! With the `dynosim` Cargo feature, the `dynosim` module composes the
//! same workloads and observers with `SimClock` plus Dynamo's passive mock
//! engine for deterministic, socket-free co-simulation.

pub mod accuracy;
pub mod adaptive;
/// Always-compiled subagent tree-spec side channel (`TreeSpec`) for the
/// `agentic_replay` timing mode.
pub mod agentic_tree;
/// AgentX agentic-replay timing mode (scheduled-runtime Workload). Requires the
/// `agentx` feature.
#[cfg(feature = "agentx")]
pub mod agentic_replay;
#[cfg(feature = "agentx")]
pub mod agentx;
#[cfg(feature = "dynamo-aic-forward-pass")]
pub mod aic_runtime;
pub mod ancillary;
#[cfg(feature = "dynosim")]
pub mod dynosim;
pub mod export;
pub mod failure;
pub mod fixed_schedule;
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

#[cfg(feature = "engine")]
pub mod engine;

pub mod accuracy_core;
pub mod adaptive_core;
pub mod body_plan;
pub mod cellular;
pub mod clock;
pub mod config;
pub mod content_server;
pub mod dataset;
pub mod dispatch;
pub mod endpoints;
pub mod extensions;
pub mod gpu_telemetry;
pub mod graph;
#[cfg(feature = "cellular")]
pub mod hub;
pub mod metrics_core;
pub mod network_latency;
pub mod rng;
pub mod server_metrics;
pub mod timing;
pub mod transport;

#[cfg(test)]
mod test_util;
