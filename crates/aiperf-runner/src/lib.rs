// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One-process execution boundary between Python orchestration and Rust runs.
//!
//! Python owns Config v2, sweep/search planning, trials, convergence, and
//! aggregation. This crate accepts exactly one versioned run request and owns
//! dataset construction, phase scheduling, HTTP dispatch, measurement, and the
//! authoritative native-v2 report.

pub mod agentic_execution;
pub mod distribution_identity;
pub mod execute;
mod gpu_telemetry;
mod graph_execution;
mod live_streaming;
mod network_latency;
#[cfg(feature = "dynamo-offline")]
pub mod offline_execution;
pub mod online_execution;
pub mod protocol;
pub mod protocol_v2;
mod records;
pub mod registry;
mod server_metrics;
pub mod turn_execution;

pub use distribution_identity::current_distribution_id;
pub use execute::{execute_run, execute_run_with_all_factories, execute_run_with_backend_factory};
pub use graph_execution::{NativeRunnerGraphPlacementFactory, RunnerGraphPlacementFactory};
pub use protocol::{RUNNER_PROTOCOL_VERSION, RunRequest, RunTerminal, RunnerCapabilities};
pub use turn_execution::{
    HttpExecutionBackendConfig, HttpExecutionBackendFactory, NativeHttpExecutionBackendFactory,
};
