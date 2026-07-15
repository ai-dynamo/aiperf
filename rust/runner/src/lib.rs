// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One-process execution boundary between Python orchestration and Rust runs.
//!
//! Python owns Config v2, sweep/search planning, trials, convergence, and
//! aggregation. This crate accepts exactly one versioned run request and owns
//! dataset construction, phase scheduling, HTTP dispatch, measurement, and the
//! authoritative native-v2 report.
pub mod application;
pub mod control_plane_http;
pub mod coordinator;
pub mod dataset_input;
pub mod distribution_identity;
pub mod execute;
pub mod execution_factories;
mod gpu_telemetry;
mod graph_execution;
pub mod graph_input;
mod graph_phase_runtime;
mod heartbeat_lane;
pub mod cellular_cell;
pub mod cellular_controller;
pub mod grpc_execution;
pub mod grpc_turn_execution;
mod live_streaming;
mod network_latency;
#[cfg(feature = "dynosim")]
pub mod offline_execution;
pub mod online_execution;
pub mod protocol;
pub mod protocol_v2;
pub mod readiness;
mod records;
pub mod redaction;
pub mod registry;
mod server_metrics;
mod sharded_scheduled;
pub mod sidecar_input;
pub mod turn_execution;

pub use application::RunnerApplication;
pub use distribution_identity::current_distribution_id;
pub use execution_factories::{RunnerExecutionFactories, native_execution_factories};
pub use graph_execution::{NativeRunnerGraphPlacementFactory, RunnerGraphPlacementFactory};
pub use grpc_turn_execution::NativeGrpcExecutionBackendFactory;
pub use turn_execution::{
    HttpExecutionBackendConfig, NativeRequestExecutorFactory, RequestExecutorFactory,
};
