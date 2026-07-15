// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One-process execution boundary between Python orchestration and Rust runs.
//!
//! Python owns Config v2, sweep/search planning, trials, convergence, and
//! aggregation. This crate accepts exactly one versioned run request and owns
//! dataset construction, phase scheduling, HTTP dispatch, measurement, and the
//! authoritative native-v2 report.
pub use aiperf::runner_protocol::application::RunnerApplication;
pub use aiperf::runner_protocol::distribution_identity::current_distribution_id;
pub use aiperf::runner_protocol::execution_factories::{
    RunnerExecutionFactories, native_execution_factories,
};
pub use aiperf::runner_protocol::graph_execution::{
    NativeRunnerGraphPlacementFactory, RunnerGraphPlacementFactory,
};
pub use aiperf::runner_protocol::grpc_turn_execution::NativeGrpcExecutionBackendFactory;
pub use aiperf::runner_protocol::turn_execution::{
    HttpExecutionBackendConfig, NativeRequestExecutorFactory, RequestExecutorFactory,
};
