// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One-process execution boundary between Python orchestration and Rust runs.
//!
//! Python owns Config v2, sweep/search planning, trials, convergence, and
//! aggregation. This crate accepts exactly one versioned run request and owns
//! dataset construction, phase scheduling, HTTP dispatch, measurement, and the
//! authoritative native-v2 report.

pub mod execute;
mod gpu_telemetry;
mod network_latency;
pub mod protocol;
mod records;
mod server_metrics;

pub use execute::execute_run;
pub use protocol::{RUNNER_PROTOCOL_VERSION, RunRequest, RunTerminal, RunnerCapabilities};
