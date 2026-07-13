// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Clock-injected TCP-connect RTT calibration.
//!
//! This resolves once, opens a fresh plain TCP connection for every sample,
//! retains failures without failing the benchmark, and computes one flat mean
//! over successful samples.

mod accumulator;
mod model;
mod probe;

pub use accumulator::{NetworkLatencyAccumulator, NetworkLatencyMergeError};
pub use model::{
    NetworkLatencyErrorDetails, NetworkLatencyErrorDetailsCount, NetworkLatencyResults,
    NetworkLatencySample, NetworkLatencyStats, NetworkLatencyTarget,
    NetworkLatencyTargetParseError, NetworkLatencyTargetSummary,
};
pub use probe::{LocalProbeFuture, NetworkLatencyProbe, TcpConnectProbe};
