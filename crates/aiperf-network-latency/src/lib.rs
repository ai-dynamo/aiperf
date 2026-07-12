// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Clock-injected TCP-connect RTT calibration.
//!
//! This is the native form of `src/aiperf/network_latency/probe.py:20-172`,
//! `manager.py:67-188`, and `accumulator.py:21-127`: resolve once, open a fresh
//! plain TCP connection for every sample, retain failures without failing the
//! benchmark, and compute one flat mean over successful samples.

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
