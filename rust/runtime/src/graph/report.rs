// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transport-neutral graph throughput reporting.

use crate::metrics_core::AccumulatorSummary;

/// Throughput, time-to-first-token, and native metric results for a graph run.
#[derive(Debug, Clone)]
pub struct GraphRpsReport {
    /// Number of successfully completed graph requests.
    pub completed: u64,
    /// Number of graph requests that ended in an error.
    pub errors: u64,
    /// Number of emitted output tokens.
    pub output_tokens: u64,
    /// Measured wall-clock duration in seconds.
    pub wall_secs: f64,
    /// Median time to first token in milliseconds.
    pub ttft_p50_ms: f64,
    /// 90th-percentile time to first token in milliseconds.
    pub ttft_p90_ms: f64,
    /// 99th-percentile time to first token in milliseconds.
    pub ttft_p99_ms: f64,
    /// Mean time to first token in milliseconds.
    pub ttft_mean_ms: f64,
    /// Native typed distributions and sweeps merged across graph workers.
    pub native_metrics: AccumulatorSummary,
}

impl GraphRpsReport {
    /// Returns successful graph requests per measured wall-clock second.
    pub fn rps(&self) -> f64 {
        if self.wall_secs > 0.0 {
            self.completed as f64 / self.wall_secs
        } else {
            0.0
        }
    }

    /// Returns output tokens per measured wall-clock second.
    pub fn output_tps(&self) -> f64 {
        if self.wall_secs > 0.0 {
            self.output_tokens as f64 / self.wall_secs
        } else {
            0.0
        }
    }
}
