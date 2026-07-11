// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared AIPerf HTTP client + measurement layer.
//!
//! Owns the OpenAI client layer (wire types in [`sse`], streaming transport in
//! [`http_sink`]) and the [`observer`] that funnels measurements into
//! `loadgen_core`'s `TraceCollector`. Extracted so both the slim `aiperf` CLI
//! and the `aiperf-graph` engine can build on it without a dependency cycle.

use std::time::Instant;

pub mod http_sink;
pub mod observer;
pub mod sse;
pub mod wire;

/// Milliseconds elapsed since `start` on the shared monotonic clock. Shared by
/// [`observer::CollectorObserver`] and [`http_sink::HttpSink`] so every
/// timestamp is computed identically.
pub(crate) fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

#[cfg(test)]
mod test_util;
