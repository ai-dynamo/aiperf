// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Graph-IR async-dataflow workload driver — online (live HTTP) mode.
//!
//! Runs a DAG of chat requests with fan-out/fan-in dependencies and firing-gate
//! timing: nodes fire when their input channels are ready, dispatch through the
//! extensible [`sink::GraphSink`] (HTTP → an OpenAI-compatible endpoint), and
//! measurement flows to `loadgen_core`'s shared `TraceCollector`. Prompts are
//! materialized from a content-addressed [`segment::SegmentStore`] plus dynamic
//! predecessor replies.
//!
//! Depends on `aiperf` (HTTP transport + collector observer), `loadgen-core`
//! (the measurement seam), and `aiperf-clock` (wall/virtual `Clock`). The
//! offline virtual-clock co-simulation path is intentionally **not** here.

pub mod bench;
pub mod channel_store;
pub mod channels;
pub mod context;
pub mod errors;
pub mod executor;
pub mod materialize;
pub mod model;
pub mod reducers;
pub mod run;
pub mod runtime;
pub mod scheduler;
pub mod segment;
pub mod sink;
mod syslimits;
pub mod transport_bench;
pub mod transport_sink;
pub mod validate;
pub mod wire;

#[cfg(test)]
mod test_util;
