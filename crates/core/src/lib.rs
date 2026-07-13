// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared AIPerf measurement layer.
//!
//! Owns the OpenAI SSE wire types ([`sse`]) and the [`observer`] that funnels
//! measurements into `loadgen_core`'s `TraceCollector`. Live HTTP dispatch is
//! done over the `aiperf-transport-http` (hyper) client by the sinks in the `aiperf`
//! CLI and the `aiperf-graph` engine, which parse SSE deltas via
//! [`sse::ChatChunk`]. Extracted so both can build on it without a dependency
//! cycle.

pub mod chat;
pub mod observer;
pub mod sse;
