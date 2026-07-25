// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Standalone, byte-exact Rust port of the Python **AgentX v1.0** feature
//! (WEKA trace replay + agentic-replay timing + scenario locks).
//!
//! This subsystem deliberately shares **no logic** with the next-gen graph-ir
//! recorded path (`crate::graph::recorded`). It is a faithful 1:1 parity port of
//! the Python implementation under `src/aiperf/dataset/loader/weka_*.py` and
//! `src/aiperf/timing/`, gated behind the `agentx` Cargo feature and intended to
//! be deleted wholesale once graph-ir supersedes AgentX. See
//! `specs/agentx-rust-port.md` for the design record.
//!
//! Parity is proven, not asserted: reconstruction is deterministic given
//! `(seed, trace)`, and every module is cross-checked byte-for-byte against its
//! Python counterpart's output over the in-repo `tests/fixtures/weka_traces*/`.

pub mod cache_bust;
pub mod chains;
pub mod config;
pub mod corpus;
pub mod export;
pub mod loader;
pub mod metrics;
pub mod plan;
pub mod prepass;
pub mod prompt;
pub mod replay_dependencies;
pub mod replay;
pub mod rng;
pub mod switch;
pub mod synth;
pub mod tool_shape;
pub mod scenario;
pub mod selection;
pub mod session_tree;
pub mod subagent;
pub mod trace;
pub mod trajectory_source;
