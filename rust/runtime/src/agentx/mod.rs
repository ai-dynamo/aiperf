// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Standalone, byte-exact Rust port of the Python **AgentX v1.0** feature
//! (WEKA trace replay + agentic-replay timing + scenario locks).
//!
//! Its reconstruction, timing, and scheduling logic is deliberately separate
//! from the graph-ir recorded path (`crate::graph::recorded`); the only shared
//! seams are infrastructural — the public-dataset row loader
//! ([`crate::dataset::load_raw_rows`]) behind [`crate::agentx::hf_dataset`], the
//! virtual-clock driver (`crate::graph::runtime::drive_sim`) behind
//! [`crate::agentx::replay`], and the canonical recorded-agent fixture the
//! [`crate::agentx::scenario`] locks are applied to. It is
//! a faithful 1:1 parity port of the Python implementation under
//! `src/aiperf/dataset/loader/weka_*.py` and `src/aiperf/timing/`, intended to be
//! deleted wholesale once graph-ir supersedes AgentX. See
//! `specs/agentx-rust-port.md` for the design record.
//!
//! Parity is proven, not asserted: reconstruction is deterministic given
//! `(seed, trace)`, and the reconstruction, corpus, scheduling, and dependency
//! modules diff byte-for-byte against Python-generated goldens
//! (`tools/agentx_*_golden.py`) over the in-repo `tests/fixtures/weka_traces*/`.

pub mod cache_bust;
pub mod chains;
pub mod config;
pub mod corpus;
pub mod export;
/// Warmup-to-profile handoff observation for the accelerated cache-warmup
/// substage (pure recorder + gate/recorder observer bundle).
pub mod handoff;
/// HuggingFace-hosted WEKA trace dataset download (JSONL/JSON/CSV always;
/// Parquet under the `parquet` feature).
pub mod hf_dataset;
pub mod idle_gap;
pub mod loader;
pub mod metrics;
pub mod plan;
pub mod prepass;
pub mod prompt;
pub mod replay;
pub mod replay_dependencies;
/// Byte-exact port of the Python replay interval-barrier coordinator
/// (`ReplayBarrierCoordinator`); single-central-driver, no async/I/O.
pub mod replay_gate;
pub mod rng;
pub mod scenario;
pub mod selection;
pub mod session_tree;
pub mod subagent;
pub mod switch;
pub mod synth;
pub mod tool_shape;
pub mod trace;
pub mod trajectory_source;
/// Compose reconstructed WEKA trajectories into a linear scheduled `Dataset` for
/// the agentic-replay timing mode.
pub mod weka_dataset;
pub mod wire;
