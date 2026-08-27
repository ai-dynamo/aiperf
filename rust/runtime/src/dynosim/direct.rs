// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Plugin-owned Dynosim direct-dispatch execution leaf.
//!
//! This module is the implementation leaf for the `dynosim` feature's
//! direct (non-replay, non-scheduled) graph-execution path. It contains the
//! direct-graph execution primitives that Task 34 (transport-dynosim plugin)
//! will equality-copy.
//!
//! The host adapter in `dynosim.rs` retains `OfflineEngineFactory`,
//! `OfflineScheduledExecution`, `OfflineGraphExecution`, and Config-v2 wiring.
//! `engine/offline_execution.rs` retains the `NativeTransportExecution` binding
//! for offline dynosim runs.
//!
//! # Split boundary (Task 6, Phase 3)
//!
//! The full content extraction from `dynosim.rs` is performed in a follow-up
//! commit; this file establishes the `dynosim::direct` module path and the
//! candidate inventory entry used by Tasks 34 to locate this leaf for
//! equality-copy into the transport-dynosim plugin crate.
//!
//! Items to be extracted here:
//! - `OfflineDirectGraphReport` — per-run report for direct-dispatch graph execution
//! - Direct-path event sink and fabrication helpers
//! - `run_offline_graph_direct` — the direct-graph execution entry point

// Re-export the direct-dispatch types that Task 34 (transport-dynosim plugin)
// will equality-copy from this module.
pub(crate) use crate::dynosim::OfflineDirectGraphReport;
