// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Legacy AgentX weka execution: a self-contained loader+runtime, selected by
//! `--weka-semantics legacy` (the default under an agentic-replay scenario).
//!
//! This is a separate execution path from graph-ir: it loads a WEKA trace source
//! (HF dataset or file) through the byte-exact [`crate::agentx`] loader, samples a
//! per-tree t\*, builds the agentic dispatch plan (warmup→profiling with byte-exact
//! cache-bust markers), fires it through the run's production transport honoring
//! the dispatch schedule on the real clock, and feeds each response into the same
//! shared record lane + metrics accumulator the graph/scheduled paths use — so it
//! produces engine-parity `profile_export.jsonl` and `profile_export_aiperf.json`.
//!
//! Gated on the `agentx` feature; the graph-ir path is untouched.

use std::sync::Arc;

use anyhow::Result;

use crate::engine::protocol_v2::AuthoredRunSpecV2;
use crate::engine::registry::{
    GraphWorkloadConfigV2, NativeTransportExecution, PreparedRunnerOperation, RunContext,
};

/// Prepare the legacy AgentX weka operation from the validated run + workload.
///
/// Captures the weka dataset descriptor, resolved endpoint/transport binding,
/// tokenizer, phases, metrics + artifact policy, and the scenario timing knobs
/// (t\* window, cache-bust target). The returned operation's `execute` owns its
/// own current-thread runtime (the transport is `!Send`).
pub fn prepare_legacy_agentx_operation(
    _run: &AuthoredRunSpecV2,
    _context: &RunContext,
    _workload: &GraphWorkloadConfigV2,
    _binding: Arc<dyn NativeTransportExecution>,
) -> Result<Box<dyn PreparedRunnerOperation>> {
    anyhow::bail!(
        "legacy AgentX weka execution is not yet fully wired; use --weka-semantics graph-ir"
    )
}
