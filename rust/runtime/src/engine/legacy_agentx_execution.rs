// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Legacy AgentX weka execution seam, reserved for `--weka-semantics legacy`
//! (the value [`crate::config::resolve`] derives by default under an
//! agentic-replay scenario).
//!
//! The intended path is separate from graph-ir: load a WEKA trace source (HF
//! dataset or file) through the byte-exact [`crate::agentx`] loader, sample a
//! per-tree t\*, build the agentic dispatch plan (warmup→profiling with
//! byte-exact cache-bust markers), fire it through the run's production
//! transport honoring the dispatch schedule on the real clock, and feed each
//! response into the same shared record lane + metrics accumulator the
//! graph/scheduled paths use.
//!
//! None of that is wired here yet: the one entry point below refuses
//! unconditionally and has no callers, so the graph-ir path is the only weka
//! runtime this module affects.

use std::sync::Arc;

use anyhow::Result;

use crate::engine::protocol_v2::AuthoredRunSpecV2;
use crate::engine::registry::{
    GraphWorkloadConfigV2, NativeTransportExecution, PreparedRunnerOperation, RunContext,
};

/// Prepare the legacy AgentX weka operation from the validated run + workload.
///
/// # Errors
///
/// Always. The preparation is unimplemented, so every call returns an error
/// directing the caller at `--weka-semantics graph-ir`; no argument is read.
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
