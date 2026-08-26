// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Always-compiled subagent tree-spec side channel for the `agentic_replay`
//! timing mode.
//!
//! [`TreeSpec`] is the DAG-free description of one session tree (a root
//! conversation, its recursive subagent children, and the root turns that must
//! block on child completion). It is intentionally defined here — outside the
//! [`crate::agentic_replay`] module — so it can be named in the shared
//! native-run plumbing (`PreparedDatasetInput`, the phase-plan builder) without
//! reaching into the replay mode's own module. A non-agentic run simply never
//! populates a non-empty `Vec<TreeSpec>`, so the gate that consumes these specs
//! ([`crate::agentic_replay::TreeGate`]) stays a pass-through.

/// Declarative description of one session tree used to build a
/// [`crate::agentic_replay::TreeGate`].
///
/// - `root` is the depth-0 root conversation/correlation id.
/// - `children` are the recursive descendant (subagent/spawn) correlation ids
///   owned by this tree.
/// - `join_turns` are the root turn indices that must block until a specified
///   set of children have terminated (a "join"): each entry pairs a
///   `turn_index` with the child ids required to be terminal before that turn
///   may dispatch. The `turn_index` is the index **as it appears in the
///   profiling (post-t\*, history-sliced) conversation**, since the specs are
///   built from the already-sliced profiling conversations.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TreeSpec {
    /// Depth-0 root correlation id.
    pub root: String,
    /// Recursive descendant correlation ids owned by this tree.
    pub children: Vec<String>,
    /// Root join points: `(turn_index, required_child_ids)`.
    pub join_turns: Vec<(usize, Vec<String>)>,
}

/// Opaque cross-phase carrier slot for the accelerated cache-warmup handoff.
///
/// The concrete payload (`Arc<Mutex<Option<LegacyWarmupHandoff>>>`) is named by
/// [`crate::agentic_replay`] over the [`crate::agentx::handoff`] type; this
/// type-erased alias lets the native phase-plan plumbing thread the carrier
/// without naming it. The WARMUP agentic phase downcasts and populates
/// it at finalize; PROFILING downcasts and reads it. Non-agentic (and non-accelerated)
/// runs carry [`empty_warmup_handoff_carrier`], which no phase ever downcasts.
pub type WarmupHandoffCarrierAny = std::sync::Arc<dyn std::any::Any + Send + Sync>;

/// An empty (never-populated) accelerated-warmup carrier for non-agentic and
/// non-accelerated runs. Downcasting it to the typed carrier fails by design, so
/// the profiling path stays exactly the non-accelerated behavior.
pub fn empty_warmup_handoff_carrier() -> WarmupHandoffCarrierAny {
    std::sync::Arc::new(())
}
