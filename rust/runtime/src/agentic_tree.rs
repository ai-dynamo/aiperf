// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Always-compiled subagent tree-spec side channel for the `agentic_replay`
//! timing mode.
//!
//! [`TreeSpec`] is the DAG-free description of one session tree (a root
//! conversation, its recursive subagent children, and the root turns that must
//! block on child completion). It is intentionally defined here — outside the
//! `agentx`-gated [`crate::agentic_replay`] module — so it can be named in the
//! shared native-run plumbing (`PreparedDatasetInput`, the phase-plan builder)
//! regardless of whether the `agentx` feature is active. The gate that consumes
//! these specs ([`crate::agentic_replay::TreeGate`]) lives behind `agentx`; a
//! build without it simply never populates a non-empty `Vec<TreeSpec>` and the
//! join gate stays a pass-through.

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
