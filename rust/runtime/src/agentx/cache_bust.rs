// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Deterministic per-conversation cache-bust marker builder, ported from
//! `src/aiperf/timing/strategies/cache_bust.py`.
//!
//! The same `(benchmark_id, recycle_pass, trajectory_index, trace_id)` always
//! yields the same digest (reproducible across reruns). The marker is a property
//! of a trajectory tree: the first member to resolve mints it, every other member
//! reuses the stored value (idempotent via a ledger). Position controls only
//! whitespace placement, not the digest.

use std::collections::HashMap;

use sha2::{Digest, Sha256};

/// Where the cache-bust marker is placed (Python `CacheBustTarget`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheBustTarget {
    /// No marker.
    None,
    /// Prepended to the system prefix.
    SystemPrefix,
    /// Appended to the system prefix.
    SystemSuffix,
    /// Prepended to the first turn.
    FirstTurnPrefix,
    /// Appended to the first turn.
    FirstTurnSuffix,
    /// Constant marker applied only during warmup at the system message.
    WarmupIsolationSystem,
    /// Constant marker applied only during warmup at the first user turn.
    WarmupIsolationFirstTurn,
}

impl CacheBustTarget {
    fn is_prefix(self) -> bool {
        matches!(
            self,
            CacheBustTarget::SystemPrefix | CacheBustTarget::FirstTurnPrefix
        )
    }
}

const DIGEST_LEN: usize = 12; // 12 hex chars = 48 bits
const SUFFIX_SEP: &str = "::";

/// Strip any descendant suffix (`::sa:`/`::fa:`/`:sN`) to the root trace id.
/// Every member of a trajectory tree shares one base trace id.
pub fn base_trace_id(conversation_id: &str) -> &str {
    conversation_id
        .split(SUFFIX_SEP)
        .next()
        .unwrap_or(conversation_id)
}

/// Render the marker text for the given inputs and target position (Python
/// `build_cache_bust_marker`). `None` when the target is `None`.
pub fn build_cache_bust_marker(
    benchmark_id: &str,
    recycle_pass: i64,
    trajectory_index: i64,
    trace_id: &str,
    target: CacheBustTarget,
) -> Option<String> {
    if target == CacheBustTarget::None {
        return None;
    }
    if matches!(
        target,
        CacheBustTarget::WarmupIsolationSystem | CacheBustTarget::WarmupIsolationFirstTurn
    ) {
        return Some("[warmup]\n\n".to_string());
    }
    let unique = format!("{benchmark_id}:{recycle_pass}:{trajectory_index}:{trace_id}");
    let digest = Sha256::digest(unique.as_bytes());
    // First DIGEST_LEN hex chars = first DIGEST_LEN/2 digest bytes.
    let hex: String = digest
        .iter()
        .take(DIGEST_LEN / 2)
        .map(|b| format!("{b:02x}"))
        .collect();
    let rid = format!("[rid:{hex}]");
    if target.is_prefix() {
        Some(format!("{rid}\n\n"))
    } else {
        Some(format!("\n\n{rid}"))
    }
}

/// The per-run cache-bust ledger (Python duck-typed ledger): the minted marker
/// per tree and the recycle-pass counter per base trace id.
#[derive(Debug, Default)]
pub struct CacheBustLedger {
    /// `root_correlation_id` -> resolved marker (`None` recorded explicitly).
    pub session_marker: HashMap<String, Option<String>>,
    /// base trace id -> recycle pass counter.
    pub recycle_pass: HashMap<String, i64>,
}

/// Resolve the cache-bust marker for a trajectory TREE, idempotently (Python
/// `resolve_tree_marker`). The first member mints it (bumping `recycle_pass`
/// once); every other member reuses the stored value.
pub fn resolve_tree_marker(
    ledger: &mut CacheBustLedger,
    root_correlation_id: &str,
    benchmark_id: &str,
    trajectory_index: i64,
    conversation_id: &str,
    target: CacheBustTarget,
) -> Option<String> {
    if let Some(existing) = ledger.session_marker.get(root_correlation_id) {
        return existing.clone();
    }
    if target == CacheBustTarget::None {
        ledger
            .session_marker
            .insert(root_correlation_id.to_string(), None);
        return None;
    }
    let base = base_trace_id(conversation_id).to_string();
    let new_pass = ledger.recycle_pass.get(&base).copied().unwrap_or(-1) + 1;
    ledger.recycle_pass.insert(base.clone(), new_pass);
    let marker = build_cache_bust_marker(benchmark_id, new_pass, trajectory_index, &base, target);
    ledger
        .session_marker
        .insert(root_correlation_id.to_string(), marker.clone());
    marker
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_python_golden_markers() {
        // build_cache_bust_marker("bench1", 0, 3, "trace_x", <target>).
        assert_eq!(
            build_cache_bust_marker("bench1", 0, 3, "trace_x", CacheBustTarget::FirstTurnPrefix),
            Some("[rid:b7fa20d59cf9]\n\n".to_string())
        );
        assert_eq!(
            build_cache_bust_marker("bench1", 0, 3, "trace_x", CacheBustTarget::SystemSuffix),
            Some("\n\n[rid:b7fa20d59cf9]".to_string())
        );
        assert_eq!(
            build_cache_bust_marker("b", 0, 0, "t", CacheBustTarget::None),
            None
        );
    }

    #[test]
    fn warmup_isolation_targets_use_constant_marker() {
        assert_eq!(
            build_cache_bust_marker(
                "bench-a",
                7,
                3,
                "trace-a",
                CacheBustTarget::WarmupIsolationSystem,
            ),
            Some("[warmup]\n\n".to_string())
        );
        assert_eq!(
            build_cache_bust_marker(
                "bench-b",
                0,
                99,
                "trace-b",
                CacheBustTarget::WarmupIsolationFirstTurn,
            ),
            Some("[warmup]\n\n".to_string())
        );
    }

    #[test]
    fn base_trace_id_strips_descendant_suffix() {
        assert_eq!(base_trace_id("trace_x::sa:agent_001"), "trace_x");
        assert_eq!(base_trace_id("trace_x::fa:003"), "trace_x");
        assert_eq!(base_trace_id("trace_x"), "trace_x");
    }

    #[test]
    fn ledger_mints_once_per_tree_and_bumps_pass() {
        let mut ledger = CacheBustLedger::default();
        let m1 = resolve_tree_marker(
            &mut ledger,
            "root1",
            "bench",
            0,
            "trace_x::sa:a",
            CacheBustTarget::FirstTurnPrefix,
        );
        // Second member of the same tree reuses the stored marker.
        let m2 = resolve_tree_marker(
            &mut ledger,
            "root1",
            "bench",
            0,
            "trace_x",
            CacheBustTarget::FirstTurnPrefix,
        );
        assert_eq!(m1, m2);
        assert_eq!(ledger.recycle_pass["trace_x"], 0);
        // A new tree on the same base bumps the recycle pass to 1.
        resolve_tree_marker(
            &mut ledger,
            "root2",
            "bench",
            1,
            "trace_x",
            CacheBustTarget::FirstTurnPrefix,
        );
        assert_eq!(ledger.recycle_pass["trace_x"], 1);
    }
}
