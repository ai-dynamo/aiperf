// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reconstruction plan assembly + the trace-wide shared prefix-cache metric
//! builder, ported from `weka_trace._build_reconstruction_plans` /
//! `_build_shared_metric_values`.
//!
//! `hash_id_scope: "local"` means one namespace per trace file, so a block first
//! sent by the parent is a cache hit when a subagent child re-sends it (and vice
//! versa). The shared metric values are computed over ONE per-trace seen-set
//! consumed in global `(t, outer_idx, stream_idx, k)` order across the parent
//! and all active child conversations.
//!
//! Flat-chain plans and the idle-gap time-warp are not yet wired here.

use std::collections::{HashMap, HashSet};

use crate::agentx::loader::NormalReq;
use crate::agentx::prepass::{compute_shared_prefix_cache_metrics, MetricRecord, SortKey};
use crate::agentx::subagent::ChildPlan;

/// A parent (root) conversation plan (Python `_ParentPlan`).
#[derive(Debug, Clone)]
pub struct ParentPlan {
    /// Root trace id.
    pub trace_id: String,
    /// Retained top-level normal requests as `(outer_idx, request)`.
    pub normals: Vec<(i64, NormalReq)>,
    /// Outer indices of the trace's subagent markers, indexed by subagent index.
    pub subagent_outer_indices: Vec<i64>,
    /// Trace block size.
    pub block_size: i64,
}

/// Subagent indices with no preceding parent turn — dropped from emission
/// (Python `_dropped_subagent_indices`).
pub fn dropped_subagent_indices(plan: &ParentPlan) -> HashSet<usize> {
    let normal_outers: Vec<i64> = plan.normals.iter().map(|(oi, _)| *oi).collect();
    let mut dropped = HashSet::new();
    for (sa_index, &sa_outer) in plan.subagent_outer_indices.iter().enumerate() {
        if !normal_outers.iter().any(|&oi| oi < sa_outer) {
            dropped.insert(sa_index);
        }
    }
    dropped
}

/// Per-trace `{(session_id, k): (hits, total)}` from one shared seen-set in
/// global order (Python `_build_shared_metric_values`). Dropped subagents are
/// excluded to match emission.
pub fn build_shared_metric_values(
    parents: &[ParentPlan],
    children: &[ChildPlan],
) -> HashMap<String, HashMap<(String, i64), (i64, i64)>> {
    let mut out: HashMap<String, HashMap<(String, i64), (i64, i64)>> = HashMap::new();

    for plan in parents {
        let mut records: Vec<MetricRecord> = Vec::new();

        // Parent normals: sort key (t, outer, 0, 0) — matches Python exactly.
        for (k, (outer_idx, req)) in plan.normals.iter().enumerate() {
            records.push(MetricRecord {
                sort_key: SortKey {
                    absolute_t: req.t,
                    outer_idx: *outer_idx,
                    stream_idx: 0,
                    k: 0,
                },
                session_id: plan.trace_id.clone(),
                k: k as i64,
                hash_ids: req.hash_ids.clone(),
            });
        }

        let dropped = dropped_subagent_indices(plan);
        // Active child conversations: sort key (t, sa_outer, chain_index, k).
        for cp in children
            .iter()
            .filter(|cp| cp.parent_trace_id == plan.trace_id && !dropped.contains(&cp.subagent_index))
        {
            let sa_outer = plan.subagent_outer_indices[cp.subagent_index];
            for (k, creq) in cp.requests.iter().enumerate() {
                records.push(MetricRecord {
                    sort_key: SortKey {
                        absolute_t: creq.t,
                        outer_idx: sa_outer,
                        stream_idx: cp.chain_index as i64,
                        k: k as i64,
                    },
                    session_id: cp.session_id.clone(),
                    k: k as i64,
                    hash_ids: creq.hash_ids.clone(),
                });
            }
        }

        out.insert(plan.trace_id.clone(), compute_shared_prefix_cache_metrics(records));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nreq(t: f64, hash_ids: &[i64]) -> NormalReq {
        NormalReq {
            t,
            api_time: Some(0.1),
            think_time: None,
            model: "m".into(),
            hash_ids: hash_ids.to_vec(),
            input_length: hash_ids.len() as i64 * 4,
            output_length: 4,
            input_types: vec![],
            stop: String::new(),
        }
    }

    #[test]
    fn dropped_when_no_preceding_parent_turn() {
        let plan = ParentPlan {
            trace_id: "t".into(),
            normals: vec![(3, nreq(0.0, &[1]))],
            subagent_outer_indices: vec![1, 5], // sa0 at outer 1 (< 3? no normal < 1) dropped; sa1 at 5 kept
            block_size: 4,
        };
        let dropped = dropped_subagent_indices(&plan);
        assert!(dropped.contains(&0)); // no normal outer < 1
        assert!(!dropped.contains(&1)); // normal outer 3 < 5
    }

    #[test]
    fn parent_child_share_prefix_cache_across_conversations() {
        // Parent turn sends [1,2,3]; the subagent child later re-sends [1,2] ->
        // 2 prefix hits under the shared local namespace.
        let plan = ParentPlan {
            trace_id: "t".into(),
            normals: vec![(0, nreq(0.0, &[1, 2, 3]))],
            subagent_outer_indices: vec![1],
            block_size: 4,
        };
        let child = ChildPlan {
            session_id: "t::sa:a".into(),
            parent_trace_id: "t".into(),
            subagent_index: 0,
            source_outer_idx: 1,
            chain_index: 0,
            requests: vec![nreq(1.0, &[1, 2])],
            request_inner_indices: vec![0],
            block_size: 4,
            init_tool_tokens: 0,
            init_system_tokens: 0,
            is_aux: false,
        };
        let m = build_shared_metric_values(&[plan], &[child]);
        let trace = &m["t"];
        assert_eq!(trace[&("t".to_string(), 0)], (0, 3)); // parent turn 0: no prior
        assert_eq!(trace[&("t::sa:a".to_string(), 0)], (2, 2)); // child sees [1,2]
    }
}
