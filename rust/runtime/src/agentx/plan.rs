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

use crate::agentx::chains::{
    ChainReq, chain_init_tokens, detect_agent_chains, is_aux_chain, is_reduction_chain,
    split_off_preamble, worker_group_assignment,
};
use crate::agentx::config::{TITLE_GEN_MAX_OUTPUT_TOKENS, WekaConfig};
use crate::agentx::loader::NormalReq;
use crate::agentx::prepass::{MetricRecord, SortKey, compute_shared_prefix_cache_metrics};
use crate::agentx::subagent::{ChildPlan, worker_suffix};

// Per-trace, per-session cache-hit and cache-total metrics keyed by turn.
type SharedMetricValues = HashMap<String, HashMap<(String, i64), (i64, i64)>>;

/// A detected flat worker-chain conversation plan (Python `_FlatChainPlan`).
#[derive(Debug, Clone, PartialEq)]
pub struct FlatChainPlan {
    /// Child session id (`{trace}::{suffix}`).
    pub session_id: String,
    /// Root trace id.
    pub parent_trace_id: String,
    /// Dense per-trace worker index (0-based).
    pub chain_index: usize,
    /// The chain's requests as `(outer_idx, request)`.
    pub requests: Vec<(i64, NormalReq)>,
    /// Turn-0 tools-prefix attribution.
    pub init_tool_tokens: i64,
    /// Turn-0 system-prefix attribution.
    pub init_system_tokens: i64,
    /// Phase-1 fork parent chain index (log/DAG only in v1).
    pub fork_parent_chain: Option<usize>,
    /// Blocks shared with the fork tail.
    pub fork_depth: i64,
    /// Trace block size.
    pub block_size: i64,
    /// True when an aux/reduction sidecar.
    pub is_aux: bool,
}

fn to_chain_req(n: &NormalReq) -> ChainReq {
    ChainReq {
        t: n.t,
        api_time: n.api_time,
        model: n.model.clone(),
        hash_ids: n.hash_ids.clone(),
        input_length: n.input_length,
        output_length: n.output_length,
    }
}

/// Run LCP chain detection on a trace's retained top-level requests, splitting
/// off detected flat worker chains (Python `_detect_and_split_flat_chains`).
/// Returns the (possibly reduced) main-chain normals and the flat-chain plans.
pub fn detect_and_split_flat_chains(
    trace_id: &str,
    normals: &[(i64, NormalReq)],
    tool_tokens: i64,
    system_tokens: i64,
    block_size: i64,
    cfg: &WekaConfig,
) -> (Vec<(i64, NormalReq)>, Vec<FlatChainPlan>) {
    let normals_by_outer: HashMap<i64, NormalReq> =
        normals.iter().map(|(oi, r)| (*oi, r.clone())).collect();
    let detect_input: Vec<(i64, ChainReq)> = normals
        .iter()
        .map(|(oi, r)| (*oi, to_chain_req(r)))
        .collect();
    let (preamble, detect_normals) = split_off_preamble(&detect_input, TITLE_GEN_MAX_OUTPUT_TOKENS);
    let detection = detect_agent_chains(
        detect_normals,
        cfg.seam_max_gap_seconds,
        cfg.seam_min_overlap_ratio,
    );
    if detection.worker_indices.is_empty() {
        return (normals.to_vec(), Vec::new());
    }

    let main_chain = &detection.chains[detection.main_index];
    let main_first_hash = main_chain
        .requests
        .iter()
        .find(|(_, r)| !r.hash_ids.is_empty())
        .map(|(_, r)| r.hash_ids.clone())
        .unwrap_or_default();
    let main_peak_isl = main_chain
        .requests
        .iter()
        .map(|(_, r)| r.input_length)
        .max()
        .unwrap_or(0);
    let main_model: Option<String> = main_chain.requests.first().map(|(_, r)| r.model.clone());
    let wg_coords = worker_group_assignment(&detection, cfg.worker_group_min);

    let mut flat_plans: Vec<FlatChainPlan> = Vec::new();
    for (n, &ci) in detection.worker_indices.iter().enumerate() {
        let chain = &detection.chains[ci];
        let chain_reqs: Vec<ChainReq> = chain.requests.iter().map(|(_, r)| r.clone()).collect();
        let (init_tool, init_system) = chain_init_tokens(
            tool_tokens,
            system_tokens,
            block_size,
            &main_first_hash,
            &chain.requests[0].1.hash_ids,
        );
        let aux = is_aux_chain(
            &chain_reqs,
            main_peak_isl,
            cfg.aux_max_requests,
            cfg.aux_isl_ratio,
            cfg.aux_isl_floor,
            main_model.as_deref(),
            cfg.aux_cross_model,
        );
        let reduction = !aux
            && is_reduction_chain(
                &chain_reqs,
                cfg.aux_reduction_osl_max,
                cfg.aux_reduction_ratio,
                cfg.aux_isl_floor,
            );
        let wg_coord = if !aux && !reduction {
            wg_coords.get(&ci).copied()
        } else {
            None
        };
        let suffix = worker_suffix(n, aux, reduction, wg_coord);
        let requests: Vec<(i64, NormalReq)> = chain
            .requests
            .iter()
            .map(|(oi, _)| (*oi, normals_by_outer[oi].clone()))
            .collect();
        flat_plans.push(FlatChainPlan {
            session_id: format!("{trace_id}::{suffix}"),
            parent_trace_id: trace_id.to_string(),
            chain_index: n,
            requests,
            init_tool_tokens: init_tool,
            init_system_tokens: init_system,
            fork_parent_chain: chain.fork.as_ref().and_then(|f| f.parent_chain),
            fork_depth: chain.fork.as_ref().map(|f| f.depth).unwrap_or(0),
            block_size,
            is_aux: aux || reduction,
        });
    }

    let mut main_normals: Vec<(i64, NormalReq)> = main_chain
        .requests
        .iter()
        .map(|(oi, _)| (*oi, normals_by_outer[oi].clone()))
        .collect();
    if !preamble.is_empty() {
        let mut combined: Vec<(i64, NormalReq)> = preamble
            .iter()
            .map(|(oi, _)| (*oi, normals_by_outer[oi].clone()))
            .collect();
        combined.extend(main_normals);
        combined.sort_by(|a, b| (a.1.t, a.0).partial_cmp(&(b.1.t, b.0)).unwrap());
        main_normals = combined;
    }
    (main_normals, flat_plans)
}

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
    flats: &[FlatChainPlan],
) -> SharedMetricValues {
    let mut out: SharedMetricValues = HashMap::new();

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

        // Flat worker chains: sort key (t, outer, 0, 0).
        for fp in flats
            .iter()
            .filter(|fp| fp.parent_trace_id == plan.trace_id)
        {
            for (k, (outer_idx, req)) in fp.requests.iter().enumerate() {
                records.push(MetricRecord {
                    sort_key: SortKey {
                        absolute_t: req.t,
                        outer_idx: *outer_idx,
                        stream_idx: 0,
                        k: 0,
                    },
                    session_id: fp.session_id.clone(),
                    k: k as i64,
                    hash_ids: req.hash_ids.clone(),
                });
            }
        }

        let dropped = dropped_subagent_indices(plan);
        // Active child conversations: sort key (t, sa_outer, chain_index, k).
        for cp in children.iter().filter(|cp| {
            cp.parent_trace_id == plan.trace_id && !dropped.contains(&cp.subagent_index)
        }) {
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

        out.insert(
            plan.trace_id.clone(),
            compute_shared_prefix_cache_metrics(records),
        );
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
    fn flat_split_extracts_worker_chain() {
        // r1 diverges from [1,2]; r2 extends the longer state -> r1 spawns a
        // flat worker conversation (::fa:000 / ::aux:000 depending on size).
        let normals = vec![
            (0, nreq(0.0, &[1, 2])),
            (1, nreq(1.0, &[1, 9])),
            (2, nreq(2.0, &[1, 2, 3])),
        ];
        let (main_normals, flats) =
            detect_and_split_flat_chains("t", &normals, 0, 0, 4, &WekaConfig::default());
        assert_eq!(flats.len(), 1);
        assert!(flats[0].session_id.starts_with("t::"));
        // The spawned request is pulled out of the main chain.
        assert!(main_normals.iter().all(|(oi, _)| *oi != 1));
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
        let m = build_shared_metric_values(&[plan], &[child], &[]);
        let trace = &m["t"];
        assert_eq!(trace[&("t".to_string(), 0)], (0, 3)); // parent turn 0: no prior
        assert_eq!(trace[&("t::sa:a".to_string(), 0)], (2, 2)); // child sees [1,2]
    }
}
