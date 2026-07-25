// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Subagent expansion, ported from
//! `weka_trace._expand_subagent_to_child_plans`.
//!
//! Runs the same hash-id LCP spawn + seam-join detection NESTED on a subagent's
//! inner requests: the chain containing the first retained request keeps the
//! `::sa:{agent_id}` session id; every spawned chain becomes a sibling child
//! (`::sa:{agent_id}:fa:NNN`, `:aux:NNN`, or `:wg:GGG_MMM`). Inner timestamps
//! are normalized to root-trace coordinates first.

use crate::agentx::chains::{
    chain_init_tokens, detect_agent_chains, is_aux_chain, is_reduction_chain, split_off_preamble,
    worker_group_assignment, ChainReq,
};
use crate::agentx::config::{WekaConfig, JOIN_EPSILON_SECONDS, TITLE_GEN_MAX_OUTPUT_TOKENS};
use crate::agentx::loader::NormalReq;
use crate::agentx::trace::{WekaInnerRequest, WekaSubagentEntry};

/// One subagent child conversation plan (Python `_ChildPlan`, fields the port
/// currently consumes).
#[derive(Debug, Clone, PartialEq)]
pub struct ChildPlan {
    /// Child session id (`::sa:{agent_id}` or a worker-suffixed sibling).
    pub session_id: String,
    /// Originating root trace id.
    pub parent_trace_id: String,
    /// Index of the subagent entry within the parent's request list.
    pub subagent_index: usize,
    /// Outer index of the subagent marker in the parent's request list.
    pub source_outer_idx: i64,
    /// 0 = the subagent's main chain; >0 = a spawned chain.
    pub chain_index: usize,
    /// The chain's requests (root-trace-normalized `t`), in chain order.
    pub requests: Vec<NormalReq>,
    /// Original zero-based indexes within `entry.requests` aligned with `requests`.
    pub request_inner_indices: Vec<usize>,
    /// Trace block size.
    pub block_size: i64,
    /// Turn-0 tools-prefix attribution for this chain.
    pub init_tool_tokens: i64,
    /// Turn-0 system-prefix attribution for this chain.
    pub init_system_tokens: i64,
    /// True when a spawned chain is an auxiliary one-shot sidecar.
    pub is_aux: bool,
}

/// A subagent inner request timestamp in root-trace coordinates (Python
/// `_subagent_request_absolute_t`): a child timestamp before the spawn marker
/// is treated as relative.
fn subagent_request_absolute_t(entry_t: f64, req_t: f64) -> f64 {
    if req_t + JOIN_EPSILON_SECONDS < entry_t {
        entry_t + req_t
    } else {
        req_t
    }
}

fn inner_to_normal(r: &WekaInnerRequest, absolute_t: f64) -> NormalReq {
    match r {
        WekaInnerRequest::Normal(n) => NormalReq {
            t: absolute_t,
            api_time: n.api_time,
            think_time: n.think_time,
            model: n.model.clone(),
            hash_ids: n.hash_ids.clone(),
            input_length: n.input_length,
            output_length: n.output_length,
            input_types: n.input_types.clone(),
            stop: n.stop.clone(),
        },
        WekaInnerRequest::Streaming(s) => NormalReq {
            t: absolute_t,
            api_time: s.api_time,
            think_time: s.think_time,
            model: s.model.clone(),
            hash_ids: s.hash_ids.clone(),
            input_length: s.input_length,
            output_length: s.output_length,
            input_types: s.input_types.clone(),
            stop: s.stop.clone(),
        },
    }
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

/// Session-id suffix for a detected worker chain (Python `_worker_suffix`).
fn worker_suffix(n: usize, is_aux: bool, is_reduction: bool, wg_coord: Option<(i64, i64)>) -> String {
    if is_aux || is_reduction {
        return format!("aux:{n:03}");
    }
    if let Some((group, member)) = wg_coord {
        return format!("wg:{group:03}_{member:03}");
    }
    format!("fa:{n:03}")
}

/// First non-empty hash list in a slice of requests (else empty).
fn first_hash(reqs: &[NormalReq]) -> Vec<i64> {
    reqs.iter()
        .find(|r| !r.hash_ids.is_empty())
        .map(|r| r.hash_ids.clone())
        .unwrap_or_default()
}

/// Partition a subagent's inner requests into per-chain child plans.
pub fn expand_subagent_to_child_plans(
    trace_id: &str,
    sa_index: usize,
    source_outer_idx: i64,
    entry: &WekaSubagentEntry,
    block_size: i64,
    cfg: &WekaConfig,
) -> Vec<ChildPlan> {
    // Normalize inner timestamps to root-trace coordinates. normalized_idx ==
    // inner_idx (enumerate is in order), matching Python's identity map.
    let normalized: Vec<NormalReq> = entry
        .requests
        .iter()
        .map(|r| {
            let raw_t = match r {
                WekaInnerRequest::Normal(n) => n.t,
                WekaInnerRequest::Streaming(s) => s.t,
            };
            inner_to_normal(r, subagent_request_absolute_t(entry.t, raw_t))
        })
        .collect();

    // Per-chain lists of normalized indices, and worker-group coordinates.
    let chain_index_lists: Vec<Vec<usize>>;
    let chain_wg_coord: Vec<Option<(i64, i64)>>;
    let classify_main_idx: Vec<usize>;

    if !cfg.split_flattened_agents || normalized.is_empty() {
        let mut ordered: Vec<usize> = (0..normalized.len()).collect();
        ordered.sort_by(|&a, &b| {
            (normalized[a].t, a)
                .partial_cmp(&(normalized[b].t, b))
                .unwrap()
        });
        classify_main_idx = ordered.clone();
        chain_index_lists = vec![ordered];
        chain_wg_coord = vec![None];
    } else {
        // Detection input: (normalized_idx, ChainReq).
        let detect_input: Vec<(i64, ChainReq)> = (0..normalized.len())
            .map(|i| (i as i64, to_chain_req(&normalized[i])))
            .collect();
        let (preamble, detect_inner) =
            split_off_preamble(&detect_input, TITLE_GEN_MAX_OUTPUT_TOKENS);
        let detection = detect_agent_chains(
            detect_inner,
            cfg.seam_max_gap_seconds,
            cfg.seam_min_overlap_ratio,
        );

        let detected_main: Vec<usize> = detection.chains[detection.main_index]
            .requests
            .iter()
            .map(|(idx, _)| *idx as usize)
            .collect();
        classify_main_idx = detected_main.clone();

        let mut main_requests: Vec<usize> = detected_main;
        if !preamble.is_empty() {
            let mut combined: Vec<usize> = preamble.iter().map(|(idx, _)| *idx as usize).collect();
            combined.extend(main_requests.iter().copied());
            combined.sort_by(|&a, &b| {
                (normalized[a].t, a)
                    .partial_cmp(&(normalized[b].t, b))
                    .unwrap()
            });
            main_requests = combined;
        }

        let mut lists: Vec<Vec<usize>> = vec![main_requests];
        for &ci in &detection.worker_indices {
            lists.push(
                detection.chains[ci]
                    .requests
                    .iter()
                    .map(|(idx, _)| *idx as usize)
                    .collect(),
            );
        }
        let wg_coords = worker_group_assignment(&detection, cfg.worker_group_min);
        let mut coords: Vec<Option<(i64, i64)>> = vec![None];
        for &ci in &detection.worker_indices {
            coords.push(wg_coords.get(&ci).copied());
        }
        chain_index_lists = lists;
        chain_wg_coord = coords;
    }

    // Classification yardstick = the DETECTED main chain (preamble excluded).
    let classify_main: Vec<NormalReq> =
        classify_main_idx.iter().map(|&i| normalized[i].clone()).collect();
    let main_first_hash = first_hash(&classify_main);
    let main_peak_isl = classify_main
        .iter()
        .map(|r| r.input_length)
        .max()
        .unwrap_or(0);
    let main_model: Option<String> = classify_main.first().map(|r| r.model.clone());

    let mut plans: Vec<ChildPlan> = Vec::with_capacity(chain_index_lists.len());
    for (chain_idx, idx_list) in chain_index_lists.iter().enumerate() {
        let requests: Vec<NormalReq> = idx_list.iter().map(|&i| normalized[i].clone()).collect();
        let request_inner_indices: Vec<usize> = idx_list.clone();
        let (session_id, init_tool, init_system, is_aux);

        if chain_idx == 0 {
            session_id = format!("{trace_id}::sa:{}", entry.agent_id);
            init_tool = entry.tool_tokens;
            init_system = entry.system_tokens;
            is_aux = false;
        } else {
            let chain_first = first_hash(&requests);
            let (it, is_) = chain_init_tokens(
                entry.tool_tokens,
                entry.system_tokens,
                block_size,
                &main_first_hash,
                &chain_first,
            );
            init_tool = it;
            init_system = is_;
            let creqs: Vec<ChainReq> = requests.iter().map(to_chain_req).collect();
            let aux = is_aux_chain(
                &creqs,
                main_peak_isl,
                cfg.aux_max_requests,
                cfg.aux_isl_ratio,
                cfg.aux_isl_floor,
                main_model.as_deref(),
                cfg.aux_cross_model,
            );
            let reduction = !aux
                && is_reduction_chain(
                    &creqs,
                    cfg.aux_reduction_osl_max,
                    cfg.aux_reduction_ratio,
                    cfg.aux_isl_floor,
                );
            let wg_coord = if !aux && !reduction {
                chain_wg_coord[chain_idx]
            } else {
                None
            };
            let suffix = worker_suffix(chain_idx - 1, aux, reduction, wg_coord);
            session_id = format!("{trace_id}::sa:{}:{suffix}", entry.agent_id);
            is_aux = aux || reduction;
        }

        plans.push(ChildPlan {
            session_id,
            parent_trace_id: trace_id.to_string(),
            subagent_index: sa_index,
            source_outer_idx,
            chain_index: chain_idx,
            requests,
            request_inner_indices,
            block_size,
            init_tool_tokens: init_tool,
            init_system_tokens: init_system,
            is_aux,
        });
    }
    plans
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agentx::trace::WekaNormalRequest;

    fn inner(t: f64, model: &str, hash_ids: &[i64], in_len: i64, out_len: i64) -> WekaInnerRequest {
        WekaInnerRequest::Normal(WekaNormalRequest {
            t,
            model: model.to_string(),
            input_length: in_len,
            output_length: out_len,
            hash_ids: hash_ids.to_vec(),
            input_types: vec![],
            output_types: vec![],
            stop: String::new(),
            api_time: Some(0.1),
            think_time: None,
        })
    }

    fn entry(agent_id: &str, t: f64, reqs: Vec<WekaInnerRequest>) -> WekaSubagentEntry {
        WekaSubagentEntry {
            t,
            agent_id: agent_id.to_string(),
            subagent_type: "Explore".to_string(),
            duration_ms: Some(1000),
            total_tokens: None,
            tool_use_count: None,
            status: "completed".to_string(),
            requests: reqs,
            models: vec!["m".to_string()],
            tool_tokens: 0,
            system_tokens: 0,
        }
    }

    #[test]
    fn single_chain_subagent_keeps_sa_session_id() {
        let e = entry(
            "agent_001",
            10.0,
            vec![
                inner(10.0, "m", &[1], 4, 4),
                inner(11.0, "m", &[1, 2], 8, 4),
            ],
        );
        let plans = expand_subagent_to_child_plans("t0", 0, 5, &e, 4, &WekaConfig::default());
        assert_eq!(plans.len(), 1);
        assert_eq!(plans[0].session_id, "t0::sa:agent_001");
        assert_eq!(plans[0].chain_index, 0);
        assert_eq!(plans[0].request_inner_indices, vec![0, 1]);
    }

    #[test]
    fn relative_inner_timestamps_normalized_to_root() {
        // req t=0.5 < entry t=10 -> normalized to 10.5.
        let e = entry("a", 10.0, vec![inner(0.5, "m", &[1], 4, 4)]);
        let plans = expand_subagent_to_child_plans("t0", 0, 5, &e, 4, &WekaConfig::default());
        assert_eq!(plans[0].requests[0].t, 10.5);
    }

    #[test]
    fn empty_subagent_emits_one_empty_child() {
        let e = entry("a", 10.0, vec![]);
        let plans = expand_subagent_to_child_plans("t0", 0, 5, &e, 4, &WekaConfig::default());
        assert_eq!(plans.len(), 1);
        assert!(plans[0].requests.is_empty());
        assert_eq!(plans[0].session_id, "t0::sa:a");
    }
}
