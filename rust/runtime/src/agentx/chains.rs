// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Flattened-agent chain detection for Weka traces via hash_id LCP evidence,
//! ported from `src/aiperf/dataset/loader/weka_agent_chains.py`.
//!
//! Untagged agent fan-outs are recorded as interleaved flat top-level requests.
//! This partitions a trace's top-level requests back into per-agent chains and
//! classifies every hash-list divergence as a join seam (same agent continuing
//! after a context edit) or a spawn (a new agent forked from a shared prefix).
//! Phase 1 builds chains greedily (forking on every shrink); phase 2 splices the
//! elected continuation back onto tails whose longer state turned out dead.
//!
//! The interval-grouping helpers (`worker_group_assignment`,
//! `compute_chain_prefix_blocks`) are ported in a follow-up; this module covers
//! the core partition (`detect_agent_chains`) and the aux one-shot classifiers.

use std::collections::HashMap;

const EPSILON_SECONDS: f64 = 1e-6;

/// The request fields chain detection depends on (a projection of
/// `WekaNormalRequest`/`WekaStreamingRequest`).
#[derive(Debug, Clone)]
pub struct ChainReq {
    /// Request timestamp in seconds from conversation start.
    pub t: f64,
    /// Server processing time in seconds (interval duration).
    pub api_time: Option<f64>,
    /// Model identifier.
    pub model: String,
    /// KV-cache block hash ids.
    pub hash_ids: Vec<i64>,
    /// Input token count.
    pub input_length: i64,
    /// Output token count.
    pub output_length: i64,
}

/// Interval end in seconds; missing/negative/non-finite `api_time` counts as 0.
fn req_end(req: &ChainReq) -> f64 {
    let duration = match req.api_time {
        Some(d) if d.is_finite() => d,
        _ => 0.0,
    };
    req.t + duration.max(0.0)
}

/// Length of the longest common prefix of two hash slices (Python `_np_lcp` /
/// `longest_common_prefix`).
fn np_lcp(a: &[i64], b: &[i64]) -> i64 {
    let n = a.len().min(b.len());
    for i in 0..n {
        if a[i] != b[i] {
            return i as i64;
        }
    }
    n as i64
}

/// Where a chain split off another chain.
#[derive(Debug, Clone)]
pub struct ChainFork {
    /// Phase-1 index of the chain forked from; `None` = no shared context.
    /// Rewritten to the live (post-splice) chain index by phase 2.
    pub parent_chain: Option<usize>,
    /// Outer index of the tail request T this chain forked from.
    pub fork_outer_idx: Option<i64>,
    /// Blocks shared with T at fork time (LCP).
    pub depth: i64,
    /// `t` of this chain's first request.
    pub fork_time: f64,
}

/// One detected agent: a time-ordered run of requests.
#[derive(Debug, Clone)]
pub struct AgentChain {
    /// The chain's requests as `(outer_idx, request)` in `(t, outer_idx)` order.
    pub requests: Vec<(i64, ChainReq)>,
    /// How this chain came to exist; `None` only for the first chain.
    pub fork: Option<ChainFork>,
    /// Set by phase 2 when this chain was a join-seam continuation.
    pub spliced_into: Option<usize>,
    /// Outer index of the last hash-bearing request (phase-1 state).
    pub tail_outer_idx: i64,
    /// Hash array of the last hash-bearing request (phase-1 state).
    pub tail_hash: Vec<i64>,
    /// Interval end of the last hash-bearing request.
    pub tail_end: f64,
    /// Model of the last hash-bearing request (same-model continuation rule).
    pub tail_model: String,
}

impl AgentChain {
    fn empty() -> Self {
        Self {
            requests: Vec::new(),
            fork: None,
            spliced_into: None,
            tail_outer_idx: -1,
            tail_hash: Vec::new(),
            tail_end: 0.0,
            tail_model: String::new(),
        }
    }
    fn with_fork(fork: ChainFork) -> Self {
        let mut c = Self::empty();
        c.fork = Some(fork);
        c
    }
}

/// Output of [`detect_agent_chains`]. Spliced chains stay in `chains` for
/// fork-history but are excluded from `worker_indices`.
#[derive(Debug, Clone)]
pub struct ChainDetectionResult {
    /// All phase-1 chains, including spliced (dead) ones.
    pub chains: Vec<AgentChain>,
    /// Index of the chain owning the trace's first retained request.
    pub main_index: usize,
    /// Live non-main chains, ordered by first request `(t, outer_idx)`.
    pub worker_indices: Vec<usize>,
    /// Number of join-seam splices performed in phase 2.
    pub seams_merged: i64,
    /// Requests with empty `hash_ids` kept on the main chain as-is.
    pub unclassified_empty_hash: i64,
}

/// Outer index of the chain's last HASH-BEARING request (empty-hash rows are
/// invisible to detection).
fn last_hash_outer_idx(chain: &AgentChain) -> Option<i64> {
    chain
        .requests
        .iter()
        .rev()
        .find(|(_, r)| !r.hash_ids.is_empty())
        .map(|(oi, _)| *oi)
}

/// Chain whose tail is a complete prefix of `h`, ended by `t`, same `model`.
/// Deepest tail wins; ties go to the lowest chain index.
fn find_extension_target(chains: &[AgentChain], h: &[i64], t: f64, model: &str) -> Option<usize> {
    let mut best: Option<usize> = None;
    let mut best_len: i64 = -1;
    let hn = h.len() as i64;
    for (idx, c) in chains.iter().enumerate() {
        let tl = c.tail_hash.len() as i64;
        if tl == 0 || tl > hn || tl <= best_len {
            continue;
        }
        if c.tail_model != model {
            continue;
        }
        if c.tail_end > t + EPSILON_SECONDS {
            continue;
        }
        if c.tail_hash[(tl - 1) as usize] != h[(tl - 1) as usize] {
            continue;
        }
        if h[..tl as usize] == c.tail_hash[..] {
            best = Some(idx);
            best_len = tl;
        }
    }
    best
}

/// Chain tail with the deepest LCP against `h` (ties: deeper tail, then lower
/// index). Returns `(None, 0)` when nothing shares a prefix.
fn max_lcp_chain(chains: &[AgentChain], h: &[i64]) -> (Option<usize>, i64) {
    let mut best_idx: Option<usize> = None;
    let mut best_key = (0i64, 0i64);
    for (idx, c) in chains.iter().enumerate() {
        if c.tail_hash.is_empty() {
            continue;
        }
        let d = np_lcp(&c.tail_hash, h);
        if d == 0 {
            continue;
        }
        let key = (d, c.tail_hash.len() as i64);
        if key > best_key {
            best_idx = Some(idx);
            best_key = key;
        }
    }
    (best_idx, best_key.0)
}

/// Working state for the greedy forward pass (Python `_Phase1State`).
struct Phase1State {
    chains: Vec<AgentChain>,
    chain_of_request: HashMap<i64, usize>,
    forks_by_tail: HashMap<i64, Vec<usize>>,
    req_by_outer: HashMap<i64, ChainReq>,
    unclassified: i64,
}

impl Phase1State {
    fn new() -> Self {
        Self {
            chains: Vec::new(),
            chain_of_request: HashMap::new(),
            forks_by_tail: HashMap::new(),
            req_by_outer: HashMap::new(),
            unclassified: 0,
        }
    }

    fn append(&mut self, chain_idx: usize, outer_idx: i64, req: &ChainReq) {
        let c = &mut self.chains[chain_idx];
        c.requests.push((outer_idx, req.clone()));
        self.chain_of_request.insert(outer_idx, chain_idx);
        if !req.hash_ids.is_empty() {
            c.tail_outer_idx = outer_idx;
            c.tail_hash = req.hash_ids.clone();
            c.tail_end = req_end(req);
            c.tail_model = req.model.clone();
        }
    }

    fn classify(&mut self, outer_idx: i64, req: &ChainReq) {
        self.req_by_outer.insert(outer_idx, req.clone());
        if req.hash_ids.is_empty() {
            // No LCP evidence: keep on the main chain, invisible to tails/forks.
            self.unclassified += 1;
            if self.chains.is_empty() {
                self.chains.push(AgentChain::empty());
            }
            self.chains[0].requests.push((outer_idx, req.clone()));
            self.chain_of_request.insert(outer_idx, 0);
            return;
        }

        let h = req.hash_ids.clone();
        if self.chains.is_empty() {
            self.chains.push(AgentChain::empty());
            self.append(0, outer_idx, req);
            return;
        }
        if let Some(target) = find_extension_target(&self.chains, &h, req.t, &req.model) {
            self.append(target, outer_idx, req);
            return;
        }

        let (parent, depth) = max_lcp_chain(&self.chains, &h);
        if parent.is_none() && self.chains.iter().all(|c| c.tail_hash.is_empty()) {
            // First hash-bearing request while only leading empty-hash rows
            // exist: it IS the main agent — join chain 0.
            self.append(0, outer_idx, req);
            return;
        }
        let fork_outer_idx = parent.map(|p| self.chains[p].tail_outer_idx);
        let fork = ChainFork {
            parent_chain: parent,
            fork_outer_idx,
            depth,
            fork_time: req.t,
        };
        let new_idx = self.chains.len();
        self.chains.push(AgentChain::with_fork(fork));
        self.append(new_idx, outer_idx, req);
        if let Some(foi) = fork_outer_idx {
            if depth > 0 {
                self.forks_by_tail.entry(foi).or_default().push(new_idx);
            }
        }
    }
}

/// Pick the seam continuation among forks registered on a dead tail. Eligibility:
/// positive depth, no temporal overlap with the tail, same model, not seam-blocked.
/// Election: deepest LCP, tie-break earliest `fork_time`, then lowest chain index.
#[allow(clippy::too_many_arguments)]
fn elect_continuation(
    chains: &[AgentChain],
    registered: &[usize],
    t_req: &ChainReq,
    max_gap_seconds: f64,
    min_overlap_ratio: f64,
) -> Option<usize> {
    let t_end = req_end(t_req);
    let tail_blocks = t_req.hash_ids.len() as i64;

    let seam_blocked = |ci: usize| -> bool {
        if tail_blocks == 0 {
            return false;
        }
        let fork = chains[ci].fork.as_ref().unwrap();
        let gap = chains[ci].requests[0].1.t - t_end;
        let overlap = fork.depth as f64 / tail_blocks as f64;
        gap > max_gap_seconds && overlap < min_overlap_ratio
    };

    let mut candidates: Vec<usize> = Vec::new();
    for &ci in registered {
        let c = &chains[ci];
        let fork = match &c.fork {
            Some(f) => f,
            None => continue,
        };
        if fork.depth > 0
            && t_end <= c.requests[0].1.t + EPSILON_SECONDS
            && c.requests[0].1.model == t_req.model
            && !seam_blocked(ci)
        {
            candidates.push(ci);
        }
    }
    if candidates.is_empty() {
        return None;
    }
    // max by (depth, -fork_time, -ci): deepest, earliest fork, lowest index.
    candidates.into_iter().max_by(|&a, &b| {
        let fa = chains[a].fork.as_ref().unwrap();
        let fb = chains[b].fork.as_ref().unwrap();
        (fa.depth, -fa.fork_time, -(a as i64))
            .partial_cmp(&(fb.depth, -fb.fork_time, -(b as i64)))
            .unwrap()
    })
}

/// Re-evaluate non-elected forks against the merged chain's new tail. Returns
/// true when any fork was re-keyed.
#[allow(clippy::too_many_arguments)]
fn rekey_leftover_forks(
    chains: &mut [AgentChain],
    forks_by_tail: &mut HashMap<i64, Vec<usize>>,
    req_by_outer: &HashMap<i64, ChainReq>,
    registered: &[usize],
    elected: usize,
    owner: usize,
    new_tail_outer: i64,
) -> bool {
    let new_tail_hash = req_by_outer[&new_tail_outer].hash_ids.clone();
    let mut rekeyed = false;
    for &ci in registered {
        if ci == elected || chains[ci].spliced_into.is_some() || chains[ci].fork.is_none() {
            continue;
        }
        let first_hash = chains[ci].requests[0].1.hash_ids.clone();
        let d = np_lcp(&new_tail_hash, &first_hash);
        if d <= 0 {
            continue;
        }
        let fork = chains[ci].fork.as_mut().unwrap();
        fork.fork_outer_idx = Some(new_tail_outer);
        fork.depth = d;
        fork.parent_chain = Some(owner);
        forks_by_tail.entry(new_tail_outer).or_default().push(ci);
        rekeyed = true;
    }
    rekeyed
}

/// Phase 2: splice join-seam continuations onto dead tails.
fn resolve_seams(
    chains: &mut Vec<AgentChain>,
    forks_by_tail: &mut HashMap<i64, Vec<usize>>,
    chain_of_request: &mut HashMap<i64, usize>,
    req_by_outer: &HashMap<i64, ChainReq>,
    max_gap_seconds: f64,
    min_overlap_ratio: f64,
) -> i64 {
    let mut alias: HashMap<usize, usize> = HashMap::new();
    fn resolve(alias: &HashMap<usize, usize>, mut i: usize) -> usize {
        while let Some(&next) = alias.get(&i) {
            i = next;
        }
        i
    }

    let mut seams = 0i64;
    // Min-heap of tail keys via a sorted, de-duplicated worklist.
    let mut keys: std::collections::BinaryHeap<std::cmp::Reverse<i64>> =
        forks_by_tail.keys().map(|&k| std::cmp::Reverse(k)).collect();
    let mut processed: std::collections::HashSet<i64> = std::collections::HashSet::new();

    while let Some(std::cmp::Reverse(fork_outer_idx)) = keys.pop() {
        if processed.contains(&fork_outer_idx) {
            continue;
        }
        processed.insert(fork_outer_idx);
        let owner = resolve(&alias, chain_of_request[&fork_outer_idx]);
        if last_hash_outer_idx(&chains[owner]) != Some(fork_outer_idx) {
            continue; // longer state was extended -> all forks are spawns
        }
        let registered: Vec<usize> = forks_by_tail[&fork_outer_idx]
            .iter()
            .copied()
            .filter(|&ci| chains[ci].spliced_into.is_none())
            .collect();
        let elected = elect_continuation(
            chains,
            &registered,
            &req_by_outer[&fork_outer_idx],
            max_gap_seconds,
            min_overlap_ratio,
        );
        let elected = match elected {
            Some(e) => e,
            None => continue,
        };
        let target_requests = chains[elected].requests.clone();
        for (oi, _) in &target_requests {
            chain_of_request.insert(*oi, owner);
        }
        chains[owner].requests.extend(target_requests);
        chains[elected].spliced_into = Some(owner);
        alias.insert(elected, owner);
        seams += 1;

        let new_tail_outer = match last_hash_outer_idx(&chains[owner]) {
            Some(x) => x,
            None => continue,
        };
        if rekey_leftover_forks(
            chains,
            forks_by_tail,
            req_by_outer,
            &registered,
            elected,
            owner,
            new_tail_outer,
        ) {
            processed.remove(&new_tail_outer);
            keys.push(std::cmp::Reverse(new_tail_outer));
        }
    }
    seams
}

/// Partition retained top-level requests into per-agent chains (spec §4).
/// `normals` is `(outer_idx, request)` for each hash-eligible top-level request.
pub fn detect_agent_chains(
    normals: Vec<(i64, ChainReq)>,
    seam_max_gap_seconds: f64,
    seam_min_overlap_ratio: f64,
) -> ChainDetectionResult {
    if normals.is_empty() {
        return ChainDetectionResult {
            chains: Vec::new(),
            main_index: 0,
            worker_indices: Vec::new(),
            seams_merged: 0,
            unclassified_empty_hash: 0,
        };
    }

    // Process in (t, outer_idx) order (stable).
    let mut ordered = normals;
    ordered.sort_by(|a, b| {
        (a.1.t, a.0)
            .partial_cmp(&(b.1.t, b.0))
            .expect("request timestamps must be finite")
    });
    let first_outer = ordered[0].0;

    let mut state = Phase1State::new();
    for (outer_idx, req) in &ordered {
        state.classify(*outer_idx, req);
    }

    let seams = resolve_seams(
        &mut state.chains,
        &mut state.forks_by_tail,
        &mut state.chain_of_request,
        &state.req_by_outer,
        seam_max_gap_seconds,
        seam_min_overlap_ratio,
    );

    // Rewrite surviving forks' parent_chain through the splice alias.
    let alias: HashMap<usize, usize> = state
        .chains
        .iter()
        .enumerate()
        .filter_map(|(i, c)| c.spliced_into.map(|s| (i, s)))
        .collect();
    let resolve = |mut i: usize| {
        while let Some(&n) = alias.get(&i) {
            i = n;
        }
        i
    };
    for c in &mut state.chains {
        if c.spliced_into.is_none() {
            if let Some(fork) = &mut c.fork {
                if let Some(p) = fork.parent_chain {
                    fork.parent_chain = Some(resolve(p));
                }
            }
        }
    }

    let main_index = resolve(state.chain_of_request[&first_outer]);
    let mut workers: Vec<usize> = state
        .chains
        .iter()
        .enumerate()
        .filter(|(i, c)| c.spliced_into.is_none() && *i != main_index)
        .map(|(i, _)| i)
        .collect();
    workers.sort_by(|&a, &b| {
        let ra = &state.chains[a].requests[0];
        let rb = &state.chains[b].requests[0];
        (ra.1.t, ra.0).partial_cmp(&(rb.1.t, rb.0)).unwrap()
    });

    ChainDetectionResult {
        chains: state.chains,
        main_index,
        worker_indices: workers,
        seams_merged: seams,
        unclassified_empty_hash: state.unclassified,
    }
}

/// Classify a detected worker chain as an auxiliary one-shot (Python
/// `is_aux_chain`).
#[allow(clippy::too_many_arguments)]
pub fn is_aux_chain(
    requests: &[ChainReq],
    main_peak_isl: i64,
    max_requests: usize,
    isl_ratio: f64,
    isl_floor: i64,
    main_model: Option<&str>,
    cross_model: bool,
) -> bool {
    if requests.is_empty() || requests.len() > max_requests {
        return false;
    }
    let first = &requests[0];
    if cross_model {
        if let Some(mm) = main_model {
            if first.model != mm {
                return true;
            }
        }
    }
    let threshold = isl_floor.max((isl_ratio * main_peak_isl as f64) as i64);
    first.input_length < threshold
}

/// Classify a single-request worker chain as an auxiliary *reduction* call
/// (Python `is_reduction_chain`).
pub fn is_reduction_chain(requests: &[ChainReq], osl_max: i64, ratio: f64, isl_floor: i64) -> bool {
    if osl_max <= 0 || requests.len() != 1 {
        return false;
    }
    let first = &requests[0];
    let osl = first.output_length.max(0);
    if osl <= 0 || osl >= osl_max {
        return false;
    }
    if first.input_length < isl_floor {
        return false;
    }
    first.input_length as f64 > ratio * osl as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(t: f64, model: &str, hash_ids: &[i64]) -> ChainReq {
        ChainReq {
            t,
            api_time: Some(0.1),
            model: model.to_string(),
            hash_ids: hash_ids.to_vec(),
            input_length: hash_ids.len() as i64 * 4,
            output_length: 4,
        }
    }

    #[test]
    fn single_chain_extension() {
        // Each request is a full prefix of the next, same model -> one chain.
        let normals = vec![
            (0, req(0.0, "m", &[1])),
            (1, req(1.0, "m", &[1, 2])),
            (2, req(2.0, "m", &[1, 2, 3])),
        ];
        let r = detect_agent_chains(normals, 3600.0, 0.5);
        assert_eq!(r.worker_indices.len(), 0);
        assert_eq!(r.chains[r.main_index].requests.len(), 3);
    }

    #[test]
    fn divergence_without_pullback_is_a_seam() {
        // r1 shares only prefix [1] then diverges; nothing later pulls back to
        // the longer pre-shrink state, so it elects as a join seam (one chain).
        let normals = vec![(0, req(0.0, "m", &[1, 2])), (1, req(1.0, "m", &[1, 9]))];
        let r = detect_agent_chains(normals, 3600.0, 0.5);
        assert_eq!(r.worker_indices.len(), 0);
        assert_eq!(r.seams_merged, 1);
    }

    #[test]
    fn divergence_with_future_pullback_spawns_worker() {
        // r1 diverges from [1,2]; r2 later EXTENDS the longer state [1,2,3],
        // proving r1 was a separate spawned agent, not a continuation.
        let normals = vec![
            (0, req(0.0, "m", &[1, 2])),
            (1, req(1.0, "m", &[1, 9])),
            (2, req(2.0, "m", &[1, 2, 3])),
        ];
        let r = detect_agent_chains(normals, 3600.0, 0.5);
        assert_eq!(r.worker_indices.len(), 1);
        assert_eq!(r.seams_merged, 0);
    }

    #[test]
    fn empty_hash_rows_stay_on_main() {
        let normals = vec![
            (0, req(0.0, "m", &[1])),
            (1, req(1.0, "m", &[])),
            (2, req(2.0, "m", &[1, 2])),
        ];
        let r = detect_agent_chains(normals, 3600.0, 0.5);
        assert_eq!(r.unclassified_empty_hash, 1);
    }

    #[test]
    fn aux_and_reduction_classifiers() {
        let small = vec![req(0.0, "haiku", &[1])];
        assert!(is_aux_chain(&small, 1000, 3, 0.1, 50, Some("opus"), true));
        let reduce = vec![ChainReq {
            t: 0.0,
            api_time: None,
            model: "m".into(),
            hash_ids: vec![1, 2, 3],
            input_length: 1000,
            output_length: 5,
        }];
        assert!(is_reduction_chain(&reduce, 50, 10.0, 100));
    }
}
