// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Eager per-trace branch resolution and taken-subgraph pruning.
//!
//! Every conditional edge is resolved to one branch key from pre-execution data
//! only — a pinned `selected_branches` entry, a per-trace distribution, or the
//! edge's static `branch_weights` (sampled by a deterministic RNG seeded on
//! `(workload_seed, trace.id, source)`). A conditional edge that offers none of
//! these is a branch-on-live-output request and is rejected, not accepted. The
//! taken targets and every static edge form the active edge set; a BFS from
//! `START` yields the fired node set, and everything unreached is pruned. The
//! result is a flat topology of surviving nodes and static edges, ready for the
//! replay fold in [`super::fold`].

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::graph::model::{END_NODE_ID, START_NODE_ID};
use crate::rng::compat::numpy_pcg64::NumpyPcg64;
use crate::rng::derive_seed_parts;

use super::model::{
    AuthoredConditionalEdge, AuthoredEdge, AuthoredGraph, AuthoredNode, AuthoredTrace,
    ConditionalError,
};

/// A resolved, pruned topology: surviving nodes plus flattened static edges.
#[derive(Debug, Clone, PartialEq)]
pub struct TakenGraph {
    /// Surviving nodes by id (START/END are not nodes).
    pub nodes: BTreeMap<String, AuthoredNode>,
    /// Active edges whose source and target both survive (START/END allowed).
    pub edges: Vec<TakenEdge>,
}

/// One active edge after conditional resolution; a conditional edge contributes
/// its taken branch as a static edge carrying the conditional edge's delays.
#[derive(Debug, Clone, PartialEq)]
pub struct TakenEdge {
    pub source: String,
    pub target: String,
    pub delay_after_predecessor_us: Option<f64>,
    pub min_start_delay_us: Option<f64>,
    pub delay_after_predecessor_start_us: Option<f64>,
    pub delay_after_predecessor_first_token_us: Option<f64>,
}

/// Resolve one conditional edge's branch key from pre-execution data.
///
/// Precedence: pinned `selected_branches[source]`, then the per-trace
/// distribution `branch_distributions[source]`, then the edge's static
/// `branch_weights`. A key that resolves to nothing (no pin, no weights) is a
/// live-output branch and is rejected.
pub fn resolve_branch_key(
    edge: &AuthoredConditionalEdge,
    trace: &AuthoredTrace,
    workload_seed: u64,
) -> Result<String, ConditionalError> {
    let key = if let Some(pinned) = trace.selected_branches.get(&edge.source) {
        pinned.clone()
    } else if let Some(distribution) = trace
        .branch_distributions
        .as_ref()
        .and_then(|distributions| distributions.get(&edge.source))
    {
        sample_weighted(distribution, workload_seed, &trace.id, &edge.source)?
    } else if let Some(weights) = &edge.branch_weights {
        sample_weighted(weights, workload_seed, &trace.id, &edge.source)?
    } else {
        return Err(ConditionalError(format!(
            "conditional edge from {:?} has no selected_branches pin, per-trace \
distribution, or branch_weights; branching on a live output is not supported",
            edge.source
        )));
    };
    if !edge.branches.contains_key(&key) {
        return Err(ConditionalError(format!(
            "branch key {key:?} for conditional edge from {:?} is not one of {:?}",
            edge.source,
            edge.branches.keys().collect::<Vec<_>>()
        )));
    }
    Ok(key)
}

/// Deterministically pick a key by weight, seeded on `(seed, trace_id, source)`.
///
/// `BTreeMap` iteration is key-sorted, so the cumulative walk is order-stable and
/// the choice is reproducible for identical inputs.
fn sample_weighted(
    weights: &BTreeMap<String, f64>,
    workload_seed: u64,
    trace_id: &str,
    source: &str,
) -> Result<String, ConditionalError> {
    let total: f64 = weights
        .values()
        .copied()
        .filter(|weight| *weight > 0.0)
        .sum();
    // Reject an all-non-positive or NaN total via positive-form logic (a bare
    // `!(total > 0.0)` trips `clippy::neg_cmp_op_on_partial_ord`).
    let usable_total = total.is_finite() && total > 0.0;
    if !usable_total {
        return Err(ConditionalError(format!(
            "branch weights for source {source:?} are all non-positive"
        )));
    }
    let seed = derive_seed_parts(&[
        &workload_seed.to_le_bytes(),
        trace_id.as_bytes(),
        source.as_bytes(),
    ]);
    let mut rng = NumpyPcg64::from_u64_seed(seed);
    let point = rng.next_double() * total;
    let mut cumulative = 0.0;
    for (key, weight) in weights {
        if *weight <= 0.0 {
            continue;
        }
        cumulative += *weight;
        if point < cumulative {
            return Ok(key.clone());
        }
    }
    // Floating-point tail: the point rounded to `total`; award the last positive
    // key deterministically rather than failing.
    Ok(weights
        .iter()
        .rfind(|(_, weight)| **weight > 0.0)
        .map(|(key, _)| key.clone())
        .expect("total > 0 guarantees a positive-weight key"))
}

/// Resolve every conditional edge for `trace`, then prune to the taken subgraph.
pub fn resolve_and_prune(
    graph: &AuthoredGraph,
    trace: &AuthoredTrace,
    workload_seed: u64,
) -> Result<TakenGraph, ConditionalError> {
    // Flatten every edge to its active transitions: static edges pass through;
    // conditional edges resolve to the taken branch's targets.
    let mut active: Vec<TakenEdge> = Vec::new();
    for edge in &graph.edges {
        match edge {
            AuthoredEdge::Static(static_edge) => active.push(TakenEdge {
                source: static_edge.source.clone(),
                target: static_edge.target.clone(),
                delay_after_predecessor_us: static_edge.delay_after_predecessor_us,
                min_start_delay_us: static_edge.min_start_delay_us,
                delay_after_predecessor_start_us: static_edge.delay_after_predecessor_start_us,
                delay_after_predecessor_first_token_us: static_edge
                    .delay_after_predecessor_first_token_us,
            }),
            AuthoredEdge::Conditional(conditional) => {
                let key = resolve_branch_key(conditional, trace, workload_seed)?;
                for target in conditional.branches[&key].targets() {
                    active.push(TakenEdge {
                        source: conditional.source.clone(),
                        target: target.to_string(),
                        delay_after_predecessor_us: conditional.delay_after_predecessor_us,
                        min_start_delay_us: conditional.min_start_delay_us,
                        delay_after_predecessor_start_us: None,
                        delay_after_predecessor_first_token_us: None,
                    });
                }
            }
        }
    }

    // BFS from START over the active edges; collect reachable real nodes.
    let mut adjacency: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for edge in &active {
        adjacency
            .entry(edge.source.as_str())
            .or_default()
            .push(edge.target.as_str());
    }
    let mut fired: BTreeSet<String> = BTreeSet::new();
    let mut queue: VecDeque<&str> = VecDeque::new();
    queue.push_back(START_NODE_ID);
    let mut visited: BTreeSet<&str> = BTreeSet::new();
    visited.insert(START_NODE_ID);
    while let Some(node) = queue.pop_front() {
        for &target in adjacency.get(node).into_iter().flatten() {
            if target == END_NODE_ID {
                continue;
            }
            if !graph.nodes.contains_key(target) {
                return Err(ConditionalError(format!(
                    "edge targets undeclared node {target:?}"
                )));
            }
            fired.insert(target.to_string());
            if visited.insert(target) {
                queue.push_back(target);
            }
        }
    }

    // Keep nodes in the fired set and edges whose endpoints both survive.
    let nodes = graph
        .nodes
        .iter()
        .filter(|(id, _)| fired.contains(id.as_str()))
        .map(|(id, node)| (id.clone(), node.clone()))
        .collect();
    let survives = |id: &str| id == START_NODE_ID || id == END_NODE_ID || fired.contains(id);
    let edges = active
        .into_iter()
        .filter(|edge| survives(&edge.source) && survives(&edge.target))
        .collect();
    Ok(TakenGraph { nodes, edges })
}

#[cfg(test)]
mod tests {
    use super::super::model::parse_authored_graph;
    use super::*;

    // Diamond: START->route (branch shopping/non_shopping) and START->safety
    // (branch safe/unsafe). The shopping path chains route->plan->summarize.
    const DOC: &str = r#"
graph:
  state:
    intent: {type: text}
  nodes:
    route:     {node_type: llm, prompt: ["@intent"], output: intent}
    plan:      {node_type: llm, prompt: ["@intent"], output: plan}
    summarize: {node_type: llm, prompt: ["@plan"], output: summary}
    safety:    {node_type: llm, prompt: ["@intent"], output: safety}
    redirect:  {node_type: llm, prompt: ["@intent"], output: redirect}
  edges:
    - {source: START, target: route}
    - {source: START, target: safety}
    - {source: route, branches: {shopping: plan, non_shopping: END}}
    - {source: plan, target: summarize}
    - {source: safety, branches: {safe: END, unsafe: redirect}}
traces:
  - {id: t-shop, selected_branches: {route: shopping, safety: safe}}
  - {id: t-noshop, selected_branches: {route: non_shopping, safety: safe}}
  - {id: t-unsafe, selected_branches: {route: shopping, safety: unsafe}}
  - {id: t-weighted, branch_distributions: {route: {shopping: 1.0}, safety: {safe: 1.0}}}
"#;

    fn fired_ids(taken: &TakenGraph) -> BTreeSet<String> {
        taken.nodes.keys().cloned().collect()
    }

    #[test]
    fn pinned_branch_selects_subgraph() {
        let doc = parse_authored_graph(DOC.as_bytes()).unwrap();
        let taken = resolve_and_prune(&doc.graph, &doc.traces[0], 0).unwrap();
        assert_eq!(
            fired_ids(&taken),
            ["route", "plan", "summarize", "safety"]
                .into_iter()
                .map(String::from)
                .collect()
        );
    }

    #[test]
    fn other_branch_prunes_complement() {
        let doc = parse_authored_graph(DOC.as_bytes()).unwrap();
        let taken = resolve_and_prune(&doc.graph, &doc.traces[1], 0).unwrap();
        assert_eq!(
            fired_ids(&taken),
            ["route", "safety"].into_iter().map(String::from).collect()
        );
        // No edge should dangle into a pruned node.
        assert!(taken.edges.iter().all(|e| {
            (e.source == "START" || taken.nodes.contains_key(&e.source))
                && (e.target == "END" || taken.nodes.contains_key(&e.target))
        }));
    }

    #[test]
    fn diamond_retains_two_terminals() {
        let doc = parse_authored_graph(DOC.as_bytes()).unwrap();
        let taken = resolve_and_prune(&doc.graph, &doc.traces[2], 0).unwrap();
        assert_eq!(
            fired_ids(&taken),
            ["route", "plan", "summarize", "safety", "redirect"]
                .into_iter()
                .map(String::from)
                .collect()
        );
    }

    #[test]
    fn weighted_branch_is_deterministic() {
        let doc = parse_authored_graph(DOC.as_bytes()).unwrap();
        let taken_a = resolve_and_prune(&doc.graph, &doc.traces[3], 7).unwrap();
        let taken_b = resolve_and_prune(&doc.graph, &doc.traces[3], 7).unwrap();
        assert_eq!(taken_a, taken_b);
        // weight 1.0 on `shopping`/`safe` always fires the shopping path.
        assert!(taken_a.nodes.contains_key("plan"));
    }

    #[test]
    fn unknown_branch_key_errors() {
        let doc = parse_authored_graph(DOC.as_bytes()).unwrap();
        let mut trace = doc.traces[0].clone();
        trace
            .selected_branches
            .insert("route".to_string(), "bogus".to_string());
        let err = resolve_and_prune(&doc.graph, &trace, 0).unwrap_err();
        assert!(err.to_string().contains("bogus"));
    }

    #[test]
    fn unresolvable_branch_errors() {
        let doc = parse_authored_graph(DOC.as_bytes()).unwrap();
        // A trace that pins nothing and offers no weights cannot resolve `route`.
        let trace = AuthoredTrace {
            id: "t-live".to_string(),
            initial_state: BTreeMap::new(),
            selected_branches: BTreeMap::new(),
            branch_distributions: None,
            replay_outputs: BTreeMap::new(),
            arrival_time: None,
            tags: Vec::new(),
        };
        let err = resolve_and_prune(&doc.graph, &trace, 0).unwrap_err();
        assert!(err.to_string().contains("live output"));
    }
}
