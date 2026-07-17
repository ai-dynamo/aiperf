// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! t* frontier chop for a segment-trie [`ParsedGraph`].
//!
//! Format-agnostic: operates purely on `ParsedGraph`
//! topology, [`StaticEdge`]s, per-node `arrival_offset_us`, and AND-fan-in
//! `inputs` — no recorded-trace (weka/dynamo) types. Any adapter that emits the
//! segment-trie IR (a graph of [`LlmNode`] + [`StaticEdge`] with
//! `metadata["arrival_offset_us"]`) can be snapshotted here.

use crate::graph::model::{GraphRecord, LlmNode, ParsedGraph, START_NODE_ID, StaticEdge};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

/// The node's recorded arrival offset in microseconds, or `0.0` when absent.
///
/// Missing, non-numeric, and zero `metadata["arrival_offset_us"]` values resolve
/// to `0.0`.
fn arrival_offset_us(node: &LlmNode) -> f64 {
    node.metadata
        .get("arrival_offset_us")
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(0.0)
}

/// Chop a segment-trie [`ParsedGraph`] to its live frontier at `t*`.
///
/// The segment-trie graph is the trivial [`LlmNode`] + [`StaticEdge`]
/// realization of one recorded trace; every node carries `arrival_offset_us`
/// (its recorded `t` * 1e6, stored in `metadata`). The snapshot chop:
///
/// * Drops every node whose `arrival_offset_us < t_star_us`.
/// * Re-roots each surviving node that lost all predecessors to the chop
///   from `START` via a synthetic [`StaticEdge`] whose `min_start_delay_us =
///   arrival_offset_us - t_star_us` — the node's absolute offset from the
///   instance run-origin `t*`. Inter-turn edges between two surviving nodes are
///   kept verbatim (whichever delay kind they carry is unchanged).
/// * Leaves each surviving node's prompt segment program unchanged: the full
///   pre-`t*` prefix stays in the path so the worker still materializes the
///   exact resume prompt.
///
/// `t_star_us <= 0` returns the graph unchanged (full `t*=0` replay).
///
/// Only `graph.graph` is chopped; named graphs and traces remain unchanged.
pub fn chop_trie_at_tstar(graph: &ParsedGraph, t_star_us: f64) -> ParsedGraph {
    if t_star_us <= 0.0 {
        return graph.clone();
    }

    let old_graph = &graph.graph;

    // BTreeMap iteration keeps survivor order deterministic.
    let survivor_ids: BTreeSet<&str> = old_graph
        .nodes
        .iter()
        .filter(|(_, node)| arrival_offset_us(node) >= t_star_us)
        .map(|(nid, _)| nid.as_str())
        .collect();

    let new_edges = chop_edges(&old_graph.edges, &survivor_ids, &old_graph.nodes, t_star_us);

    // A requirement on a dropped predecessor's `{src}_out` channel would deadlock
    // `await_inputs` (that channel is never written post-chop). Keep only
    // requirements whose source survives; a node re-rooted entirely from START
    // ends with empty `inputs`.
    let survivor_out_channels: BTreeSet<String> = survivor_ids
        .iter()
        .map(|nid| format!("{nid}_out"))
        .collect();

    let mut rescoped: BTreeMap<String, LlmNode> = BTreeMap::new();
    for nid in &survivor_ids {
        let node = &old_graph.nodes[*nid];
        rescoped.insert(
            (*nid).to_owned(),
            chop_node_inputs(node, &survivor_out_channels),
        );
    }

    let mut new_graph = old_graph.clone();
    new_graph.nodes = rescoped;
    new_graph.edges = new_edges;

    let mut out = graph.clone();
    out.graph = new_graph;
    out
}

/// Recompute the chopped graph's edge set against the surviving frontier.
///
/// An edge survives only when BOTH endpoints survive (or it roots an explicitly
/// kept node at `START`). Each surviving node that lost ALL its predecessors to
/// the chop is re-rooted from `START` at its `t*`-relative absolute offset,
/// dropping any kept `START` edge for it.
fn chop_edges(
    edges: &[StaticEdge],
    survivors: &BTreeSet<&str>,
    nodes: &BTreeMap<String, LlmNode>,
    t_star_us: f64,
) -> Vec<StaticEdge> {
    let mut kept_edges: Vec<StaticEdge> = Vec::new();
    let mut has_surviving_pred: BTreeSet<&str> = BTreeSet::new();
    for edge in edges {
        let (src, tgt) = (edge.source.as_str(), edge.target.as_str());
        if !survivors.contains(tgt) {
            continue;
        }
        if src == START_NODE_ID || survivors.contains(src) {
            kept_edges.push(edge.clone());
            if src != START_NODE_ID {
                // `tgt` is a survivor id (checked above); reborrow it from the
                // survivor set so the reference outlives this loop iteration.
                if let Some(id) = survivors.get(tgt) {
                    has_surviving_pred.insert(*id);
                }
            }
        }
    }

    let mut new_edges: Vec<StaticEdge> = kept_edges
        .into_iter()
        // Drop a kept START edge only when its target gained a re-root.
        .filter(|e| e.source != START_NODE_ID || has_surviving_pred.contains(e.target.as_str()))
        .collect();

    for nid in survivors {
        if !has_surviving_pred.contains(nid) {
            let arrival = arrival_offset_us(&nodes[*nid]);
            new_edges.push(StaticEdge {
                source: START_NODE_ID.to_owned(),
                target: (*nid).to_owned(),
                delay_after_predecessor_us: None,
                min_start_delay_us: Some(arrival - t_star_us),
                delay_after_predecessor_start_us: None,
                delay_after_predecessor_first_token_us: None,
            });
        }
    }
    new_edges
}

/// Drop a surviving node's `inputs` requirements on dropped predecessors.
///
/// An input-free node passes through untouched. An `inputs` list whose surviving
/// subset equals the original is returned unchanged.
fn chop_node_inputs(node: &LlmNode, survivor_out_channels: &BTreeSet<String>) -> LlmNode {
    if node.inputs.is_empty() {
        return node.clone();
    }
    let kept: Vec<_> = node
        .inputs
        .iter()
        .filter(|req| survivor_out_channels.contains(&req.channel))
        .cloned()
        .collect();
    if kept.len() == node.inputs.len() {
        return node.clone();
    }
    let mut out = node.clone();
    out.inputs = kept;
    out
}

/// Chop a segment-trie [`ParsedGraph`] to its extended-warmup handoff frontier.
///
/// The extended (cache-pressure) warmup replays the post-`t*` remainder with
/// zero idle delay; at drain, profiling must resume each chain at its first
/// NOT-yet-executed node rather than re-firing from `t*`. The chop:
///
/// * Drops every node with `arrival_offset_us < t_star_us` (pre-`t*` history,
///   primed by the boundary warmup) and every node in `executed`
///   (dispatched-and-returned during warmup/pressure — the server holds their
///   KV).
/// * Keeps inter-survivor edges verbatim (recorded pacing resumes past the
///   frontier).
/// * Re-roots each surviving node that lost all its real predecessors from
///   `START` with `min_start_delay_us` set to its RESIDUAL delay: for each
///   dropped predecessor edge, the residual is the recorded gap
///   `recorded_delay` minus the wall time `max(0, drain_end_wall_us
///   − return_wall_us[pred])` already spent waiting for the drain, floored at 0.
///   AND-fan-in takes the max across dropped predecessors, and the result is
///   clamped to `residual_cap_us` when set. The recorded base uses ONLY
///   end-anchored quantities (`delay_after_predecessor_us`, edge
///   `min_start_delay_us`): the ledger wall is the predecessor's RETURN, so a
///   dispatch-anchored delay (`delay_after_predecessor_start_us` / first-token)
///   debited from a return-anchored elapsed would over-delay by the pred's live
///   service time, so start-anchored edges contribute 0. A dropped
///   predecessor with no recorded wall contributes 0 (fire immediately).
/// * Rescopes surviving nodes' AND-fan-in `inputs` exactly like
///   [`chop_trie_at_tstar`]. A survivor that keeps a surviving-pred edge but lost
///   a residual-carrying binding edge to the chop is NOT re-rooted; instead that
///   dropped edge's residual is FOLDED into the node's `min_start_delay_us`
///   (max-combined with any existing node value). Under
///   `absolute_start_offsets=True` the executor anchors that node-level gate to
///   the instance run-start — the same anchor the re-root residuals use.
///
/// `return_wall_us` and `drain_end_wall_us` share one monotonic clock (the
/// warmup strategy's ledger); only differences are meaningful. Unlike
/// [`chop_trie_at_tstar`], this chop has no `t_star_us <= 0` early return: at
/// `t*<=0` every node still passes the arrival test, but `executed` nodes are
/// still dropped and their successors re-rooted at residuals.
pub fn chop_trie_at_frontier(
    graph: &ParsedGraph,
    t_star_us: f64,
    executed: &HashSet<String>,
    return_wall_us: &HashMap<String, f64>,
    drain_end_wall_us: f64,
    residual_cap_us: Option<f64>,
) -> ParsedGraph {
    let old_graph = &graph.graph;

    // Survivors: post-`t*` AND not-yet-executed, in stable node-id order.
    let survivor_ids: BTreeSet<&str> = old_graph
        .nodes
        .iter()
        .filter(|(nid, node)| {
            arrival_offset_us(node) >= t_star_us && !executed.contains(nid.as_str())
        })
        .map(|(nid, _)| nid.as_str())
        .collect();

    let (new_edges, kept_pred_residuals) = frontier_edges(
        &old_graph.edges,
        &survivor_ids,
        return_wall_us,
        drain_end_wall_us,
        residual_cap_us,
    );

    let survivor_out_channels: BTreeSet<String> = survivor_ids
        .iter()
        .map(|nid| format!("{nid}_out"))
        .collect();

    let mut rescoped: BTreeMap<String, LlmNode> = BTreeMap::new();
    for nid in &survivor_ids {
        let node = &old_graph.nodes[*nid];
        let mut node = chop_node_inputs(node, &survivor_out_channels);
        // A dropped binding edge's residual must not vanish just because a
        // zero-delay join edge from a surviving predecessor remains.
        // node-level gate, max-combined with any existing node value.
        if let Some(&residual) = kept_pred_residuals.get(*nid) {
            node.min_start_delay_us = Some(node.min_start_delay_us.unwrap_or(0.0).max(residual));
        }
        rescoped.insert((*nid).to_owned(), node);
    }

    let mut new_graph = old_graph.clone();
    new_graph.nodes = rescoped;
    new_graph.edges = new_edges;

    let mut out = graph.clone();
    out.graph = new_graph;
    out
}

/// Edge set for the handoff chop: keep inter-survivor, re-root at residuals.
///
/// Mirrors [`chop_edges`] structurally; the divergence is the re-root
/// offset — the t* chop rebases to the recorded absolute offset (`arrival - t*`)
/// because nothing was replayed yet, while the frontier chop uses the
/// residual-of-recorded-gap because pressure already consumed the recorded leads.
///
/// Returns `(edges, kept_pred_residuals)`. `kept_pred_residuals` maps a survivor
/// id to the leftover residual of a dropped binding edge whose target ALSO
/// retains a surviving-pred edge (so it is not re-rooted); the caller folds those
/// node-level so a dropped binding gap is not lost behind a zero-delay join edge.
fn frontier_edges(
    edges: &[StaticEdge],
    survivors: &BTreeSet<&str>,
    return_wall_us: &HashMap<String, f64>,
    drain_end_wall_us: f64,
    residual_cap_us: Option<f64>,
) -> (Vec<StaticEdge>, BTreeMap<String, f64>) {
    let mut kept_edges: Vec<StaticEdge> = Vec::new();
    let mut has_surviving_pred: BTreeSet<&str> = BTreeSet::new();
    let mut residual_by_target: BTreeMap<&str, f64> = BTreeMap::new();
    for edge in edges {
        let (src, tgt) = (edge.source.as_str(), edge.target.as_str());
        if !survivors.contains(tgt) {
            continue;
        }
        if src == START_NODE_ID || survivors.contains(src) {
            kept_edges.push(edge.clone());
            if src != START_NODE_ID
                && let Some(id) = survivors.get(tgt)
            {
                has_surviving_pred.insert(*id);
            }
            continue;
        }
        // `src` was dropped (executed / pre-t* history): fold its recorded gap,
        // minus the wall time already waited since its return, into the target's
        // re-root offset. END-anchored quantities only (see the doc comment); a
        // dropped pred with no recorded wall bursts (residual 0).
        let recorded_us = edge
            .delay_after_predecessor_us
            .unwrap_or(0.0)
            .max(edge.min_start_delay_us.unwrap_or(0.0));
        let mut residual = match return_wall_us.get(src) {
            Some(&wall) => (recorded_us - (drain_end_wall_us - wall).max(0.0)).max(0.0),
            None => 0.0,
        };
        if let Some(cap) = residual_cap_us {
            residual = residual.min(cap);
        }
        // `tgt` is a survivor id (checked above); reborrow it so the key
        // reference outlives this loop iteration.
        if let Some(id) = survivors.get(tgt) {
            let slot = residual_by_target.entry(*id).or_insert(0.0);
            *slot = slot.max(residual);
        }
    }

    let mut new_edges: Vec<StaticEdge> = kept_edges
        .into_iter()
        // Drop a kept START edge only when its target gained a re-root (lost all
        // surviving preds) — same De Morgan filter as [`chop_edges`].
        .filter(|e| e.source != START_NODE_ID || has_surviving_pred.contains(e.target.as_str()))
        .collect();

    let mut kept_pred_residuals: BTreeMap<String, f64> = BTreeMap::new();
    for nid in survivors {
        if !has_surviving_pred.contains(nid) {
            new_edges.push(StaticEdge {
                source: START_NODE_ID.to_owned(),
                target: (*nid).to_owned(),
                delay_after_predecessor_us: None,
                min_start_delay_us: Some(residual_by_target.get(nid).copied().unwrap_or(0.0)),
                delay_after_predecessor_start_us: None,
                delay_after_predecessor_first_token_us: None,
            });
        } else {
            let residual = residual_by_target.get(nid).copied().unwrap_or(0.0);
            if residual > 0.0 {
                kept_pred_residuals.insert((*nid).to_owned(), residual);
            }
        }
    }
    (new_edges, kept_pred_residuals)
}

/// The per-session chain key a trie node belongs to.
///
/// Recorded builders mint node ids with `:` as the ordinal separator
/// (weka `{scope}:{turn_index}` — `graph/recorded/weka/mod.rs:170`; aiperf_trace
/// `{chain}:{turn}`; dynamo `{session_id}:{k}`), so a `_`-based id split does not
/// recover the enclosing session chain and would make
/// every recorded node its own singleton. Instead group by the AUTHORITATIVE
/// chain identity the trie lowerer stamps into
/// `metadata["conversation_id"]` (= the weka scope / dynamo `session_id` /
/// aiperf_trace agent-or-session, written at `graph/recorded/trie/mod.rs:170`),
/// with `metadata["turn_index"]` as the ordinal.
///
/// A node lacking `metadata["conversation_id"]` (e.g. a synthetic/dag
/// graph that reaches this path) forms a defensive singleton keyed by its own
/// id.
fn chain_key(node_id: &str, node: &LlmNode) -> String {
    node.metadata
        .get("conversation_id")
        .and_then(serde_json::Value::as_str)
        .map_or_else(|| node_id.to_owned(), str::to_owned)
}

/// Return `{node_id: node}` of each chain-live-at-`t*` boundary turn.
///
/// Chains are the per-session linear paths keyed by `metadata["conversation_id"]`
/// ([`chain_key`]), ordered by recorded arrival. A chain is LIVE when it has
/// BOTH a node arriving before `t*` and a node arriving at/after `t*`; its
/// boundary is the LAST pre-`t*` node. Chains with no pre-`t*` node need no
/// priming (profiling replays them from their own start); chains entirely
/// pre-`t*` are not live (nothing of them is profiled). The returned map borrows
/// the boundary nodes from `graph` unchanged — the warmup re-root is applied by
/// [`rewrite_for_warmup`].
pub fn warmup_boundary_nodes(graph: &GraphRecord, t_star_us: f64) -> BTreeMap<String, &LlmNode> {
    let mut chains: BTreeMap<String, Vec<(f64, &str)>> = BTreeMap::new();
    for (nid, node) in &graph.nodes {
        chains
            .entry(chain_key(nid, node))
            .or_default()
            .push((arrival_offset_us(node), nid.as_str()));
    }
    let mut boundary: BTreeMap<String, &LlmNode> = BTreeMap::new();
    for members in chains.values_mut() {
        // Recorded offsets are finite, so this comparison is total.
        members.sort_by(|a, b| a.partial_cmp(b).expect("finite arrival offsets"));
        let last_pre = members
            .iter()
            .rfind(|(arrival, _)| *arrival < t_star_us)
            .map(|(_, nid)| *nid);
        let any_post = members.iter().any(|(arrival, _)| *arrival >= t_star_us);
        if let Some(nid) = last_pre
            && any_post
        {
            boundary.insert(nid.to_owned(), &graph.nodes[nid]);
        }
    }
    boundary
}

/// Rewrite `parsed` into the WARMUP boundary-priming graph at `t*`.
///
/// Warmup dispatches exactly one priming credit per chain live at `t*`: the
/// chain's boundary turn, the last node of that
/// per-session chain whose recorded arrival precedes `t*`
/// ([`warmup_boundary_nodes`]). Because trie prompts are cumulative along a
/// chain, priming the boundary turn's prompt (at the worker-side warmup
/// `max_tokens` cap, keyed off the `"warmup"` phase variant) warms the chain's
/// whole prefix.
///
/// The produced graph is flat: only the boundary nodes survive, each re-rooted
/// from `START` with no leading offset (warmup bursts every priming credit at
/// phase start rather than replaying recorded gaps — the synthetic edge carries
/// no delay of any kind) and with fan-in `inputs` cleared and `min_start_delay_us`
/// dropped (their predecessors are gone). Node identity, the trie envelope
/// (`metadata`, including the retained `arrival_offset_us`), `max_tokens`, and
/// the prompt program (`items`) are preserved so the worker resolves the
/// unmodified catalog ordinal and materializes the exact recorded prompt.
/// `t_star_us <= 0` (full replay or a zero-duration trace) yields an
/// empty graph so the warmup phase finalizes immediately. The worker applies the
/// warmup `max_tokens` cap downstream; the node remains unchanged.
pub fn rewrite_for_warmup(parsed: &ParsedGraph, t_star_us: f64) -> ParsedGraph {
    let boundary = if t_star_us > 0.0 {
        warmup_boundary_nodes(&parsed.graph, t_star_us)
    } else {
        BTreeMap::new()
    };

    let mut new_nodes: BTreeMap<String, LlmNode> = BTreeMap::new();
    for (nid, node) in &boundary {
        let mut rewritten = (*node).clone();
        rewritten.inputs = Vec::new();
        rewritten.min_start_delay_us = None;
        new_nodes.insert(nid.clone(), rewritten);
    }

    let new_edges: Vec<StaticEdge> = new_nodes
        .keys()
        .map(|nid| StaticEdge {
            source: START_NODE_ID.to_owned(),
            target: nid.clone(),
            delay_after_predecessor_us: None,
            min_start_delay_us: None,
            delay_after_predecessor_start_us: None,
            delay_after_predecessor_first_token_us: None,
        })
        .collect();

    let mut new_graph = parsed.graph.clone();
    new_graph.nodes = new_nodes;
    new_graph.edges = new_edges;

    let mut out = parsed.clone();
    out.graph = new_graph;
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::model::{ChannelRequirement, GraphRecord};
    use serde_json::json;

    fn node(arrival_us: u64, input_channels: &[&str]) -> LlmNode {
        let mut metadata = BTreeMap::new();
        metadata.insert("arrival_offset_us".to_owned(), json!(arrival_us));
        LlmNode {
            output: "unused_out".to_owned(),
            streaming: true,
            inputs: input_channels
                .iter()
                .map(|c| ChannelRequirement {
                    channel: (*c).to_owned(),
                    count: Default::default(),
                })
                .collect(),
            min_start_delay_us: None,
            max_tokens: None,
            items: Vec::new(),
            metadata,
        }
    }

    fn edge(
        source: &str,
        target: &str,
        delay_after: Option<f64>,
        min_start: Option<f64>,
    ) -> StaticEdge {
        StaticEdge {
            source: source.to_owned(),
            target: target.to_owned(),
            delay_after_predecessor_us: delay_after,
            min_start_delay_us: min_start,
            delay_after_predecessor_start_us: None,
            delay_after_predecessor_first_token_us: None,
        }
    }

    /// 3-node chain n0 -> n1 -> n2 at arrival offsets 0 / 1e6 / 2e6 us, edges
    /// START->n0, n0->n1, n1->n2, n2->END.
    fn fixture() -> ParsedGraph {
        let mut nodes = BTreeMap::new();
        nodes.insert("n0".to_owned(), node(0, &[]));
        nodes.insert("n1".to_owned(), node(1_000_000, &["n0_out"]));
        nodes.insert("n2".to_owned(), node(2_000_000, &["n1_out"]));
        let edges = vec![
            edge("START", "n0", None, Some(0.0)),
            edge("n0", "n1", Some(1_000_000.0), None),
            edge("n1", "n2", Some(1_000_000.0), None),
            edge("n2", "END", None, None),
        ];
        ParsedGraph {
            graph: GraphRecord {
                nodes,
                edges,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn edge_tuples(g: &ParsedGraph) -> Vec<(String, String, Option<f64>, Option<f64>)> {
        g.graph
            .edges
            .iter()
            .map(|e| {
                (
                    e.source.clone(),
                    e.target.clone(),
                    e.min_start_delay_us,
                    e.delay_after_predecessor_us,
                )
            })
            .collect()
    }

    #[test]
    fn chop_tstar_mid_chain_drops_n0_reroots_n1() {
        let chopped = chop_trie_at_tstar(&fixture(), 1_000_000.0);

        // Survivors: n1, n2 (n0 dropped: arrival 0 < t*).
        assert_eq!(
            chopped.graph.nodes.keys().cloned().collect::<Vec<_>>(),
            vec!["n1".to_owned(), "n2".to_owned()]
        );
        // n1 lost its only pred (n0) -> inputs rescoped to empty.
        assert!(chopped.graph.nodes["n1"].inputs.is_empty());
        // n2 keeps its surviving-pred requirement on n1_out.
        assert_eq!(chopped.graph.nodes["n2"].inputs.len(), 1);
        assert_eq!(chopped.graph.nodes["n2"].inputs[0].channel, "n1_out");
        // Node-level min_start_delay untouched by the t* chop.
        assert_eq!(chopped.graph.nodes["n1"].min_start_delay_us, None);

        // Edges: kept inter-survivor n1->n2 (verbatim), plus re-root START->n1
        // at arrival(1e6) - t*(1e6) = 0.0. n2 keeps its surviving pred so is NOT
        // re-rooted.
        assert_eq!(
            edge_tuples(&chopped),
            vec![
                ("n1".to_owned(), "n2".to_owned(), None, Some(1_000_000.0)),
                ("START".to_owned(), "n1".to_owned(), Some(0.0), None),
            ]
        );
    }

    #[test]
    fn chop_tstar_zero_returns_unchanged() {
        let chopped = chop_trie_at_tstar(&fixture(), 0.0);
        assert_eq!(
            chopped.graph.nodes.keys().cloned().collect::<Vec<_>>(),
            vec!["n0".to_owned(), "n1".to_owned(), "n2".to_owned()]
        );
        assert_eq!(edge_tuples(&chopped), edge_tuples(&fixture()));
    }

    #[test]
    fn chop_tstar_negative_returns_unchanged() {
        let chopped = chop_trie_at_tstar(&fixture(), -5.0);
        assert_eq!(chopped.graph.nodes.len(), 3);
        assert_eq!(edge_tuples(&chopped), edge_tuples(&fixture()));
    }

    #[test]
    fn chop_tstar_all_dropped_yields_empty() {
        let chopped = chop_trie_at_tstar(&fixture(), 5_000_000.0);
        assert!(chopped.graph.nodes.is_empty());
        assert!(chopped.graph.edges.is_empty());
    }

    /// Node with `metadata["conversation_id"] = chain`, arrival, inputs, optional
    /// min-start-delay, and optional `max_tokens`, so warmup preservation/clearing
    /// is observable. Chain identity comes from the conversation_id metadata (as
    /// the recorded trie lowerer writes it), NOT the node id's `_`-suffix.
    fn wnode(
        chain: &str,
        arrival_us: u64,
        inputs: &[&str],
        min_start: Option<f64>,
        max_tokens: Option<usize>,
    ) -> LlmNode {
        let mut n = node(arrival_us, inputs);
        n.min_start_delay_us = min_start;
        n.max_tokens = max_tokens;
        n.metadata
            .insert("conversation_id".to_owned(), json!(chain));
        n
    }

    /// Four chains at `t* = 1.5e6`: `chainA` straddles t* (boundary = chainA_1),
    /// `chainB` entirely pre-t* (not live), `chainC` entirely post-t* (no prime),
    /// `chainD` straddles with a node-level min-start-delay + max_tokens on its
    /// boundary (boundary = chainD_1).
    fn warmup_fixture() -> ParsedGraph {
        let mut nodes = BTreeMap::new();
        nodes.insert("chainA_0".to_owned(), wnode("chainA", 0, &[], None, None));
        nodes.insert(
            "chainA_1".to_owned(),
            wnode("chainA", 1_000_000, &["chainA_0_out"], None, None),
        );
        nodes.insert(
            "chainA_2".to_owned(),
            wnode("chainA", 2_000_000, &["chainA_1_out"], None, None),
        );
        nodes.insert("chainB_0".to_owned(), wnode("chainB", 0, &[], None, None));
        nodes.insert(
            "chainB_1".to_owned(),
            wnode("chainB", 500_000, &["chainB_0_out"], None, None),
        );
        nodes.insert(
            "chainC_0".to_owned(),
            wnode("chainC", 3_000_000, &[], None, None),
        );
        nodes.insert("chainD_0".to_owned(), wnode("chainD", 0, &[], None, None));
        nodes.insert(
            "chainD_1".to_owned(),
            wnode(
                "chainD",
                1_000_000,
                &["chainD_0_out"],
                Some(999.0),
                Some(42),
            ),
        );
        nodes.insert(
            "chainD_2".to_owned(),
            wnode("chainD", 2_000_000, &["chainD_1_out"], None, None),
        );
        ParsedGraph {
            graph: GraphRecord {
                nodes,
                edges: Vec::new(),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    const TSTAR: f64 = 1_500_000.0;

    #[test]
    fn warmup_boundary_is_last_pre_tstar_of_each_live_chain() {
        let g = warmup_fixture();
        let boundary = warmup_boundary_nodes(&g.graph, TSTAR);
        // chainA_1 and chainD_1 straddle t*; chainB (all pre) and chainC (all
        // post) are not live.
        assert_eq!(
            boundary.keys().cloned().collect::<Vec<_>>(),
            vec!["chainA_1".to_owned(), "chainD_1".to_owned()]
        );
    }

    #[test]
    fn rewrite_for_warmup_flattens_and_reroots_from_start() {
        let g = warmup_fixture();
        let warmup = rewrite_for_warmup(&g, TSTAR);

        assert_eq!(
            warmup.graph.nodes.keys().cloned().collect::<Vec<_>>(),
            vec!["chainA_1".to_owned(), "chainD_1".to_owned()]
        );

        // Boundary nodes: inputs cleared, node-level min_start_delay dropped, but
        // max_tokens + trie envelope (arrival metadata) preserved.
        let d1 = &warmup.graph.nodes["chainD_1"];
        assert!(d1.inputs.is_empty());
        assert_eq!(d1.min_start_delay_us, None);
        assert_eq!(d1.max_tokens, Some(42));
        assert_eq!(arrival_offset_us(d1), 1_000_000.0);
        assert!(warmup.graph.nodes["chainA_1"].inputs.is_empty());

        // One priming edge per boundary node, rooted at START with NO delay of
        // any kind (bursts at phase start).
        let edges: Vec<_> = warmup
            .graph
            .edges
            .iter()
            .map(|e| {
                (
                    e.source.clone(),
                    e.target.clone(),
                    e.delay_after_predecessor_us,
                    e.min_start_delay_us,
                    e.delay_after_predecessor_start_us,
                    e.delay_after_predecessor_first_token_us,
                )
            })
            .collect();
        assert_eq!(
            edges,
            vec![
                (
                    "START".to_owned(),
                    "chainA_1".to_owned(),
                    None,
                    None,
                    None,
                    None
                ),
                (
                    "START".to_owned(),
                    "chainD_1".to_owned(),
                    None,
                    None,
                    None,
                    None
                ),
            ]
        );
    }

    /// A recorded-id node: id uses the Rust `:` ordinal separator and carries
    /// the authoritative chain identity in `metadata["conversation_id"]` (as the
    /// recorded trie lowerer writes it, `graph/recorded/trie/mod.rs:170`).
    fn rnode(conversation_id: &str, arrival_us: u64) -> LlmNode {
        let mut n = node(arrival_us, &[]);
        n.metadata
            .insert("conversation_id".to_owned(), json!(conversation_id));
        n
    }

    /// One recorded session chain "root" with two `:`-ordinal-id turns straddling
    /// `t*`: `root:0` (arrival 0, pre-t*) and `root:1` (arrival 2e6, post-t*).
    /// Both carry `metadata["conversation_id"] = "root"`. The chain is LIVE, so
    /// its boundary must be the LAST pre-t* node, `root:0`.
    #[test]
    fn warmup_boundary_groups_colon_recorded_ids_by_conversation_id() {
        let mut nodes = BTreeMap::new();
        nodes.insert("root:0".to_owned(), rnode("root", 0));
        nodes.insert("root:1".to_owned(), rnode("root", 2_000_000));
        let g = ParsedGraph {
            graph: GraphRecord {
                nodes,
                edges: Vec::new(),
                ..Default::default()
            },
            ..Default::default()
        };
        let boundary = warmup_boundary_nodes(&g.graph, 1_000_000.0);
        assert_eq!(
            boundary.keys().cloned().collect::<Vec<_>>(),
            vec!["root:0".to_owned()],
            "chain 'root' straddles t*; boundary is its last pre-t* node"
        );
    }

    /// A nested subagent chain: the subagent's conversation_id is the PARENT
    /// turn id (`root:1`), and its own turns are `root:1:0` / `root:1:1`. Grouping
    /// by `conversation_id` metadata (not id-suffix) is what makes this correct —
    /// an id-suffix split of `root:1:0` on any separator would misgroup it.
    #[test]
    fn warmup_boundary_groups_nested_subagent_chain_by_conversation_id() {
        let mut nodes = BTreeMap::new();
        // Root chain: single pre-t* node — not live on its own.
        nodes.insert("root:0".to_owned(), rnode("root", 0));
        // Subagent chain conversation_id="root:1", straddling t*.
        nodes.insert("root:1:0".to_owned(), rnode("root:1", 500_000));
        nodes.insert("root:1:1".to_owned(), rnode("root:1", 3_000_000));
        let g = ParsedGraph {
            graph: GraphRecord {
                nodes,
                edges: Vec::new(),
                ..Default::default()
            },
            ..Default::default()
        };
        let boundary = warmup_boundary_nodes(&g.graph, 1_000_000.0);
        assert_eq!(
            boundary.keys().cloned().collect::<Vec<_>>(),
            vec!["root:1:0".to_owned()],
            "subagent chain 'root:1' straddles t*; boundary is its last pre-t* node"
        );
    }

    /// Fallback: a node with NO `conversation_id` metadata (e.g. a synthetic/dag
    /// graph reaching this path) forms a defensive singleton keyed by its own id.
    /// Two such nodes never group, so a lone
    /// straddle across two DIFFERENT synthetic ids yields no boundary.
    #[test]
    fn warmup_boundary_falls_back_to_node_id_without_conversation_id() {
        let mut nodes = BTreeMap::new();
        nodes.insert("syn_0".to_owned(), node(0, &[]));
        nodes.insert("syn_1".to_owned(), node(2_000_000, &[]));
        let g = ParsedGraph {
            graph: GraphRecord {
                nodes,
                edges: Vec::new(),
                ..Default::default()
            },
            ..Default::default()
        };
        // Each is its own singleton chain (no conversation_id, distinct ids), so
        // neither is live -> empty boundary.
        let boundary = warmup_boundary_nodes(&g.graph, 1_000_000.0);
        assert!(boundary.is_empty());
    }

    #[test]
    fn rewrite_for_warmup_tstar_zero_yields_empty_graph() {
        let warmup = rewrite_for_warmup(&warmup_fixture(), 0.0);
        assert!(warmup.graph.nodes.is_empty());
        assert!(warmup.graph.edges.is_empty());
    }

    #[test]
    fn rewrite_for_warmup_tstar_negative_yields_empty_graph() {
        let warmup = rewrite_for_warmup(&warmup_fixture(), -5.0);
        assert!(warmup.graph.nodes.is_empty());
        assert!(warmup.graph.edges.is_empty());
    }

    /// Linear chain `n0 -> n1 -> n2 -> n3` at arrivals 0/1e6/2e6/3e6 us, edges
    /// START->n0, n0->n1, n1->n2, n2->n3, n3->END (each inter-node edge carries
    /// an end-anchored `delay_after_predecessor_us = 1e6`).
    fn frontier_fixture() -> ParsedGraph {
        let mut nodes = BTreeMap::new();
        nodes.insert("n0".to_owned(), node(0, &[]));
        nodes.insert("n1".to_owned(), node(1_000_000, &["n0_out"]));
        nodes.insert("n2".to_owned(), node(2_000_000, &["n1_out"]));
        nodes.insert("n3".to_owned(), node(3_000_000, &["n2_out"]));
        let edges = vec![
            edge("START", "n0", None, Some(0.0)),
            edge("n0", "n1", Some(1_000_000.0), None),
            edge("n1", "n2", Some(1_000_000.0), None),
            edge("n2", "n3", Some(1_000_000.0), None),
            edge("n3", "END", None, None),
        ];
        ParsedGraph {
            graph: GraphRecord {
                nodes,
                edges,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn set(ids: &[&str]) -> HashSet<String> {
        ids.iter().map(|s| (*s).to_owned()).collect()
    }

    fn walls(pairs: &[(&str, f64)]) -> HashMap<String, f64> {
        pairs.iter().map(|(k, v)| ((*k).to_owned(), *v)).collect()
    }

    #[test]
    fn frontier_drops_pre_tstar_and_executed_reroots_at_residual() {
        // t*=1e6 drops n0 (arrival 0 < t*); executed={n1} drops n1 mid-graph.
        // Survivors n2, n3. n2 lost pred n1 (dropped) -> re-root from START at
        // residual = max(0, recorded 1e6 - (drain 300 - return 100)) = 999800.
        let chopped = chop_trie_at_frontier(
            &frontier_fixture(),
            1_000_000.0,
            &set(&["n1"]),
            &walls(&[("n1", 100.0)]),
            300.0,
            None,
        );

        assert_eq!(
            chopped.graph.nodes.keys().cloned().collect::<Vec<_>>(),
            vec!["n2".to_owned(), "n3".to_owned()]
        );
        // n2 lost its only pred -> inputs rescoped to empty; n3 keeps n2_out.
        assert!(chopped.graph.nodes["n2"].inputs.is_empty());
        assert_eq!(chopped.graph.nodes["n3"].inputs.len(), 1);
        assert_eq!(chopped.graph.nodes["n3"].inputs[0].channel, "n2_out");
        // Kept inter-survivor n2->n3 verbatim; re-root START->n2 at residual.
        assert_eq!(
            edge_tuples(&chopped),
            vec![
                ("n2".to_owned(), "n3".to_owned(), None, Some(1_000_000.0)),
                ("START".to_owned(), "n2".to_owned(), Some(999_800.0), None),
            ]
        );
    }

    #[test]
    fn frontier_residual_cap_clamps_reroot_offset() {
        // Same as above but residual_cap_us=500 clamps 999800 -> 500.
        let chopped = chop_trie_at_frontier(
            &frontier_fixture(),
            1_000_000.0,
            &set(&["n1"]),
            &walls(&[("n1", 100.0)]),
            300.0,
            Some(500.0),
        );
        assert_eq!(
            edge_tuples(&chopped),
            vec![
                ("n2".to_owned(), "n3".to_owned(), None, Some(1_000_000.0)),
                ("START".to_owned(), "n2".to_owned(), Some(500.0), None),
            ]
        );
    }

    #[test]
    fn frontier_dropped_pred_without_wall_bursts_at_zero() {
        // No recorded wall for n1 -> residual 0 (burst).
        let chopped = chop_trie_at_frontier(
            &frontier_fixture(),
            1_000_000.0,
            &set(&["n1"]),
            &walls(&[]),
            300.0,
            None,
        );
        assert_eq!(
            edge_tuples(&chopped),
            vec![
                ("n2".to_owned(), "n3".to_owned(), None, Some(1_000_000.0)),
                ("START".to_owned(), "n2".to_owned(), Some(0.0), None),
            ]
        );
    }

    #[test]
    fn frontier_tstar_zero_still_drops_executed_no_early_return() {
        // Unlike chop_trie_at_tstar, t*<=0 does NOT short-circuit: executed
        // nodes are still dropped. executed={n0,n2} -> survivors n1, n3.
        // n1 lost pred n0 (executed) -> residual = max(0, 1e6 - (100-50)) = 999950.
        // n3 lost pred n2 (executed, no wall) -> residual 0.
        let chopped = chop_trie_at_frontier(
            &frontier_fixture(),
            0.0,
            &set(&["n0", "n2"]),
            &walls(&[("n0", 50.0)]),
            100.0,
            None,
        );
        assert_eq!(
            chopped.graph.nodes.keys().cloned().collect::<Vec<_>>(),
            vec!["n1".to_owned(), "n3".to_owned()]
        );
        assert!(chopped.graph.nodes["n1"].inputs.is_empty());
        assert!(chopped.graph.nodes["n3"].inputs.is_empty());
        assert_eq!(
            edge_tuples(&chopped),
            vec![
                ("START".to_owned(), "n1".to_owned(), Some(999_950.0), None),
                ("START".to_owned(), "n3".to_owned(), Some(0.0), None),
            ]
        );
    }

    /// AND-fan-in `j` depends on both `a1` (recorded chain pred) and `x` (a
    /// binding pred). Dropping `x` while `a1` survives exercises the
    /// fold-into-kept-survivor path: `j` is NOT re-rooted (it keeps a1->j) but
    /// x->j's residual is folded into `j.min_start_delay_us`.
    fn fold_fixture() -> ParsedGraph {
        let mut nodes = BTreeMap::new();
        nodes.insert("a0".to_owned(), node(0, &[]));
        nodes.insert("a1".to_owned(), node(1_000_000, &["a0_out"]));
        nodes.insert("x".to_owned(), node(1_000_000, &[]));
        nodes.insert("j".to_owned(), node(2_000_000, &["a1_out", "x_out"]));
        let edges = vec![
            edge("START", "a0", None, Some(0.0)),
            edge("a0", "a1", Some(1_000_000.0), None),
            edge("START", "x", None, Some(1_000_000.0)),
            edge("a1", "j", Some(500_000.0), None),
            edge("x", "j", Some(800_000.0), None),
        ];
        ParsedGraph {
            graph: GraphRecord {
                nodes,
                edges,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn frontier_folds_dropped_binding_residual_into_kept_survivor() {
        // t*=5e5 drops a0 (pre-t*); executed={x} drops the binding pred x.
        // Survivors a1, j. a1 lost pred a0 (pre-t*, no wall) -> re-root residual 0.
        // j keeps surviving pred a1 (edge a1->j) so is NOT re-rooted; the dropped
        // x->j residual = max(0, 800000 - (1000-400)) = 799400 folds into
        // j.min_start_delay_us. j's inputs rescope to just a1_out (x_out dropped).
        let chopped = chop_trie_at_frontier(
            &fold_fixture(),
            500_000.0,
            &set(&["x"]),
            &walls(&[("x", 400.0)]),
            1000.0,
            None,
        );
        assert_eq!(
            chopped.graph.nodes.keys().cloned().collect::<Vec<_>>(),
            vec!["a1".to_owned(), "j".to_owned()]
        );
        assert!(chopped.graph.nodes["a1"].inputs.is_empty());
        assert_eq!(chopped.graph.nodes["a1"].min_start_delay_us, None);
        // j: binding x_out dropped, a1_out kept; residual folded node-level.
        assert_eq!(chopped.graph.nodes["j"].inputs.len(), 1);
        assert_eq!(chopped.graph.nodes["j"].inputs[0].channel, "a1_out");
        assert_eq!(chopped.graph.nodes["j"].min_start_delay_us, Some(799_400.0));
        // Kept a1->j verbatim; re-root START->a1 at residual 0. No START->j.
        assert_eq!(
            edge_tuples(&chopped),
            vec![
                ("a1".to_owned(), "j".to_owned(), None, Some(500_000.0)),
                ("START".to_owned(), "a1".to_owned(), Some(0.0), None),
            ]
        );
    }
}
