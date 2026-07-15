// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! t* frontier chop for a segment-trie [`ParsedGraph`].
//!
//! Port of `src/aiperf/timing/snapshot_chop.py:21-114` (`chop_trie_at_tstar`
//! plus helpers `_chop_edges`, `_chop_node_inputs`) from branch
//! `ajc/aiperf-graph-ir`. Format-agnostic: operates purely on `ParsedGraph`
//! topology, [`StaticEdge`]s, per-node `arrival_offset_us`, and AND-fan-in
//! `inputs` — no recorded-trace (weka/dynamo) types. Any adapter that emits the
//! segment-trie IR (a graph of [`LlmNode`] + [`StaticEdge`] with
//! `metadata["arrival_offset_us"]`) can be snapshotted here.
//!
//! Signature adaptation vs the Python original: the Python model carries
//! `arrival_offset_us` as a first-class `LlmNode` field; the Rust IR carries it
//! in `LlmNode.metadata["arrival_offset_us"]` (a JSON `u64`, written by the
//! recorded-trie builder in `graph/recorded/trie/mod.rs`). `t_star_us` is taken
//! as `f64` (the brief's shape) rather than Python's `int`; the comparison and
//! the re-root offset arithmetic are otherwise identical.

use crate::graph::model::{GraphRecord, LlmNode, ParsedGraph, START_NODE_ID, StaticEdge};
use std::collections::{BTreeMap, BTreeSet};

/// The node's recorded arrival offset in microseconds, or `0.0` when absent.
///
/// Mirrors Python's `(node.arrival_offset_us or 0)`: a missing or non-numeric
/// `metadata["arrival_offset_us"]` (and the recorded value `0`) all resolve to
/// `0.0`.
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
/// * Drops every node whose `arrival_offset_us < t_star_us` — the pre-`t*`
///   turns were already WARMED (primed), not PROFILED.
/// * Re-roots each SURVIVING node that lost ALL its predecessors to the chop
///   from `START` via a synthetic [`StaticEdge`] whose `min_start_delay_us =
///   arrival_offset_us - t_star_us` — the node's ABSOLUTE offset from the
///   instance run-origin `t*`. Inter-turn edges between two SURVIVING nodes are
///   kept verbatim (whichever delay kind they carry is unchanged).
/// * Leaves each surviving node's prompt segment program UNCHANGED: the full
///   pre-`t*` prefix stays in the path so the worker still materializes the
///   EXACT resume prompt.
///
/// `t_star_us <= 0` returns the graph unchanged (full `t*=0` replay).
///
/// Port of `snapshot_chop.py:21-114`. Only `graph.graph` is chopped; the named
/// `graphs` map and `traces` are carried through verbatim, matching Python
/// (which rebuilds only `graph.graph`).
pub fn chop_trie_at_tstar(graph: &ParsedGraph, t_star_us: f64) -> ParsedGraph {
    if t_star_us <= 0.0 {
        return graph.clone();
    }

    let old_graph = &graph.graph;

    // Survivors, in the graph's node-id order (BTreeMap iteration is stable and
    // deterministic — the Rust analogue of the Python dict's insertion order for
    // an adapter that inserts nodes in id order).
    let survivor_ids: BTreeSet<&str> = old_graph
        .nodes
        .iter()
        .filter(|(_, node)| arrival_offset_us(node) >= t_star_us)
        .map(|(nid, _)| nid.as_str())
        .collect();

    let new_edges = chop_edges(&old_graph.edges, &survivor_ids, &old_graph.nodes, t_star_us);

    // Recompute each surviving node's AND-fan-in `inputs` against the chop: a
    // requirement on a DROPPED predecessor's `{src}_out` channel would deadlock
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
/// dropping any kept `START` edge for it. Port of `snapshot_chop.py:_chop_edges`.
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
        // De Morgan of Python's `not (source == START and target not in
        // has_surviving_pred)`: drop a kept START edge only when its target
        // gained a re-root (i.e. lost all surviving preds).
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
/// subset equals the original is returned unchanged (mirrors Python's
/// no-rebuild fast path). Port of `snapshot_chop.py:_chop_node_inputs`.
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

/// The per-session chain key a trie node id belongs to.
///
/// Both live trie producers mint node ids as `{chain_prefix}_{ordinal}` — one
/// linear chain per recorded session (the weka walk emits `{trace_id}:{k}` for
/// the root request list and `r_1_0`/`r_1_1` for a spawned subagent; the dynamo
/// lowering emits `{session_id}:{k}` per session). Stripping the final
/// `_`-delimited token recovers the enclosing session chain. Chain identity must
/// come from node ids because the sidecar-loaded timing plane strips
/// `metadata["trie"]`, leaving ids + edges as the only chain signal, and the
/// interval-order edges are cross-chain ordering edges, not session boundaries.
/// A node id with no `_` forms a defensive singleton chain.
///
/// Port of `graph_ir_replay.py:106-127` (`_chain_key`). Python uses
/// `str.rpartition("_")`, which splits on the LAST `_`; `str::rfind('_')` is the
/// byte-exact equivalent over the id's ASCII segment names.
fn chain_key(node_id: &str) -> String {
    match node_id.rfind('_') {
        Some(idx) => node_id[..idx].to_owned(),
        None => node_id.to_owned(),
    }
}

/// Return `{node_id: node}` of each chain-live-at-`t*` boundary turn.
///
/// Chains are the per-session linear paths the trie node ids encode
/// ([`chain_key`]), ordered by recorded arrival. A chain is LIVE when it has
/// BOTH a node arriving before `t*` and a node arriving at/after `t*`; its
/// boundary is the LAST pre-`t*` node. Chains with no pre-`t*` node need no
/// priming (profiling replays them from their own start); chains entirely
/// pre-`t*` are not live (nothing of them is profiled). The returned map borrows
/// the boundary nodes from `graph` unchanged — the warmup re-root is applied by
/// [`rewrite_for_warmup`], mirroring Python's returning of the same node objects.
///
/// Port of `graph_ir_replay.py:128-149` (`_warmup_boundary_nodes`). Signature
/// adaptation: the Python original takes a `GraphRecord` and reads a first-class
/// `LlmNode.arrival_offset_us`; the Rust IR reads it from
/// `metadata["arrival_offset_us"]` via [`arrival_offset_us`], and `t_star_us` is
/// `f64` (the brief's shape) rather than Python's `int`. Python sorts `(arrival,
/// nid)` tuples; the Rust sort is byte-identical (arrival then id).
pub fn warmup_boundary_nodes(graph: &GraphRecord, t_star_us: f64) -> BTreeMap<String, &LlmNode> {
    let mut chains: BTreeMap<String, Vec<(f64, &str)>> = BTreeMap::new();
    for (nid, node) in &graph.nodes {
        chains
            .entry(chain_key(nid))
            .or_default()
            .push((arrival_offset_us(node), nid.as_str()));
    }
    let mut boundary: BTreeMap<String, &LlmNode> = BTreeMap::new();
    for members in chains.values_mut() {
        // Python `list.sort()` on `(arrival, nid)` tuples; arrivals are never
        // NaN (they come from finite recorded offsets), so `partial_cmp` is total.
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
/// AgentX-parity contract: warmup dispatches exactly ONE priming credit per
/// chain LIVE at `t*` — the chain's boundary turn, the last node of that
/// per-session chain whose recorded arrival precedes `t*`
/// ([`warmup_boundary_nodes`]). Because trie prompts are cumulative along a
/// chain, priming the boundary turn's prompt (at the worker-side warmup
/// `max_tokens` cap, keyed off the `"warmup"` phase variant) warms the chain's
/// whole prefix.
///
/// The produced graph is FLAT: only the boundary nodes survive, each re-rooted
/// from `START` with NO leading offset (warmup bursts every priming credit at
/// phase start rather than replaying recorded gaps — the synthetic edge carries
/// no delay of any kind) and with fan-in `inputs` cleared and `min_start_delay_us`
/// dropped (their predecessors are gone). Node identity, the trie envelope
/// (`metadata`, including the retained `arrival_offset_us`), `max_tokens`, and
/// the prompt program (`items`) are preserved so the worker resolves the
/// unmodified catalog ordinal and materializes the exact recorded prompt.
/// `t_star_us <= 0` (full native replay, or a zero-duration trace) yields an
/// EMPTY graph so the warmup phase finalizes immediately.
///
/// Port of `graph_ir_replay.py:151-181` (`rewrite_for_warmup`). Signature
/// adaptation: `t_star_us` is `f64`; the worker-side warmup `max_tokens` cap is
/// applied downstream by the `"warmup"` phase variant and is NOT encoded into the
/// node here, matching the Python source (which likewise leaves `max_tokens`
/// untouched). Only `graph.graph` is rewritten; the named `graphs` map and
/// `traces` carry through verbatim, matching Python's `replace(parsed, graph=…)`.
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
    /// START->n0, n0->n1, n1->n2, n2->END. Mirrors the Python fixture in
    /// `/tmp/tstar/run.py`.
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

    // Expected values below were produced by running the VERBATIM Python chop
    // (`snapshot_chop.py:21-114`) against the equivalent fixture — see the
    // task-B1 report and `/tmp/tstar/run.py`.

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

    // --- warmup rewrite (task B2) ---------------------------------------------
    //
    // Expected values below were produced by running the VERBATIM Python warmup
    // rewrite (`graph_ir_replay.py:106-181`: `_chain_key`,
    // `_warmup_boundary_nodes`, `rewrite_for_warmup`) against a shape-equivalent
    // standalone fixture — see the task-B2 report and `/tmp/warmup_derive.py`.

    /// Node id `{chain}_{ordinal}` with arrival, inputs, optional min-start-delay,
    /// and optional `max_tokens`, so warmup preservation/clearing is observable.
    fn wnode(
        arrival_us: u64,
        inputs: &[&str],
        min_start: Option<f64>,
        max_tokens: Option<usize>,
    ) -> LlmNode {
        let mut n = node(arrival_us, inputs);
        n.min_start_delay_us = min_start;
        n.max_tokens = max_tokens;
        n
    }

    /// Four chains at `t* = 1.5e6`: `chainA` straddles t* (boundary = chainA_1),
    /// `chainB` entirely pre-t* (not live), `chainC` entirely post-t* (no prime),
    /// `chainD` straddles with a node-level min-start-delay + max_tokens on its
    /// boundary (boundary = chainD_1). Mirrors `/tmp/warmup_derive.py`.
    fn warmup_fixture() -> ParsedGraph {
        let mut nodes = BTreeMap::new();
        nodes.insert("chainA_0".to_owned(), wnode(0, &[], None, None));
        nodes.insert(
            "chainA_1".to_owned(),
            wnode(1_000_000, &["chainA_0_out"], None, None),
        );
        nodes.insert(
            "chainA_2".to_owned(),
            wnode(2_000_000, &["chainA_1_out"], None, None),
        );
        nodes.insert("chainB_0".to_owned(), wnode(0, &[], None, None));
        nodes.insert(
            "chainB_1".to_owned(),
            wnode(500_000, &["chainB_0_out"], None, None),
        );
        nodes.insert("chainC_0".to_owned(), wnode(3_000_000, &[], None, None));
        nodes.insert("chainD_0".to_owned(), wnode(0, &[], None, None));
        nodes.insert(
            "chainD_1".to_owned(),
            wnode(1_000_000, &["chainD_0_out"], Some(999.0), Some(42)),
        );
        nodes.insert(
            "chainD_2".to_owned(),
            wnode(2_000_000, &["chainD_1_out"], None, None),
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
}
