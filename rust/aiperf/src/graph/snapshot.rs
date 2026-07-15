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

use crate::graph::model::{LlmNode, ParsedGraph, START_NODE_ID, StaticEdge};
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
}
