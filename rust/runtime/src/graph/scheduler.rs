// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Graph-derived adjacency helper for the async-dataflow executor.
//!
//! A pure adjacency view over the parsed graph's static edges; holds no per-trace
//! state (that lives on `TraceContext`), so it is shared across every trace.

use crate::graph::model::{END_NODE_ID, GraphRecord, START_NODE_ID, StaticEdge};
use std::collections::BTreeMap;

/// Why an anchored edge topology is unsupported and rejected at construction:
/// a START edge anchored on a dispatch or first-token event it cannot have, or a
/// start-anchored in-edge that is not its target's only in-edge (mixed-anchor /
/// multi-start-anchored fan-in).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AnchorFanInKind {
    /// START cannot provide a dispatch or first-token event.
    NonCompletionStart,
    /// A target has both start-anchored and completion-anchored inputs.
    Mixed,
    /// A target has more than one start-anchored input.
    MultipleStartAnchored,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct AnchorFanInFinding {
    kind: AnchorFanInKind,
    target: String,
    message: String,
}

impl AnchorFanInFinding {
    pub(crate) fn kind(&self) -> AnchorFanInKind {
        self.kind
    }

    pub(crate) fn target(&self) -> &str {
        &self.target
    }

    pub(crate) fn message(&self) -> &str {
        &self.message
    }
}

/// A scheduler construction error for an unsupported anchored fan-in topology.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MixedAnchorFanInError(pub String);

impl std::fmt::Display for MixedAnchorFanInError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for MixedAnchorFanInError {}

pub struct Scheduler {
    static_succ: BTreeMap<String, Vec<String>>,
    start_anchored_succ: BTreeMap<String, Vec<String>>,
    static_pred_edges: BTreeMap<String, Vec<StaticEdge>>,
    entry: Vec<String>,
}

impl Scheduler {
    pub fn new(graph: &GraphRecord) -> Result<Self, MixedAnchorFanInError> {
        let mut static_succ: BTreeMap<String, Vec<String>> = BTreeMap::new();
        let mut start_anchored_succ: BTreeMap<String, Vec<String>> = BTreeMap::new();
        let mut static_pred_edges: BTreeMap<String, Vec<StaticEdge>> = BTreeMap::new();

        for edge in &graph.edges {
            if let Some(finding) = non_completion_start_finding(edge) {
                return Err(MixedAnchorFanInError(finding.message().to_owned()));
            }
            if edge.delay_after_predecessor_start_us.is_some() {
                start_anchored_succ
                    .entry(edge.source.clone())
                    .or_default()
                    .push(edge.target.clone());
            } else {
                static_succ
                    .entry(edge.source.clone())
                    .or_default()
                    .push(edge.target.clone());
            }
            static_pred_edges
                .entry(edge.target.clone())
                .or_default()
                .push(edge.clone());
        }
        if let Some(finding) = anchor_fan_in_finding_from_predecessors(&static_pred_edges) {
            return Err(MixedAnchorFanInError(finding.message().to_owned()));
        }

        // Entry nodes: successors of START (dedup, END suppressed), edge order.
        let mut entry: Vec<String> = Vec::new();
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        if let Some(targets) = static_succ.get(START_NODE_ID) {
            for target in targets {
                if target == END_NODE_ID || seen.contains(target) {
                    continue;
                }
                seen.insert(target.clone());
                entry.push(target.clone());
            }
        }

        Ok(Scheduler {
            static_succ,
            start_anchored_succ,
            static_pred_edges,
            entry,
        })
    }

    /// Node ids that fire at trace start (successors of START; END suppressed).
    ///
    /// Borrows the precomputed entry list; hot scheduling path, so no per-call
    /// allocation or id clone.
    pub fn entry_nodes(&self) -> impl Iterator<Item = &str> {
        self.entry.iter().map(String::as_str)
    }

    /// Static successors after `node_id` completes (start-anchored excluded; END
    /// suppressed).
    ///
    /// Borrows the immutable adjacency; hot scheduling path, so no per-call
    /// allocation or id clone.
    pub fn successors_after(&self, node_id: &str) -> impl Iterator<Item = &str> {
        self.static_succ
            .get(node_id)
            .into_iter()
            .flatten()
            .filter(|t| *t != END_NODE_ID)
            .map(String::as_str)
    }

    /// StaticEdge objects targeting `node_id` (for gate computation).
    pub fn incoming_static_edges(&self, node_id: &str) -> &[StaticEdge] {
        self.static_pred_edges
            .get(node_id)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Successors wired via start-anchored edges; scheduled at `node_id`'s
    /// DISPATCH, not completion. END suppressed.
    ///
    /// Borrows the immutable adjacency; hot scheduling path, so no per-call
    /// allocation or id clone.
    pub fn start_anchored_successors(&self, node_id: &str) -> impl Iterator<Item = &str> {
        self.start_anchored_succ
            .get(node_id)
            .into_iter()
            .flatten()
            .filter(|t| *t != END_NODE_ID)
            .map(String::as_str)
    }
}

pub(crate) fn anchor_fan_in_finding(graph: &GraphRecord) -> Option<AnchorFanInFinding> {
    let mut static_pred_edges: BTreeMap<String, Vec<StaticEdge>> = BTreeMap::new();
    for edge in &graph.edges {
        if let Some(finding) = non_completion_start_finding(edge) {
            return Some(finding);
        }
        static_pred_edges
            .entry(edge.target.clone())
            .or_default()
            .push(edge.clone());
    }
    anchor_fan_in_finding_from_predecessors(&static_pred_edges)
}

fn non_completion_start_finding(edge: &StaticEdge) -> Option<AnchorFanInFinding> {
    (edge.source == START_NODE_ID
        && (edge.delay_after_predecessor_start_us.is_some()
            || edge.delay_after_predecessor_first_token_us.is_some()))
    .then(|| AnchorFanInFinding {
        kind: AnchorFanInKind::NonCompletionStart,
        target: edge.target.clone(),
        message: format!(
            "START edge to {:?} must use completion anchoring because START has no dispatch or first-token event",
            edge.target
        ),
    })
}

fn anchor_fan_in_finding_from_predecessors(
    static_pred_edges: &BTreeMap<String, Vec<StaticEdge>>,
) -> Option<AnchorFanInFinding> {
    for (target, edges) in static_pred_edges {
        let start_anchored: Vec<&StaticEdge> = edges
            .iter()
            .filter(|e| e.delay_after_predecessor_start_us.is_some())
            .collect();
        if start_anchored.is_empty() || edges.len() == 1 {
            continue;
        }
        let completion: Vec<&StaticEdge> = edges
            .iter()
            .filter(|e| e.delay_after_predecessor_start_us.is_none())
            .collect();
        let (kind, shape, detail) = if let Some(comp) = completion.first() {
            (
                AnchorFanInKind::Mixed,
                "mixed-anchor fan-in",
                format!(
                    "start-anchored edge {:?} -> {:?} (delay_after_predecessor_start_us) and \
completion edge {:?} -> {:?} arrive at the same node",
                    start_anchored[0].source, target, comp.source, target
                ),
            )
        } else {
            (
                AnchorFanInKind::MultipleStartAnchored,
                "multi-start-anchored fan-in",
                format!(
                    "start-anchored edges {:?} -> {:?} and {:?} -> {:?} \
(delay_after_predecessor_start_us) arrive at the same node",
                    start_anchored[0].source, target, start_anchored[1].source, target
                ),
            )
        };
        return Some(AnchorFanInFinding {
            kind,
            target: target.clone(),
            message: format!(
                "node {target:?}: {shape} is unsupported: {detail}. A start-anchored in-edge \
must be its target's ONLY in-edge."
            ),
        });
    }
    None
}

/// Zero the leading phase-start offsets (`--burst-phase-starts` collapse).
///
/// Returns a rebuilt graph; identity-preserving for untouched nodes/edges.
pub fn collapse_leading_start_offsets(graph: &GraphRecord) -> GraphRecord {
    let has_real_pred: std::collections::HashSet<&str> = graph
        .edges
        .iter()
        .filter(|e| e.source != START_NODE_ID)
        .map(|e| e.target.as_str())
        .collect();

    let mut new_graph = graph.clone();
    for edge in &mut new_graph.edges {
        if edge.source == START_NODE_ID && edge.min_start_delay_us.unwrap_or(0.0) != 0.0 {
            edge.min_start_delay_us = Some(0.0);
        }
    }
    for (nid, node) in new_graph.nodes.iter_mut() {
        if let Some(node) = node.as_llm_mut() {
            let has_delay = node.min_start_delay_us.unwrap_or(0.0) != 0.0;
            if has_delay && !has_real_pred.contains(nid.as_str()) {
                node.min_start_delay_us = Some(0.0);
            }
        }
    }
    new_graph
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph(json: &str) -> GraphRecord {
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn public_error_tuple_source_compatibility() {
        let fan_in = MixedAnchorFanInError("fan-in".to_owned());
        assert_eq!(fan_in.0, "fan-in");
    }

    #[test]
    fn error_compatibility_classifier_reads_existing_predecessor_adjacency() {
        let g = graph(
            r#"{
            "nodes": {"a": {"node_type":"llm","prompt":[],"output":"oa"},
                      "b": {"node_type":"llm","prompt":[],"output":"ob"},
                      "c": {"node_type":"llm","prompt":[],"output":"oc"}},
            "edges": [
                {"edge_type":"static","source":"a","target":"c","delay_after_predecessor_start_us":5.0},
                {"edge_type":"static","source":"b","target":"c"}
            ]
        }"#,
        );
        let mut predecessors = BTreeMap::new();
        for edge in &g.edges {
            predecessors
                .entry(edge.target.clone())
                .or_insert_with(Vec::new)
                .push(edge.clone());
        }

        let finding =
            anchor_fan_in_finding_from_predecessors(&predecessors).expect("mixed fan-in finding");
        assert_eq!(finding.kind(), AnchorFanInKind::Mixed);
        assert_eq!(finding.target(), "c");
    }

    #[test]
    fn entry_successors_and_end_suppressed() {
        let g = graph(
            r#"{
            "nodes": {
                "a": {"node_type":"llm","prompt":[],"output":"oa"},
                "b": {"node_type":"llm","prompt":[],"output":"ob"}
            },
            "edges": [
                {"edge_type":"static","source":"START","target":"a"},
                {"edge_type":"static","source":"a","target":"b"},
                {"edge_type":"static","source":"b","target":"END"}
            ]
        }"#,
        );
        let sched = Scheduler::new(&g).unwrap();
        assert_eq!(sched.entry_nodes().collect::<Vec<_>>(), vec!["a"]);
        assert_eq!(sched.successors_after("a").collect::<Vec<_>>(), vec!["b"]);
        assert_eq!(sched.successors_after("b").count(), 0); // END suppressed
    }

    #[test]
    fn start_anchored_routed_separately() {
        let g = graph(
            r#"{
            "nodes": {"a": {"node_type":"llm","prompt":[],"output":"oa"},
                      "b": {"node_type":"llm","prompt":[],"output":"ob"}},
            "edges": [
                {"edge_type":"static","source":"START","target":"a"},
                {"edge_type":"static","source":"a","target":"b","delay_after_predecessor_start_us":5.0}
            ]
        }"#,
        );
        let sched = Scheduler::new(&g).unwrap();
        assert_eq!(sched.successors_after("a").count(), 0);
        assert_eq!(
            sched.start_anchored_successors("a").collect::<Vec<_>>(),
            vec!["b"]
        );
    }

    #[test]
    fn start_edges_reject_dispatch_and_first_token_anchors() {
        for timing in [
            "\"delay_after_predecessor_start_us\":5.0",
            "\"delay_after_predecessor_first_token_us\":5.0",
        ] {
            let g = graph(&format!(
                r#"{{"nodes":{{"a":{{"node_type":"llm","prompt":[],"output":"oa"}}}},"edges":[{{"edge_type":"static","source":"START","target":"a",{timing}}}]}}"#
            ));
            assert!(Scheduler::new(&g).is_err(), "{timing}");
        }
    }

    #[test]
    fn mixed_anchor_fan_in_error_compatibility_projects_typed_finding() {
        let g = graph(
            r#"{
            "nodes": {"a": {"node_type":"llm","prompt":[],"output":"oa"},
                      "b": {"node_type":"llm","prompt":[],"output":"ob"},
                      "c": {"node_type":"llm","prompt":[],"output":"oc"}},
            "edges": [
                {"edge_type":"static","source":"a","target":"c","delay_after_predecessor_start_us":5.0},
                {"edge_type":"static","source":"b","target":"c"}
            ]
        }"#,
        );
        let error = match Scheduler::new(&g) {
            Err(error) => error,
            Ok(_) => panic!("mixed-anchor fan-in must be rejected"),
        };
        let finding = anchor_fan_in_finding(&g).expect("mixed fan-in finding");
        assert_eq!(finding.kind(), AnchorFanInKind::Mixed);
        assert_eq!(finding.target(), "c");
        assert_eq!(error.0, finding.message());
    }

    #[test]
    fn multi_start_anchor_fan_in_error_compatibility_projects_typed_finding() {
        let g = graph(
            r#"{
            "nodes": {"a": {"node_type":"llm","prompt":[],"output":"oa"},
                      "b": {"node_type":"llm","prompt":[],"output":"ob"},
                      "c": {"node_type":"llm","prompt":[],"output":"oc"}},
            "edges": [
                {"edge_type":"static","source":"a","target":"c","delay_after_predecessor_start_us":5.0},
                {"edge_type":"static","source":"b","target":"c","delay_after_predecessor_start_us":6.0}
            ]
        }"#,
        );
        let error = match Scheduler::new(&g) {
            Err(error) => error,
            Ok(_) => panic!("multi-start-anchored fan-in must be rejected"),
        };
        let finding = anchor_fan_in_finding(&g).expect("multi-start fan-in finding");
        assert_eq!(finding.kind(), AnchorFanInKind::MultipleStartAnchored);
        assert_eq!(finding.target(), "c");
        assert_eq!(error.0, finding.message());
    }

    #[test]
    fn collapse_zeros_leading_start_offset() {
        let g = graph(
            r#"{
            "nodes": {"a": {"node_type":"llm","prompt":[],"output":"oa","min_start_delay_us":9.0}},
            "edges": [
                {"edge_type":"static","source":"START","target":"a","min_start_delay_us":7.0}
            ]
        }"#,
        );
        let collapsed = collapse_leading_start_offsets(&g);
        assert_eq!(collapsed.edges[0].min_start_delay_us, Some(0.0));
        assert_eq!(
            collapsed
                .nodes
                .get("a")
                .unwrap()
                .as_llm()
                .unwrap()
                .min_start_delay_us,
            Some(0.0)
        );
    }
}
