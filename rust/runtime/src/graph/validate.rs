// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Structural graph validation — fail-fast on topologies the runtime cannot run
//! (and that would otherwise deadlock/hang on the live clock).
//!
//! This is deliberately small: it checks only the structural invariants the
//! dataflow runtime depends on, not the parse-plane grammar. It is not a
//! complete deadlock-freedom prover — a Sim dry-run remains the backstop — but
//! it catches the common breakages: dangling edges, undeclared channels,
//! unreachable nodes, and channels no reachable producer can satisfy.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::graph::inspect::{GraphInspectionIssue, GraphInspectionSeverity};
use crate::graph::model::{Count, END_NODE_ID, GraphRecord, START_NODE_ID};

/// One structural problem found in a graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationError(pub String);

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl std::error::Error for ValidationError {}

/// Validate `graph`, returning every structural problem found (empty = valid).
pub fn validate(graph: &GraphRecord) -> Vec<ValidationError> {
    validate_detailed(graph)
        .into_iter()
        .map(|issue| ValidationError(issue.message))
        .collect()
}

/// Validate `graph`, returning structural problems with stable inspection data.
pub fn validate_detailed(graph: &GraphRecord) -> Vec<GraphInspectionIssue> {
    let mut issues = Vec::new();
    let node_ids: BTreeSet<&str> = graph.nodes.keys().map(String::as_str).collect();

    // 1. Every edge endpoint is START/END or a declared node.
    for (edge_index, edge) in graph.edges.iter().enumerate() {
        if edge.source != START_NODE_ID && !node_ids.contains(edge.source.as_str()) {
            issues.push(issue(
                "edge-source-unknown",
                Some(format!("graph.edges[{edge_index}].source")),
                format!("edge source {:?} is not a declared node", edge.source),
                [
                    ("source", edge.source.as_str()),
                    ("target", edge.target.as_str()),
                    ("edge_index", &edge_index.to_string()),
                ],
            ));
        }
        if edge.target != END_NODE_ID && !node_ids.contains(edge.target.as_str()) {
            issues.push(issue(
                "edge-target-unknown",
                Some(format!("graph.edges[{edge_index}].target")),
                format!("edge target {:?} is not a declared node", edge.target),
                [
                    ("source", edge.source.as_str()),
                    ("target", edge.target.as_str()),
                    ("edge_index", &edge_index.to_string()),
                ],
            ));
        }
    }

    // 2. Every node's output and input channels are declared in `state`.
    for (nid, node) in &graph.nodes {
        if !graph.state.contains_key(node.output()) {
            issues.push(issue(
                "channel-write-undeclared",
                Some(format!("graph.nodes.{nid}.output")),
                format!("node {nid:?} writes undeclared channel {:?}", node.output()),
                [("node_id", nid.as_str()), ("channel", node.output())],
            ));
        }
    }
    for (nid, node) in &graph.nodes {
        for (input_index, channel) in node.read_channels().iter().enumerate() {
            if graph.state.contains_key(*channel) {
                continue;
            }
            issues.push(issue(
                "channel-read-undeclared",
                Some(format!("graph.nodes.{nid}.inputs[{input_index}]")),
                format!("node {nid:?} reads undeclared channel {:?}", channel),
                [("node_id", nid.as_str()), ("channel", *channel)],
            ));
        }
    }

    // 3. Every node is reachable from START (an unreachable node is never
    //    scheduled, so it — and anything gated on its output — never fires).
    let reachable = reachable_from_start(graph);
    for nid in &node_ids {
        if !reachable.contains(*nid) {
            issues.push(issue(
                "node-unreachable",
                Some(format!("graph.nodes.{nid}")),
                format!("node {nid:?} is unreachable from START (it would never fire)"),
                [("node_id", *nid)],
            ));
        }
    }

    // 4. Deadlock-freedom by fireability fixpoint. A node can fire once every
    //    input channel has `count` producers that can themselves fire. Iterating
    //    to a fixpoint, any reachable node that never becomes fireable is a
    //    deadlock — which subsumes self-dependency, unreachable producers,
    //    count>producers, AND mutual/cyclic gates (n0 waits on n1, n1 on n0).
    let mut writers: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for (nid, node) in &graph.nodes {
        writers.entry(node.output()).or_default().push(nid);
    }
    // `count` target for a requirement: `all` resolves to the channel's total
    // producer count (an unreachable producer then makes it unsatisfiable).
    let target = |chan: &str, count: &Count| -> usize {
        match count {
            Count::N(k) => (*k).max(0) as usize,
            Count::Word(_) => writers.get(chan).map_or(0, Vec::len),
        }
    };
    let fireable_producers = |chan: &str, fireable: &BTreeSet<&str>| -> usize {
        writers
            .get(chan)
            .map_or(0, |ws| ws.iter().filter(|w| fireable.contains(**w)).count())
    };

    let mut fireable: BTreeSet<&str> = BTreeSet::new();
    loop {
        let mut changed = false;
        for (nid, node) in &graph.nodes {
            if !reachable.contains(nid.as_str()) || fireable.contains(nid.as_str()) {
                continue;
            }
            if node
                .input_requirements()
                .iter()
                .any(|requirement| !graph.state.contains_key(&requirement.channel))
            {
                continue;
            }
            let gated = node.input_requirements().iter().any(|req| {
                fireable_producers(&req.channel, &fireable) < target(&req.channel, &req.count)
            });
            if !gated {
                fireable.insert(nid.as_str());
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    for (nid, node) in &graph.nodes {
        if !reachable.contains(nid.as_str()) || fireable.contains(nid.as_str()) {
            continue;
        }
        if node
            .input_requirements()
            .iter()
            .any(|requirement| !graph.state.contains_key(&requirement.channel))
        {
            continue;
        }
        // Reachable but never fireable: name the blocking input and why.
        for req in node.input_requirements() {
            let chan = req.channel.as_str();
            let need = target(chan, &req.count);
            let can = fireable_producers(chan, &fireable);
            if can >= need {
                continue;
            }
            let all_producers = writers.get(chan).map_or(0, Vec::len);
            let reason = if writers.get(chan).map(Vec::as_slice) == Some(&[nid.as_str()]) {
                "it is the sole producer (self-deadlock)".to_string()
            } else if all_producers < need {
                format!("only {all_producers} producer(s) exist")
            } else {
                format!("only {can} of {all_producers} producer(s) can fire (dependency cycle)")
            };
            issues.push(issue(
                "node-never-fireable",
                Some(format!("graph.nodes.{nid}")),
                format!(
                    "node {nid:?} can never fire: input channel {chan:?} needs {need} \
producer(s) but {reason}"
                ),
                [
                    ("node_id", nid.as_str()),
                    ("channel", chan),
                    ("needed", &need.to_string()),
                    ("fireable_producers", &can.to_string()),
                    ("all_producers", &all_producers.to_string()),
                ],
            ));
            break;
        }
    }

    issues
}

fn issue<'a, const N: usize>(
    code: &str,
    location: Option<String>,
    message: String,
    context: [(&'a str, &'a str); N],
) -> GraphInspectionIssue {
    GraphInspectionIssue {
        code: code.to_string(),
        severity: GraphInspectionSeverity::Error,
        trace_id: None,
        phase: None,
        location,
        message,
        context: context
            .into_iter()
            .map(|(key, value)| (key.to_string(), value.to_string()))
            .collect(),
    }
}

/// Node ids reachable from START via any static edge (including start-anchored).
fn reachable_from_start(graph: &GraphRecord) -> BTreeSet<String> {
    let mut succ: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for edge in &graph.edges {
        succ.entry(edge.source.as_str())
            .or_default()
            .push(edge.target.as_str());
    }
    let mut seen = BTreeSet::new();
    let mut queue: VecDeque<&str> = VecDeque::new();
    queue.push_back(START_NODE_ID);
    while let Some(cur) = queue.pop_front() {
        for &next in succ.get(cur).map(Vec::as_slice).unwrap_or(&[]) {
            if next != END_NODE_ID && seen.insert(next.to_string()) {
                queue.push_back(next);
            }
        }
    }
    seen
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::inspect::GraphInspectionSeverity;
    use serde_json::json;

    fn graph(v: serde_json::Value) -> GraphRecord {
        serde_json::from_value(v).unwrap()
    }

    fn issue_pairs(
        issues: &[crate::graph::inspect::GraphInspectionIssue],
    ) -> Vec<(&str, Option<&str>)> {
        issues
            .iter()
            .map(|issue| (issue.code.as_str(), issue.location.as_deref()))
            .collect()
    }

    #[test]
    fn detailed_reports_stable_code_locations_and_context() {
        let g = graph(json!({
            "state": {
                "blocked_input": {"type":"messages","reducer":"add_messages"},
                "good": {"type":"messages","reducer":"add_messages"}
            },
            "nodes": {
                "blocked": {"node_type":"llm","prompt":[],"output":"good","inputs":[{"channel":"blocked_input","count":1}]},
                "orphan": {"node_type":"llm","prompt":[],"output":"good"},
                "reader": {"node_type":"llm","prompt":[],"output":"good","inputs":[{"channel":"missing_read","count":1}]},
                "writer": {"node_type":"llm","prompt":[],"output":"missing_write"}
            },
            "edges": [
                {"edge_type":"static","source":"ghost_source","target":"writer"},
                {"edge_type":"static","source":"START","target":"ghost_target"},
                {"edge_type":"static","source":"START","target":"writer"},
                {"edge_type":"static","source":"START","target":"reader"},
                {"edge_type":"static","source":"START","target":"blocked"}
            ]
        }));

        let issues = crate::graph::inspect::validate_detailed(&g);
        assert_eq!(
            issue_pairs(&issues),
            vec![
                ("edge-source-unknown", Some("graph.edges[0].source")),
                ("edge-target-unknown", Some("graph.edges[1].target")),
                (
                    "channel-write-undeclared",
                    Some("graph.nodes.writer.output")
                ),
                (
                    "channel-read-undeclared",
                    Some("graph.nodes.reader.inputs[0]")
                ),
                ("node-unreachable", Some("graph.nodes.orphan")),
                ("node-never-fireable", Some("graph.nodes.blocked")),
            ],
        );

        let expected_contexts = [
            [
                ("edge_index", "0"),
                ("source", "ghost_source"),
                ("target", "writer"),
            ]
            .as_slice(),
            [
                ("edge_index", "1"),
                ("source", "START"),
                ("target", "ghost_target"),
            ]
            .as_slice(),
            [("channel", "missing_write"), ("node_id", "writer")].as_slice(),
            [("channel", "missing_read"), ("node_id", "reader")].as_slice(),
            [("node_id", "orphan")].as_slice(),
            [
                ("all_producers", "0"),
                ("channel", "blocked_input"),
                ("fireable_producers", "0"),
                ("needed", "1"),
                ("node_id", "blocked"),
            ]
            .as_slice(),
        ];
        for (issue, expected_context) in issues.iter().zip(expected_contexts) {
            assert_eq!(issue.severity, GraphInspectionSeverity::Error);
            assert!(issue.trace_id.is_none());
            assert!(issue.phase.is_none());
            assert!(!issue.message.is_empty() && issue.message.len() <= 512);
            assert_eq!(
                issue
                    .context
                    .iter()
                    .map(|(key, value)| (key.as_str(), value.as_str()))
                    .collect::<Vec<_>>(),
                expected_context,
            );
        }
    }

    #[test]
    fn accepts_a_valid_chain() {
        let g = graph(json!({
            "state": {"a": {"type":"messages","reducer":"add_messages"},
                      "b": {"type":"messages","reducer":"add_messages"}},
            "nodes": {
                "n0": {"node_type":"llm","prompt":[],"output":"a"},
                "n1": {"node_type":"llm","prompt":[],"output":"b","inputs":[{"channel":"a","count":1}]}},
            "edges": [
                {"edge_type":"static","source":"START","target":"n0"},
                {"edge_type":"static","source":"n0","target":"n1"},
                {"edge_type":"static","source":"n1","target":"END"}]
        }));
        assert!(validate(&g).is_empty());
    }

    #[test]
    fn flags_the_exotic_breakages() {
        // dangling edge target
        assert!(!validate(&graph(json!({
            "state":{}, "nodes":{}, "edges":[{"edge_type":"static","source":"START","target":"ghost"}]
        }))).is_empty());
        // self-dependency
        assert!(validate(&graph(json!({
            "state":{"a":{"type":"messages","reducer":"add_messages"}},
            "nodes":{"n0":{"node_type":"llm","prompt":[],"output":"a","inputs":[{"channel":"a","count":1}]}},
            "edges":[{"edge_type":"static","source":"START","target":"n0"}]
        }))).iter().any(|e| e.0.contains("self-deadlock")));
        // unreachable producer
        assert!(validate(&graph(json!({
            "state":{"c":{"type":"messages","reducer":"add_messages"},"z":{"type":"messages","reducer":"add_messages"}},
            "nodes":{"producer":{"node_type":"llm","prompt":[],"output":"c"},
                     "reader":{"node_type":"llm","prompt":[],"output":"z","inputs":[{"channel":"c","count":1}]}},
            "edges":[{"edge_type":"static","source":"START","target":"reader"}]
        }))).iter().any(|e| e.0.contains("unreachable")));
        // count exceeds producers
        assert!(validate(&graph(json!({
            "state":{"c":{"type":"messages","reducer":"add_messages"},"z":{"type":"messages","reducer":"add_messages"}},
            "nodes":{"p0":{"node_type":"llm","prompt":[],"output":"c"},
                     "j":{"node_type":"llm","prompt":[],"output":"z","inputs":[{"channel":"c","count":2}]}},
            "edges":[{"edge_type":"static","source":"START","target":"p0"},
                     {"edge_type":"static","source":"START","target":"j"}]
        }))).iter().any(|e| e.0.contains("needs 2 producer")));
    }

    #[test]
    fn catches_mutual_gate_cycle() {
        // n0 gates on c1 (only n1 writes it); n1 gates on c0 (only n0 writes it).
        // Both reachable from START, but neither can ever fire — a dependency
        // cycle the fixpoint catches (the earlier ad-hoc checks did not).
        let g = graph(json!({
            "state":{"c0":{"type":"messages","reducer":"add_messages"},
                     "c1":{"type":"messages","reducer":"add_messages"}},
            "nodes":{
                "n0":{"node_type":"llm","prompt":[],"output":"c0","inputs":[{"channel":"c1","count":1}]},
                "n1":{"node_type":"llm","prompt":[],"output":"c1","inputs":[{"channel":"c0","count":1}]}},
            "edges":[{"edge_type":"static","source":"START","target":"n0"},
                     {"edge_type":"static","source":"START","target":"n1"}]
        }));
        let errs = validate(&g);
        assert!(
            errs.iter()
                .any(|e| e.0.contains("can never fire") && e.0.contains("cycle")),
            "expected a dependency-cycle deadlock, got {errs:?}"
        );
    }
}
