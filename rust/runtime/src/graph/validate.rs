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
use crate::graph::model::{Count, END_NODE_ID, GraphRecord, PromptItem, START_NODE_ID};
use crate::graph::scheduler::Scheduler;
use crate::graph::static_readiness::analyze_static_readiness;

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
    let mut errors = collect_findings(graph)
        .into_iter()
        .map(|finding| ValidationError(finding.message()))
        .collect::<Vec<_>>();
    errors.extend(
        non_completion_start_errors(graph)
            .into_iter()
            .map(|(_, error)| error),
    );
    errors
}

/// Validate `graph`, returning structural problems with stable inspection data.
pub fn validate_detailed(graph: &GraphRecord) -> Vec<GraphInspectionIssue> {
    let findings = collect_findings(graph);
    let mut issues = Vec::new();
    for rank in 0..=5 {
        for finding in &findings {
            if finding.detailed_rank() != rank {
                continue;
            }
            if let Some(issue) = finding.detailed_issue() {
                issues.push(issue);
            }
        }
    }
    issues.extend(
        non_completion_start_errors(graph)
            .into_iter()
            .map(|(edge_index, error)| {
                issue(
                    "start-anchor-non-completion",
                    Some(format!("graph.edges[{edge_index}]")),
                    error.0,
                    [("edge_index", &edge_index.to_string())],
                )
            }),
    );
    issues
}

fn non_completion_start_errors(graph: &GraphRecord) -> Vec<(usize, ValidationError)> {
    graph.edges.iter().enumerate().filter(|(_, edge)| edge.source == START_NODE_ID
        && (edge.delay_after_predecessor_start_us.is_some() || edge.delay_after_predecessor_first_token_us.is_some()))
        .map(|(index, _)| (index, ValidationError("START edges must use completion anchoring because START has no dispatch or first-token event".to_owned())))
        .collect()
}

#[derive(Clone, Debug)]
enum Finding {
    EdgeSourceUnknown {
        edge_index: usize,
        source: String,
        target: String,
    },
    EdgeTargetUnknown {
        edge_index: usize,
        source: String,
        target: String,
    },
    ChannelWriteUndeclared {
        node_id: String,
        channel: String,
    },
    ChannelReadUndeclared {
        node_id: String,
        channel: String,
        location: ChannelReadLocation,
    },
    NodeUnreachable {
        node_id: String,
    },
    NodeNeverFireable {
        node_id: String,
        channel: String,
        needed: usize,
        fireable_producers: usize,
        all_producers: usize,
        reason: String,
        has_undeclared_gate_input: bool,
    },
    StaticChannelReadinessDeadlock {
        blocked_node_ids: Vec<String>,
    },
}

#[derive(Clone, Debug)]
enum ChannelReadLocation {
    Input(usize),
    Splice(usize),
}

impl Finding {
    fn message(&self) -> String {
        match self {
            Finding::EdgeSourceUnknown { source, .. } => {
                format!("edge source {source:?} is not a declared node")
            }
            Finding::EdgeTargetUnknown { target, .. } => {
                format!("edge target {target:?} is not a declared node")
            }
            Finding::ChannelWriteUndeclared { node_id, channel } => {
                format!("node {node_id:?} writes undeclared channel {channel:?}")
            }
            Finding::ChannelReadUndeclared {
                node_id, channel, ..
            } => format!("node {node_id:?} reads undeclared channel {channel:?}"),
            Finding::NodeUnreachable { node_id } => {
                format!("node {node_id:?} is unreachable from START (it would never fire)")
            }
            Finding::NodeNeverFireable {
                node_id,
                channel,
                needed,
                reason,
                ..
            } => format!(
                "node {node_id:?} can never fire: input channel {channel:?} needs {needed} \
producer(s) but {reason}"
            ),
            Finding::StaticChannelReadinessDeadlock { blocked_node_ids } => format!(
                "graph cannot make static readiness progress; blocked nodes: {}",
                blocked_node_ids
                    .iter()
                    .map(|node_id| format!("{node_id:?}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    }

    const fn detailed_rank(&self) -> usize {
        match self {
            Finding::EdgeSourceUnknown { .. } | Finding::EdgeTargetUnknown { .. } => 0,
            Finding::ChannelWriteUndeclared { .. } => 1,
            Finding::ChannelReadUndeclared { .. } => 2,
            Finding::NodeUnreachable { .. } => 3,
            Finding::NodeNeverFireable { .. } => 4,
            Finding::StaticChannelReadinessDeadlock { .. } => 5,
        }
    }

    fn detailed_issue(&self) -> Option<GraphInspectionIssue> {
        match self {
            Finding::EdgeSourceUnknown {
                edge_index,
                source,
                target,
            } => Some(issue(
                "edge-source-unknown",
                Some(format!("graph.edges[{edge_index}].source")),
                self.message(),
                [
                    ("source", source.as_str()),
                    ("target", target.as_str()),
                    ("edge_index", &edge_index.to_string()),
                ],
            )),
            Finding::EdgeTargetUnknown {
                edge_index,
                source,
                target,
            } => Some(issue(
                "edge-target-unknown",
                Some(format!("graph.edges[{edge_index}].target")),
                self.message(),
                [
                    ("source", source.as_str()),
                    ("target", target.as_str()),
                    ("edge_index", &edge_index.to_string()),
                ],
            )),
            Finding::ChannelWriteUndeclared { node_id, channel } => Some(issue(
                "channel-write-undeclared",
                Some(format!("graph.nodes.{node_id}.output")),
                self.message(),
                [("node_id", node_id.as_str()), ("channel", channel.as_str())],
            )),
            Finding::ChannelReadUndeclared {
                node_id,
                channel,
                location,
            } => {
                let location = match location {
                    ChannelReadLocation::Input(index) => {
                        format!("graph.nodes.{node_id}.inputs[{index}]")
                    }
                    ChannelReadLocation::Splice(index) => {
                        format!("graph.nodes.{node_id}.items[{index}].splice")
                    }
                };
                Some(issue(
                    "channel-read-undeclared",
                    Some(location),
                    self.message(),
                    [("node_id", node_id.as_str()), ("channel", channel.as_str())],
                ))
            }
            Finding::NodeUnreachable { node_id } => Some(issue(
                "node-unreachable",
                Some(format!("graph.nodes.{node_id}")),
                self.message(),
                [("node_id", node_id.as_str())],
            )),
            Finding::NodeNeverFireable {
                node_id,
                channel,
                needed,
                fireable_producers,
                all_producers,
                has_undeclared_gate_input,
                ..
            } => {
                if *has_undeclared_gate_input {
                    return None;
                }
                Some(issue(
                    "node-never-fireable",
                    Some(format!("graph.nodes.{node_id}")),
                    self.message(),
                    [
                        ("node_id", node_id.as_str()),
                        ("channel", channel.as_str()),
                        ("needed", &needed.to_string()),
                        ("fireable_producers", &fireable_producers.to_string()),
                        ("all_producers", &all_producers.to_string()),
                    ],
                ))
            }
            Finding::StaticChannelReadinessDeadlock { blocked_node_ids } => {
                let blocked_nodes = match serde_json::to_string(blocked_node_ids) {
                    Ok(blocked_nodes) => blocked_nodes,
                    Err(_) => "[]".to_string(),
                };
                Some(issue(
                    "static-channel-readiness-deadlock",
                    Some("graph".to_string()),
                    self.message(),
                    [("blocked_nodes", blocked_nodes.as_str())],
                ))
            }
        }
    }
}

fn collect_findings(graph: &GraphRecord) -> Vec<Finding> {
    let mut findings = Vec::new();
    let node_ids: BTreeSet<&str> = graph.nodes.keys().map(String::as_str).collect();

    for (edge_index, edge) in graph.edges.iter().enumerate() {
        if edge.source != START_NODE_ID && !node_ids.contains(edge.source.as_str()) {
            findings.push(Finding::EdgeSourceUnknown {
                edge_index,
                source: edge.source.clone(),
                target: edge.target.clone(),
            });
        }
        if edge.target != END_NODE_ID && !node_ids.contains(edge.target.as_str()) {
            findings.push(Finding::EdgeTargetUnknown {
                edge_index,
                source: edge.source.clone(),
                target: edge.target.clone(),
            });
        }
    }

    for (nid, node) in &graph.nodes {
        if !graph.state.contains_key(node.output()) {
            findings.push(Finding::ChannelWriteUndeclared {
                node_id: nid.clone(),
                channel: node.output().to_string(),
            });
        }
        for (input_index, requirement) in node.input_requirements().iter().enumerate() {
            if !graph.state.contains_key(&requirement.channel) {
                findings.push(Finding::ChannelReadUndeclared {
                    node_id: nid.clone(),
                    channel: requirement.channel.clone(),
                    location: ChannelReadLocation::Input(input_index),
                });
            }
        }
        if let Some(llm) = node.as_llm() {
            for (item_index, item) in llm.items.iter().enumerate() {
                let PromptItem::Splice { splice } = item else {
                    continue;
                };
                if !graph.state.contains_key(splice) {
                    findings.push(Finding::ChannelReadUndeclared {
                        node_id: nid.clone(),
                        channel: splice.clone(),
                        location: ChannelReadLocation::Splice(item_index),
                    });
                }
            }
        }
    }

    let reachable = reachable_from_start(graph);
    for nid in &node_ids {
        if !reachable.contains(*nid) {
            findings.push(Finding::NodeUnreachable {
                node_id: (*nid).to_string(),
            });
        }
    }

    let mut writers: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for (nid, node) in &graph.nodes {
        writers.entry(node.output()).or_default().push(nid);
    }
    let target = |chan: &str, count: &Count| -> usize {
        match count {
            Count::N(k) => (*k).max(0) as usize,
            Count::Word(_) => writers.get(chan).map_or(0, Vec::len),
        }
    };
    let fireable_producers = |chan: &str, fireable: &BTreeSet<&str>| -> usize {
        writers.get(chan).map_or(0, |ws| {
            ws.iter()
                .filter(|writer| fireable.contains(**writer))
                .count()
        })
    };

    let mut fireable: BTreeSet<&str> = BTreeSet::new();
    loop {
        let mut changed = false;
        for (nid, node) in &graph.nodes {
            if !reachable.contains(nid.as_str()) || fireable.contains(nid.as_str()) {
                continue;
            }
            let gated = node.input_requirements().iter().any(|requirement| {
                fireable_producers(&requirement.channel, &fireable)
                    < target(&requirement.channel, &requirement.count)
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
        for requirement in node.input_requirements() {
            let channel = requirement.channel.as_str();
            let needed = target(channel, &requirement.count);
            let fireable_producers = fireable_producers(channel, &fireable);
            if fireable_producers >= needed {
                continue;
            }
            let all_producers = writers.get(channel).map_or(0, Vec::len);
            let reason = if writers.get(channel).map(Vec::as_slice) == Some(&[nid.as_str()]) {
                "it is the sole producer (self-deadlock)".to_string()
            } else if all_producers < needed {
                format!("only {all_producers} producer(s) exist")
            } else {
                format!(
                    "only {fireable_producers} of {all_producers} producer(s) can fire (dependency cycle)"
                )
            };
            findings.push(Finding::NodeNeverFireable {
                node_id: nid.clone(),
                channel: channel.to_string(),
                needed,
                fireable_producers,
                all_producers,
                reason,
                has_undeclared_gate_input: node
                    .input_requirements()
                    .iter()
                    .any(|input| !graph.state.contains_key(&input.channel)),
            });
            break;
        }
    }

    let prerequisites_hold = findings.iter().all(|finding| {
        !matches!(
            finding,
            Finding::EdgeSourceUnknown { .. }
                | Finding::EdgeTargetUnknown { .. }
                | Finding::ChannelWriteUndeclared { .. }
                | Finding::ChannelReadUndeclared { .. }
                | Finding::NodeUnreachable { .. }
                | Finding::NodeNeverFireable { .. }
        )
    });
    if prerequisites_hold && let Ok(scheduler) = Scheduler::new(graph) {
        let analysis = analyze_static_readiness(graph, &scheduler);
        if !analysis.blocked_node_ids.is_empty() {
            findings.push(Finding::StaticChannelReadinessDeadlock {
                blocked_node_ids: analysis.blocked_node_ids,
            });
        }
    }

    findings
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
    fn legacy_validation_preserves_message_order_and_cardinality() {
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

        assert_eq!(
            validate(&g)
                .into_iter()
                .map(|error| error.0)
                .collect::<Vec<_>>(),
            vec![
                "edge source \"ghost_source\" is not a declared node",
                "edge target \"ghost_target\" is not a declared node",
                "node \"reader\" reads undeclared channel \"missing_read\"",
                "node \"writer\" writes undeclared channel \"missing_write\"",
                "node \"orphan\" is unreachable from START (it would never fire)",
                "node \"blocked\" can never fire: input channel \"blocked_input\" needs 1 producer(s) but only 0 producer(s) exist",
                "node \"reader\" can never fire: input channel \"missing_read\" needs 1 producer(s) but only 0 producer(s) exist",
            ],
        );
    }

    #[test]
    fn detailed_distinguishes_input_and_splice_read_locations() {
        let g = graph(json!({
            "state": {"output": {"type":"messages","reducer":"add_messages"}},
            "nodes": {
                "reader": {
                    "node_type":"llm",
                    "prompt":[],
                    "output":"output",
                    "inputs":[{"channel":"missing_input","count":1}],
                    "items":[{"splice":"missing_splice"}]
                }
            },
            "edges": [{"edge_type":"static","source":"START","target":"reader"}]
        }));

        assert_eq!(
            issue_pairs(
                &crate::graph::inspect::validate_detailed(&g)
                    .into_iter()
                    .filter(|issue| issue.code == "channel-read-undeclared")
                    .collect::<Vec<_>>(),
            ),
            vec![
                (
                    "channel-read-undeclared",
                    Some("graph.nodes.reader.inputs[0]")
                ),
                (
                    "channel-read-undeclared",
                    Some("graph.nodes.reader.items[0].splice")
                ),
            ],
        );
    }

    #[test]
    fn legacy_and_detailed_share_messages_for_common_findings() {
        let g = graph(json!({
            "state": {
                "blocked_input": {"type":"messages","reducer":"add_messages"},
                "good": {"type":"messages","reducer":"add_messages"}
            },
            "nodes": {
                "blocked": {"node_type":"llm","prompt":[],"output":"good","inputs":[{"channel":"blocked_input","count":1}]},
                "writer": {"node_type":"llm","prompt":[],"output":"missing_write"}
            },
            "edges": [
                {"edge_type":"static","source":"ghost_source","target":"writer"},
                {"edge_type":"static","source":"START","target":"writer"},
                {"edge_type":"static","source":"START","target":"blocked"}
            ]
        }));

        assert_eq!(
            validate(&g)
                .into_iter()
                .map(|error| error.0)
                .collect::<Vec<_>>(),
            validate_detailed(&g)
                .into_iter()
                .map(|issue| issue.message)
                .collect::<Vec<_>>(),
        );
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
    fn static_channel_readiness_rejects_mixed_dependency_deadlock() {
        let g = graph(json!({
            "state": {
                "produced": {"type":"messages","reducer":"add_messages"},
                "done": {"type":"messages","reducer":"add_messages"}
            },
            "nodes": {
                "reader": {
                    "node_type":"llm", "prompt":[], "output":"done",
                    "inputs":[{"channel":"produced","count":1}]
                },
                "producer": {"node_type":"llm", "prompt":[], "output":"produced"}
            },
            "edges": [
                {"source":"START","target":"reader"},
                {"source":"reader","target":"producer"}
            ]
        }));

        assert!(
            validate(&g)
                .iter()
                .all(|error| !error.0.contains("can never fire")),
            "each channel has enough declared producers"
        );
        let issues = validate_detailed(&g);
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].code, "static-channel-readiness-deadlock");
        assert_eq!(issues[0].location.as_deref(), Some("graph"));
        assert_eq!(
            issues[0].context.get("blocked_nodes"),
            Some(&"[\"reader\",\"producer\"]".to_owned())
        );
    }

    #[test]
    fn static_channel_readiness_treats_completion_predecessors_as_or_triggers() {
        let g = graph(json!({
            "state": {"a": {}, "b": {}, "done": {}},
            "nodes": {
                "a": {"output":"a"},
                "b": {"output":"b", "inputs":[{"channel":"a","count":1}]},
                "target": {"output":"done"}
            },
            "edges": [
                {"source":"START","target":"a"},
                {"source":"START","target":"b"},
                {"source":"a","target":"target"},
                {"source":"b","target":"target"}
            ]
        }));

        assert!(validate_detailed(&g).is_empty());
    }

    #[test]
    fn static_channel_readiness_admits_dispatch_successor_after_output_completion() {
        let g = graph(json!({
            "state": {"produced": {}, "done": {}},
            "nodes": {
                "producer": {"output":"produced"},
                "reader": {
                    "output":"done",
                    "inputs":[{"channel":"produced","count":1}]
                }
            },
            "edges": [
                {"source":"START","target":"producer"},
                {
                    "source":"producer","target":"reader",
                    "delay_after_predecessor_start_us":0.0
                }
            ]
        }));

        assert!(validate_detailed(&g).is_empty());
    }

    #[test]
    fn rejects_non_completion_start_anchors() {
        for timing in [
            "\"delay_after_predecessor_start_us\":5.0",
            "\"delay_after_predecessor_first_token_us\":5.0",
        ] {
            let graph: GraphRecord = serde_json::from_str(&format!(
                r#"{{"nodes":{{"n":{{"node_type":"llm","prompt":[],"output":"out"}}}},"edges":[{{"edge_type":"static","source":"START","target":"n",{timing}}}]}}"#
            ))
            .expect("valid graph");
            assert!(!validate(&graph).is_empty(), "{timing}");
        }
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
