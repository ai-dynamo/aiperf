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
    let mut errors = Vec::new();
    let node_ids: BTreeSet<&str> = graph.nodes.keys().map(String::as_str).collect();

    // 1. Every edge endpoint is START/END or a declared node.
    for edge in &graph.edges {
        if edge.source != START_NODE_ID && !node_ids.contains(edge.source.as_str()) {
            errors.push(ValidationError(format!(
                "edge source {:?} is not a declared node",
                edge.source
            )));
        }
        if edge.target != END_NODE_ID && !node_ids.contains(edge.target.as_str()) {
            errors.push(ValidationError(format!(
                "edge target {:?} is not a declared node",
                edge.target
            )));
        }
    }

    // 2. Every node's output and input channels are declared in `state`.
    for (nid, node) in &graph.nodes {
        if !graph.state.contains_key(node.output()) {
            errors.push(ValidationError(format!(
                "node {nid:?} writes undeclared channel {:?}",
                node.output()
            )));
        }
        for channel in node.read_channels() {
            if !graph.state.contains_key(channel) {
                errors.push(ValidationError(format!(
                    "node {nid:?} reads undeclared channel {:?}",
                    channel
                )));
            }
        }
    }

    // 3. Every node is reachable from START (an unreachable node is never
    //    scheduled, so it — and anything gated on its output — never fires).
    let reachable = reachable_from_start(graph);
    for nid in &node_ids {
        if !reachable.contains(*nid) {
            errors.push(ValidationError(format!(
                "node {nid:?} is unreachable from START (it would never fire)"
            )));
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
            errors.push(ValidationError(format!(
                "node {nid:?} can never fire: input channel {chan:?} needs {need} \
producer(s) but {reason}"
            )));
            break;
        }
    }

    errors
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
    use serde_json::json;

    fn graph(v: serde_json::Value) -> GraphRecord {
        serde_json::from_value(v).unwrap()
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
