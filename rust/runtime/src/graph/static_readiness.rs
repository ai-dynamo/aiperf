// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Static scheduler-and-channel readiness analysis shared by validation and inspection.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::graph::model::{
    Count, CountValidationError, ExecutableGraphNode, GraphRecord, START_NODE_ID,
};
use crate::graph::scheduler::Scheduler;

/// Deterministic static readiness facts for one declared graph.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct StaticReadinessAnalysis {
    /// Illustrative admission waves in normalized node order.
    pub(crate) waves: Vec<StaticReadinessWave>,
    /// Declared nodes that no static schedule and channel state can admit.
    pub(crate) blocked_node_ids: Vec<String>,
}

/// One deterministic readiness admission wave.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct StaticReadinessWave {
    /// Nodes admitted in this wave.
    pub(crate) node_ids: Vec<String>,
    /// Explanatory scheduler event that made the wave available.
    pub(crate) trigger: String,
}

/// Analyze the scheduler's OR-triggered static edges together with AND channel gates.
pub(crate) fn analyze_static_readiness(
    graph: &GraphRecord,
    scheduler: &Scheduler,
) -> Result<StaticReadinessAnalysis, CountValidationError> {
    for node in graph.nodes.values() {
        for requirement in node.input_requirements() {
            requirement.count.validate()?;
        }
    }
    let node_order = normalized_node_order(graph);
    let writers = channel_writer_counts(graph);
    let mut scheduled = BTreeSet::new();
    let mut admitted = BTreeSet::new();
    let mut completed_channels = BTreeMap::<String, usize>::new();
    let mut triggers = BTreeMap::<String, TriggerSources>::new();
    let mut waves = Vec::new();
    let mut prior_wave_node_ids = Vec::new();

    for node_id in scheduler.entry_nodes() {
        if graph.nodes.contains_key(node_id) {
            scheduled.insert(node_id.to_string());
            triggers.entry(node_id.to_string()).or_default().has_start = true;
        }
    }

    for _ in 0..graph.nodes.len() {
        let node_ids = node_order
            .iter()
            .filter(|node_id| scheduled.contains(node_id.as_str()))
            .filter(|node_id| !admitted.contains(node_id.as_str()))
            .filter(|node_id| {
                graph
                    .nodes
                    .get(node_id.as_str())
                    .is_some_and(|node| channels_ready(node, &completed_channels, &writers))
            })
            .cloned()
            .collect::<Vec<_>>();
        if node_ids.is_empty() {
            break;
        }

        let trigger = wave_trigger(
            graph,
            &node_ids,
            &triggers,
            &prior_wave_node_ids,
            &completed_channels,
            &writers,
        );
        for node_id in &node_ids {
            admitted.insert(node_id.clone());
            if let Some(node) = graph.nodes.get(node_id) {
                *completed_channels
                    .entry(node.output().to_string())
                    .or_default() += 1;
            }
        }
        for node_id in &node_ids {
            for successor in scheduler.start_anchored_successors(node_id) {
                schedule_successor(
                    graph,
                    &mut scheduled,
                    &mut triggers,
                    successor,
                    Trigger::Dispatched(node_id),
                );
            }
            for successor in scheduler.first_token_anchored_successors(node_id) {
                schedule_successor(
                    graph,
                    &mut scheduled,
                    &mut triggers,
                    successor,
                    Trigger::FirstToken(node_id),
                );
            }
            for successor in scheduler.successors_after(node_id) {
                schedule_successor(
                    graph,
                    &mut scheduled,
                    &mut triggers,
                    successor,
                    Trigger::Completed(node_id),
                );
            }
        }
        prior_wave_node_ids = node_ids.clone();
        waves.push(StaticReadinessWave { node_ids, trigger });
    }

    let blocked_node_ids = node_order
        .into_iter()
        .filter(|node_id| !admitted.contains(node_id))
        .collect();
    Ok(StaticReadinessAnalysis {
        waves,
        blocked_node_ids,
    })
}

pub(crate) fn normalized_node_order(graph: &GraphRecord) -> Vec<String> {
    let mut successors: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for edge in &graph.edges {
        successors
            .entry(edge.source.as_str())
            .or_default()
            .push(edge.target.as_str());
    }
    let mut queue = VecDeque::new();
    if let Some(entries) = successors.get(START_NODE_ID) {
        queue.extend(entries.iter().copied());
    }
    let mut seen = BTreeSet::new();
    let mut nodes = Vec::new();
    while let Some(node) = queue.pop_front() {
        if !graph.nodes.contains_key(node) || !seen.insert(node) {
            continue;
        }
        nodes.push(node.to_string());
        if let Some(next) = successors.get(node) {
            queue.extend(next.iter().copied());
        }
    }
    for node in graph.nodes.keys() {
        if seen.insert(node.as_str()) {
            nodes.push(node.clone());
        }
    }
    nodes
}

fn channel_writer_counts(graph: &GraphRecord) -> BTreeMap<String, usize> {
    let mut writers = BTreeMap::new();
    for node in graph.nodes.values() {
        *writers.entry(node.output().to_string()).or_default() += 1;
    }
    writers
}

fn channels_ready(
    node: &ExecutableGraphNode,
    completed_channels: &BTreeMap<String, usize>,
    writers: &BTreeMap<String, usize>,
) -> bool {
    node.input_requirements().iter().all(|requirement| {
        let Some(needed) = required_count(&requirement.count, &requirement.channel, writers) else {
            return false;
        };
        completed_channels
            .get(&requirement.channel)
            .copied()
            .unwrap_or(0)
            >= needed
    })
}

fn required_count(
    count: &Count,
    channel: &str,
    writers: &BTreeMap<String, usize>,
) -> Option<usize> {
    match count.validate() {
        Ok(Some(count)) => Some(count),
        Ok(None) => Some(writers.get(channel).copied().unwrap_or(0)),
        Err(_) => None,
    }
}

fn schedule_successor(
    graph: &GraphRecord,
    scheduled: &mut BTreeSet<String>,
    triggers: &mut BTreeMap<String, TriggerSources>,
    successor: &str,
    trigger: Trigger<'_>,
) {
    if !graph.nodes.contains_key(successor) {
        return;
    }
    scheduled.insert(successor.to_string());
    let sources = triggers.entry(successor.to_string()).or_default();
    match trigger {
        Trigger::Dispatched(source) => {
            append_source(&mut sources.dispatched, source);
        }
        Trigger::FirstToken(source) => {
            append_source(&mut sources.first_token, source);
        }
        Trigger::Completed(source) => {
            append_source(&mut sources.completed, source);
        }
    }
}

fn append_source(sources: &mut Vec<String>, source: &str) {
    if !sources.iter().any(|candidate| candidate == source) {
        sources.push(source.to_string());
    }
}

#[derive(Default)]
struct TriggerSources {
    has_start: bool,
    dispatched: Vec<String>,
    first_token: Vec<String>,
    completed: Vec<String>,
}

enum Trigger<'a> {
    Dispatched(&'a str),
    FirstToken(&'a str),
    Completed(&'a str),
}

fn wave_trigger(
    graph: &GraphRecord,
    node_ids: &[String],
    triggers: &BTreeMap<String, TriggerSources>,
    prior_wave_node_ids: &[String],
    completed_channels: &BTreeMap<String, usize>,
    writers: &BTreeMap<String, usize>,
) -> String {
    let mut completed = Vec::new();
    let mut first_token = Vec::new();
    let mut dispatched = Vec::new();
    let mut has_start = false;
    for node_id in node_ids {
        let Some(sources) = triggers.get(node_id) else {
            continue;
        };
        has_start |= sources.has_start;
        for source in &sources.completed {
            append_source(&mut completed, source);
        }
        for source in &sources.first_token {
            append_source(&mut first_token, source);
        }
        for source in &sources.dispatched {
            append_source(&mut dispatched, source);
        }
    }
    if has_channel_gate_unlocked_by_prior_wave(
        graph,
        node_ids,
        prior_wave_node_ids,
        completed_channels,
        writers,
    ) {
        return format!("completed: {}", prior_wave_node_ids.join(","));
    }
    if !completed.is_empty() {
        return format!("completed: {}", completed.join(","));
    }
    if !first_token.is_empty() {
        return format!("first-token: {}", first_token.join(","));
    }
    if !dispatched.is_empty() {
        return format!("dispatched: {}", dispatched.join(","));
    }
    if has_start {
        return "START".to_string();
    }
    "completed: none".to_string()
}

fn has_channel_gate_unlocked_by_prior_wave(
    graph: &GraphRecord,
    node_ids: &[String],
    prior_wave_node_ids: &[String],
    completed_channels: &BTreeMap<String, usize>,
    writers: &BTreeMap<String, usize>,
) -> bool {
    node_ids.iter().any(|node_id| {
        let Some(node) = graph.nodes.get(node_id) else {
            return false;
        };
        node.input_requirements().iter().any(|requirement| {
            let Some(needed) = required_count(&requirement.count, &requirement.channel, writers)
            else {
                return false;
            };
            let completed = completed_channels
                .get(&requirement.channel)
                .copied()
                .unwrap_or(0);
            let contributed_by_prior_wave = prior_wave_node_ids
                .iter()
                .filter(|source| {
                    graph
                        .nodes
                        .get(source.as_str())
                        .is_some_and(|source_node| source_node.output() == requirement.channel)
                })
                .count();
            completed >= needed && completed.saturating_sub(contributed_by_prior_wave) < needed
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph(source: &str) -> GraphRecord {
        serde_json::from_str(source).expect("valid graph")
    }

    #[test]
    fn static_channel_readiness_reports_scheduled_but_channel_blocked_nodes() {
        let graph = graph(
            r#"{
                "state":{"produced":{},"done":{}},
                "nodes":{
                    "reader":{"output":"done","inputs":[{"channel":"produced","count":1}]},
                    "producer":{"output":"produced"}
                },
                "edges":[
                    {"source":"START","target":"reader"},
                    {"source":"reader","target":"producer"}
                ]
            }"#,
        );
        let scheduler = Scheduler::new(&graph).expect("constructible scheduler");

        assert_eq!(
            analyze_static_readiness(&graph, &scheduler)
                .expect("valid channel counts")
                .blocked_node_ids,
            vec!["reader", "producer"]
        );
    }

    #[test]
    fn static_channel_readiness_keeps_dispatch_successors_scheduled_until_channels_complete() {
        let graph = graph(
            r#"{
                "state":{"produced":{},"done":{}},
                "nodes":{
                    "producer":{"output":"produced"},
                    "reader":{"output":"done","inputs":[{"channel":"produced","count":1}]}
                },
                "edges":[
                    {"source":"START","target":"producer"},
                    {"source":"producer","target":"reader","delay_after_predecessor_start_us":0.0}
                ]
            }"#,
        );
        let scheduler = Scheduler::new(&graph).expect("constructible scheduler");

        assert_eq!(
            analyze_static_readiness(&graph, &scheduler)
                .expect("valid channel counts")
                .waves,
            vec![
                StaticReadinessWave {
                    node_ids: vec!["producer".to_string()],
                    trigger: "START".to_string(),
                },
                StaticReadinessWave {
                    node_ids: vec!["reader".to_string()],
                    trigger: "completed: producer".to_string(),
                },
            ]
        );
    }

    #[test]
    fn first_token_successor_is_not_reported_as_static_readiness_blocked() {
        let graph = graph(
            r#"{
                "state":{"parent_output":{},"child_output":{}},
                "nodes":{
                    "parent":{"output":"parent_output"},
                    "child":{"output":"child_output"}
                },
                "edges":[
                    {"source":"START","target":"parent"},
                    {"source":"parent","target":"child","delay_after_predecessor_first_token_us":0.0}
                ]
            }"#,
        );
        let scheduler = Scheduler::new(&graph).expect("constructible scheduler");

        let analysis = analyze_static_readiness(&graph, &scheduler).expect("valid channel counts");
        assert!(analysis.blocked_node_ids.is_empty());
        assert_eq!(analysis.waves[1].trigger, "first-token: parent");
    }

    #[test]
    fn combined_dispatch_and_first_token_edges_remain_static_readiness_reachable() {
        let graph = graph(
            r#"{
                "state":{"parent_output":{},"child_output":{}},
                "nodes":{
                    "parent":{"output":"parent_output"},
                    "child":{"output":"child_output"}
                },
                "edges":[
                    {"source":"START","target":"parent"},
                    {"source":"parent","target":"child","delay_after_predecessor_start_us":0.0,"delay_after_predecessor_first_token_us":0.0}
                ]
            }"#,
        );
        let scheduler = Scheduler::new(&graph).expect("constructible scheduler");

        assert!(
            analyze_static_readiness(&graph, &scheduler)
                .expect("valid channel counts")
                .blocked_node_ids
                .is_empty()
        );
    }
}
