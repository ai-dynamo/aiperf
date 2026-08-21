// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Authored-prompt compilation and replay-fold emission into the flat Graph-IR.
//!
//! Two steps run here. [`compile_prompts`] translates every authored LLM node's
//! prompt grammar into `PromptItem`s over one shared `SegmentPool`: a `@channel`
//! reference becomes a `Splice`, and a `{role, content}` message is interned as a
//! dense message segment. [`fold_replay_and_emit`] then lowers one resolved,
//! pruned trace: each surviving replay node's recorded `outputs` are pre-seeded
//! into `initial_state`, the node is dropped, and its recorded latency moves onto
//! the successor edge — leaving a flat `LlmNode`/`StaticEdge` `GraphRecord` that
//! the runtime executes unchanged.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};

use serde_json::Value;

use crate::dataset::{SegmentPool, TextTokenizer};
use crate::graph::model::{
    ChannelSpec, ExecutableGraphNode, GraphRecord, LlmNode, PromptItem, START_NODE_ID, StaticEdge,
};
use crate::graph::segment::intern_message;
use crate::graph::timing::{checked_add_microseconds, milliseconds_to_microseconds};
use crate::graph::validate::{validate, validate_detailed};
use crate::graph::wire::OpenAiChatMessage;

use super::model::{
    AuthoredChannelSpec, AuthoredGraph, AuthoredLlmNode, AuthoredNode, AuthoredTrace,
    ConditionalError, MessagePart, PromptGrammarItem,
};
use super::resolve::{TakenEdge, TakenGraph};

/// Compiled prompt programs keyed by authored node id.
#[derive(Debug, Clone)]
pub struct CompiledPrompts {
    pub per_node: BTreeMap<String, Vec<PromptItem>>,
}

/// Compile every authored LLM node's prompt grammar into `PromptItem`s.
///
/// One shared pool interns all message segments so identical prompts across
/// nodes and traces deduplicate under the store's content identity.
pub fn compile_prompts(
    graph: &AuthoredGraph,
    pool: &mut SegmentPool,
    tokenizer: &dyn TextTokenizer,
) -> Result<CompiledPrompts, ConditionalError> {
    let mut per_node = BTreeMap::new();
    for (id, node) in &graph.nodes {
        if let AuthoredNode::Llm(llm) = node {
            let items = compile_prompt_items(id, llm, pool, tokenizer)?;
            per_node.insert(id.clone(), items);
        }
    }
    Ok(CompiledPrompts { per_node })
}

fn compile_prompt_items(
    id: &str,
    llm: &AuthoredLlmNode,
    pool: &mut SegmentPool,
    tokenizer: &dyn TextTokenizer,
) -> Result<Vec<PromptItem>, ConditionalError> {
    let mut items = Vec::new();
    for grammar in &llm.prompt {
        match grammar {
            PromptGrammarItem::ChannelRef(channel) => {
                items.push(PromptItem::Splice {
                    splice: channel.clone(),
                });
            }
            PromptGrammarItem::Text(text) => {
                items.push(message_item(id, "user", text, pool, tokenizer)?);
            }
            PromptGrammarItem::Message { role, content } => {
                // A message's literal parts form one message object; each
                // `@channel` part becomes a splice placed after it, so a message
                // assembled from channel content still carries its recorded
                // upstream values into the prompt.
                let mut literal = String::new();
                let mut refs = Vec::new();
                for part in content {
                    match part {
                        MessagePart::Text(text) => {
                            if !literal.is_empty() {
                                literal.push(' ');
                            }
                            literal.push_str(text);
                        }
                        MessagePart::ChannelRef(channel) => refs.push(channel.clone()),
                    }
                }
                items.push(message_item(id, role, &literal, pool, tokenizer)?);
                for channel in refs {
                    items.push(PromptItem::Splice { splice: channel });
                }
            }
        }
    }
    Ok(items)
}

fn message_item(
    id: &str,
    role: &str,
    content: &str,
    pool: &mut SegmentPool,
    tokenizer: &dyn TextTokenizer,
) -> Result<PromptItem, ConditionalError> {
    let message = OpenAiChatMessage::new(role, content);
    let handle = intern_message(pool, &message, None, tokenizer).map_err(|error| {
        ConditionalError::message(format!("node {id:?} prompt segment: {error}"))
    })?;
    Ok(PromptItem::Seg { seg: handle })
}

/// The flat graph and seeded channel state for one resolved trace.
#[derive(Debug, Clone)]
pub struct FoldedTrace {
    pub graph: GraphRecord,
    pub initial_state: BTreeMap<String, Value>,
}

/// Fold recorded replay nodes into `initial_state` and emit a flat `GraphRecord`.
pub fn fold_replay_and_emit(
    taken: &TakenGraph,
    trace: &AuthoredTrace,
    state: &BTreeMap<String, AuthoredChannelSpec>,
    prompts: &CompiledPrompts,
) -> Result<FoldedTrace, ConditionalError> {
    let mut replay_ids: BTreeSet<String> = BTreeSet::new();
    let mut replay_duration_ms: BTreeMap<String, f64> = BTreeMap::new();
    let mut initial_state = trace.initial_state.clone();
    let mut nodes: BTreeMap<String, ExecutableGraphNode> = BTreeMap::new();

    for (id, node) in &taken.nodes {
        match node {
            AuthoredNode::Replay(replay) => {
                replay_ids.insert(id.clone());
                replay_duration_ms.insert(id.clone(), replay.duration_ms);
                let recorded = trace.replay_outputs.get(id).ok_or_else(|| {
                    ConditionalError::message(format!(
                        "trace {:?} is missing replay_outputs for fired replay node {id:?}",
                        trace.id
                    ))
                })?;
                for channel in &replay.outputs {
                    if !state.contains_key(channel) {
                        return Err(ConditionalError::message(format!(
                            "replay node {id:?} writes undeclared channel {channel:?}"
                        )));
                    }
                    let value = recorded.get(channel).ok_or_else(|| {
                        ConditionalError::message(format!(
                            "trace {:?} replay_outputs[{id:?}] is missing channel {channel:?}",
                            trace.id
                        ))
                    })?;
                    initial_state.insert(channel.clone(), value.clone());
                }
            }
            AuthoredNode::Llm(llm) => {
                let items = prompts.per_node.get(id).cloned().unwrap_or_default();
                let mut metadata = llm.metadata.clone();
                if llm.terminal_for_user {
                    metadata.insert("terminal_for_user".to_string(), Value::Bool(true));
                }
                nodes.insert(
                    id.clone(),
                    ExecutableGraphNode::Llm(LlmNode {
                        output: llm.output.clone(),
                        streaming: llm.streaming,
                        inputs: llm.inputs.clone(),
                        min_start_delay_us: llm.min_start_delay_us,
                        max_tokens: llm.max_tokens,
                        items,
                        request: None,
                        metadata,
                    }),
                );
            }
        }
    }

    if let Some((id, _)) = taken.nodes.iter().find(|(_, node)| {
        matches!(node, AuthoredNode::Replay(replay) if replay.min_start_delay_us.is_some())
    }) {
        return Err(ConditionalError::message(format!(
            "cannot fold replay node {id:?} with min_start_delay_us"
        )));
    }

    let edges = emit_edges(taken, &replay_ids, &replay_duration_ms)?;

    let state_map = state
        .iter()
        .map(|(name, spec)| {
            (
                name.clone(),
                ChannelSpec {
                    channel_type: spec.channel_type,
                    reducer: spec.reducer,
                },
            )
        })
        .collect();

    let graph = GraphRecord {
        version: None,
        system: None,
        state: state_map,
        nodes,
        edges,
    };

    let validation = validate(&graph);
    // Keep this tooling diagnostic in the retained Graph-IR so `graph validate`
    // can return its structured finding; `run_trace` validates again before any
    // executor state is created.
    let retain_for_inspection = validation.len() == 1
        && validate_detailed(&graph).iter().any(|issue| {
            matches!(
                issue.code.as_str(),
                "graph-cycle" | "static-channel-readiness-deadlock"
            )
        });
    if !validation.is_empty() && !retain_for_inspection {
        let detail = validation
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("; ");
        return Err(ConditionalError::message(format!(
            "trace {:?} lowered to an invalid graph: {detail}",
            trace.id
        )));
    }

    // Every seeded channel must be a declared state channel or the per-trace
    // store rejects it at construction.
    for channel in initial_state.keys() {
        if !state.contains_key(channel) {
            return Err(ConditionalError::message(format!(
                "trace {:?} seeds undeclared channel {channel:?}",
                trace.id
            )));
        }
    }

    Ok(FoldedTrace {
        graph,
        initial_state,
    })
}

/// Preserve direct edges and replace replay paths with their maximum composed
/// completion delay for each reachable non-replay target.
fn emit_edges(
    taken: &TakenGraph,
    replay_ids: &BTreeSet<String>,
    replay_duration_ms: &BTreeMap<String, f64>,
) -> Result<Vec<StaticEdge>, ConditionalError> {
    let mut out_adjacency: HashMap<&str, Vec<&TakenEdge>> = HashMap::new();
    for edge in &taken.edges {
        out_adjacency
            .entry(edge.source.as_str())
            .or_default()
            .push(edge);
    }
    let replay_order = replay_topological_order(&out_adjacency, replay_ids)?;
    let replay_tails = replay_tail_delays(
        &out_adjacency,
        replay_ids,
        replay_duration_ms,
        &replay_order,
    )?;

    let mut folded_delays = BTreeMap::<(String, String), f64>::new();
    for edge in &taken.edges {
        if replay_ids.contains(&edge.source) || !replay_ids.contains(&edge.target) {
            continue;
        }
        reject_unfoldable_replay_edge_timing(edge)?;
        let Some(tails) = replay_tails.get(&edge.target) else {
            return Err(ConditionalError::message(format!(
                "replay node {:?} has no folded timing state",
                edge.target
            )));
        };
        for (target, tail_delay) in tails {
            let total =
                add_replay_delay(edge.delay_after_predecessor_us.unwrap_or(0.0), *tail_delay)?;
            if edge.source == START_NODE_ID && total != 0.0 {
                return Err(ConditionalError::message(
                    "cannot fold nonzero replay completion delay from START",
                ));
            }
            folded_delays
                .entry((edge.source.clone(), target.clone()))
                .and_modify(|current| *current = current.max(total))
                .or_insert(total);
        }
    }

    let mut result = Vec::new();
    let mut emitted_folded = BTreeSet::new();
    for edge in &taken.edges {
        if replay_ids.contains(&edge.source) {
            continue;
        }
        if !replay_ids.contains(&edge.target) {
            result.push(StaticEdge {
                source: edge.source.clone(),
                target: edge.target.clone(),
                delay_after_predecessor_us: edge.delay_after_predecessor_us,
                min_start_delay_us: edge.min_start_delay_us,
                delay_after_predecessor_start_us: edge.delay_after_predecessor_start_us,
                delay_after_predecessor_first_token_us: edge.delay_after_predecessor_first_token_us,
            });
            continue;
        }
        let Some(tails) = replay_tails.get(&edge.target) else {
            continue;
        };
        for target in tails.keys() {
            let key = (edge.source.clone(), target.clone());
            if !emitted_folded.insert(key.clone()) {
                continue;
            }
            let Some(delay) = folded_delays.get(&key).copied() else {
                continue;
            };
            result.push(StaticEdge {
                source: key.0,
                target: key.1,
                delay_after_predecessor_us: Some(delay),
                min_start_delay_us: None,
                delay_after_predecessor_start_us: None,
                delay_after_predecessor_first_token_us: None,
            });
        }
    }
    Ok(result)
}

fn replay_topological_order(
    out_adjacency: &HashMap<&str, Vec<&TakenEdge>>,
    replay_ids: &BTreeSet<String>,
) -> Result<Vec<String>, ConditionalError> {
    let replay_nodes = replay_ids.iter().map(String::as_str).collect::<Vec<_>>();
    let node_index = replay_nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (*node, index))
        .collect::<HashMap<_, _>>();
    let mut replay_adjacency = vec![Vec::<usize>::new(); replay_nodes.len()];
    let mut indegree = vec![0_usize; replay_nodes.len()];
    for (source_index, source) in replay_nodes.iter().enumerate() {
        for edge in out_adjacency.get(source).into_iter().flatten() {
            if let Some(&target_index) = node_index.get(edge.target.as_str()) {
                replay_adjacency[source_index].push(target_index);
                indegree[target_index] += 1;
            }
        }
    }
    let mut ready = indegree
        .iter()
        .enumerate()
        .filter_map(|(index, degree)| (*degree == 0).then_some(index))
        .collect::<VecDeque<_>>();
    let mut order = Vec::with_capacity(replay_nodes.len());
    while let Some(node_index) = ready.pop_front() {
        for &target_index in &replay_adjacency[node_index] {
            indegree[target_index] -= 1;
            if indegree[target_index] == 0 {
                ready.push_back(target_index);
            }
        }
        order.push(replay_nodes[node_index].to_owned());
    }
    if order.len() == replay_nodes.len() {
        return Ok(order);
    }
    let cyclic = replay_cycle_witness(&replay_nodes, &replay_adjacency, &indegree).join(", ");
    Err(ConditionalError::message(format!(
        "cannot fold replay cycle through [{cyclic}]"
    )))
}

fn replay_cycle_witness(
    replay_nodes: &[&str],
    adjacency: &[Vec<usize>],
    residual_indegree: &[usize],
) -> Vec<String> {
    let mut state = vec![0_u8; replay_nodes.len()];
    let mut position = vec![usize::MAX; replay_nodes.len()];
    for start in 0..replay_nodes.len() {
        if residual_indegree[start] == 0 || state[start] != 0 {
            continue;
        }
        state[start] = 1;
        position[start] = 0;
        let mut stack = vec![(start, 0_usize)];
        while !stack.is_empty() {
            let frame_index = stack.len() - 1;
            let node = stack[frame_index].0;
            let next_edge = stack[frame_index].1;
            if next_edge == adjacency[node].len() {
                stack.pop();
                position[node] = usize::MAX;
                state[node] = 2;
                continue;
            }
            stack[frame_index].1 += 1;
            let target = adjacency[node][next_edge];
            if residual_indegree[target] == 0 {
                continue;
            }
            match state[target] {
                0 => {
                    state[target] = 1;
                    position[target] = stack.len();
                    stack.push((target, 0));
                }
                1 => {
                    return stack[position[target]..]
                        .iter()
                        .map(|(index, _)| replay_nodes[*index].to_owned())
                        .collect();
                }
                _ => {}
            }
        }
    }
    Vec::new()
}

fn replay_tail_delays(
    out_adjacency: &HashMap<&str, Vec<&TakenEdge>>,
    replay_ids: &BTreeSet<String>,
    replay_duration_ms: &BTreeMap<String, f64>,
    replay_order: &[String],
) -> Result<BTreeMap<String, BTreeMap<String, f64>>, ConditionalError> {
    let mut tails = BTreeMap::<String, BTreeMap<String, f64>>::new();
    for node in replay_order.iter().rev() {
        let duration =
            milliseconds_to_microseconds(replay_duration_ms.get(node).copied().unwrap_or(0.0))
                .map_err(|_| {
                    ConditionalError::message(format!(
                        "duration_ms must fit i64 nanoseconds on replay node {node:?}"
                    ))
                })?;
        let mut node_tails = BTreeMap::<String, f64>::new();
        for edge in out_adjacency.get(node.as_str()).into_iter().flatten() {
            reject_unfoldable_replay_edge_timing(edge)?;
            let edge_delay = edge.delay_after_predecessor_us.unwrap_or(0.0);
            let prefix_delay = add_replay_delay(duration, edge_delay)?;
            if replay_ids.contains(&edge.target) {
                let Some(child_tails) = tails.get(&edge.target) else {
                    return Err(ConditionalError::message(format!(
                        "replay successor {:?} has no folded timing state",
                        edge.target
                    )));
                };
                for (target, child_delay) in child_tails {
                    let total = add_replay_delay(prefix_delay, *child_delay)?;
                    node_tails
                        .entry(target.clone())
                        .and_modify(|current| *current = current.max(total))
                        .or_insert(total);
                }
            } else {
                node_tails
                    .entry(edge.target.clone())
                    .and_modify(|current| *current = current.max(prefix_delay))
                    .or_insert(prefix_delay);
            }
        }
        tails.insert(node.clone(), node_tails);
    }
    Ok(tails)
}

fn add_replay_delay(left: f64, right: f64) -> Result<f64, ConditionalError> {
    checked_add_microseconds(left, right).map_err(|_| {
        ConditionalError::message("folded replay completion delay must fit i64 nanoseconds")
    })
}

fn reject_unfoldable_replay_edge_timing(edge: &TakenEdge) -> Result<(), ConditionalError> {
    for (field, value) in [
        ("min_start_delay_us", edge.min_start_delay_us),
        (
            "delay_after_predecessor_start_us",
            edge.delay_after_predecessor_start_us,
        ),
        (
            "delay_after_predecessor_first_token_us",
            edge.delay_after_predecessor_first_token_us,
        ),
    ] {
        if value.is_some() {
            return Err(ConditionalError::message(format!(
                "cannot fold replay path with {field} on edge {:?} -> {:?}",
                edge.source, edge.target
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::model::parse_authored_graph;
    use super::super::resolve::resolve_and_prune;
    use super::*;
    use crate::dataset::TiktokenTokenizer;

    const DOC: &str = r#"
graph:
  state:
    messages:    {type: messages, reducer: add_messages}
    intent:      {type: text}
    plan:        {type: text}
    raw_results: {type: json}
    brand_cands: {type: text}
    brand_mapping: {type: text}
    preprocessed: {type: text}
    summary:     {type: text}
  nodes:
    route:
      node_type: llm
      prompt:
        - {role: system, content: "Classify intent."}
        - "@messages"
      output: intent
    plan:
      node_type: llm
      prompt: [{role: system, content: "Plan tool calls."}]
      output: plan
    tool_exec:
      node_type: replay
      outputs: [raw_results, brand_cands]
      duration_ms: 80
    brandmap:
      node_type: llm
      prompt: [{role: user, content: ["@brand_cands"]}]
      output: brand_mapping
    preprocess:
      node_type: replay
      outputs: [preprocessed]
      duration_ms: 5
    summarize:
      node_type: llm
      prompt: [{role: user, content: ["@preprocessed"]}]
      output: summary
      terminal_for_user: true
  edges:
    - {source: START, target: route}
    - {source: route, branches: {shopping: plan, non_shopping: END}}
    - {source: plan, target: tool_exec}
    - {source: tool_exec, target: brandmap}
    - {source: brandmap, target: preprocess}
    - {source: preprocess, target: summarize}
traces:
  - id: t-shop
    selected_branches: {route: shopping}
    initial_state: {messages: []}
    replay_outputs:
      tool_exec: {raw_results: [], brand_cands: "Nike, Adidas"}
      preprocess: {preprocessed: "Top hits"}
"#;

    fn fold_shopping() -> FoldedTrace {
        let doc = parse_authored_graph(DOC.as_bytes()).unwrap();
        let tokenizer = TiktokenTokenizer::builtin();
        let mut pool = SegmentPool::new();
        let prompts = compile_prompts(&doc.graph, &mut pool, &tokenizer).unwrap();
        let taken = resolve_and_prune(&doc.graph, &doc.traces[0], 0).unwrap();
        fold_replay_and_emit(&taken, &doc.traces[0], &doc.graph.state, &prompts).unwrap()
    }

    fn fold_source(source: &str) -> std::result::Result<FoldedTrace, ConditionalError> {
        let doc = parse_authored_graph(source.as_bytes())?;
        let tokenizer = TiktokenTokenizer::builtin();
        let mut pool = SegmentPool::new();
        let prompts = compile_prompts(&doc.graph, &mut pool, &tokenizer)?;
        let taken = resolve_and_prune(&doc.graph, &doc.traces[0], 0)?;
        fold_replay_and_emit(&taken, &doc.traces[0], &doc.graph.state, &prompts)
    }

    #[test]
    fn channel_ref_becomes_splice_and_message_interns_segment() {
        let doc = parse_authored_graph(DOC.as_bytes()).unwrap();
        let tokenizer = TiktokenTokenizer::builtin();
        let mut pool = SegmentPool::new();
        let prompts = compile_prompts(&doc.graph, &mut pool, &tokenizer).unwrap();
        let route = &prompts.per_node["route"];
        assert!(matches!(route[0], PromptItem::Seg { .. }));
        assert!(matches!(&route[1], PromptItem::Splice { splice } if splice == "messages"));
    }

    #[test]
    fn replay_outputs_seed_initial_state_and_drop_node() {
        let folded = fold_shopping();
        assert!(folded.initial_state.contains_key("raw_results"));
        assert!(folded.initial_state.contains_key("brand_cands"));
        assert!(folded.initial_state.contains_key("preprocessed"));
        // Replay nodes are folded out; only LLM nodes remain.
        assert!(!folded.graph.nodes.contains_key("tool_exec"));
        assert!(!folded.graph.nodes.contains_key("preprocess"));
        let mut fired: Vec<&String> = folded.graph.nodes.keys().collect();
        fired.sort();
        assert_eq!(fired, vec!["brandmap", "plan", "route", "summarize"]);
    }

    #[test]
    fn replay_duration_moves_to_successor_edge() {
        let folded = fold_shopping();
        let plan_to_brandmap = folded
            .graph
            .edges
            .iter()
            .find(|e| e.source == "plan" && e.target == "brandmap")
            .expect("plan should connect straight to brandmap across tool_exec");
        assert!(plan_to_brandmap.delay_after_predecessor_us.unwrap() >= 80_000.0);
        let brandmap_to_summarize = folded
            .graph
            .edges
            .iter()
            .find(|e| e.source == "brandmap" && e.target == "summarize")
            .expect("brandmap should connect straight to summarize across preprocess");
        assert!(brandmap_to_summarize.delay_after_predecessor_us.unwrap() >= 5_000.0);
    }

    #[test]
    fn emitted_graph_is_llm_only_and_valid() {
        let folded = fold_shopping();
        assert!(validate(&folded.graph).is_empty());
        // summarize kept its terminal marker in metadata.
        assert_eq!(
            folded.graph.nodes["summarize"]
                .as_llm()
                .unwrap()
                .metadata
                .get("terminal_for_user"),
            Some(&Value::Bool(true))
        );
    }

    #[test]
    fn replay_fold_preserves_nonpositive_completion_delay() {
        let folded = fold_source(
            r#"graph:
  state: {out: {}}
  nodes:
    source: {prompt: [x], output: out}
    replay: {node_type: replay, outputs: [out], duration_ms: 0}
    target: {prompt: [x], output: out}
  edges:
    - {source: START, target: source}
    - {source: source, target: replay, delay_after_predecessor_us: -1}
    - {source: replay, target: target}
traces:
  - id: trace
    replay_outputs: {replay: {out: replayed}}
"#,
        )
        .expect("folded replay graph");
        let edge = folded
            .graph
            .edges
            .iter()
            .find(|edge| edge.source == "source" && edge.target == "target")
            .expect("folded edge");
        assert_eq!(edge.delay_after_predecessor_us, Some(-1.0));
    }

    #[test]
    fn replay_fold_refuses_unrepresentable_timing_and_cycles() {
        for timing in [
            "min_start_delay_us: 1",
            "delay_after_predecessor_start_us: 1",
            "delay_after_predecessor_first_token_us: 1",
        ] {
            let source = format!(
                "graph:\n  state: {{out: {{}}}}\n  nodes:\n    source: {{prompt: [x], output: out}}\n    replay: {{node_type: replay, outputs: [out]}}\n    target: {{prompt: [x], output: out}}\n  edges:\n    - {{source: START, target: source}}\n    - {{source: source, target: replay, {timing}}}\n    - {{source: replay, target: target}}\ntraces:\n  - id: trace\n    replay_outputs: {{replay: {{out: replayed}}}}\n"
            );
            let error = fold_source(&source).expect_err("unrepresentable replay timing");
            assert!(error.to_string().contains("cannot fold replay path"));
        }

        let error = fold_source(
            r#"graph:
  state: {out: {}}
  nodes:
    source: {prompt: [x], output: out}
    r1: {node_type: replay, outputs: [out]}
    r2: {node_type: replay, outputs: [out]}
    tail: {node_type: replay, outputs: [out]}
    target: {prompt: [x], output: out}
  edges:
    - {source: START, target: source}
    - {source: source, target: r1}
    - {source: r1, target: r2}
    - {source: r2, target: r1}
    - {source: r2, target: tail}
    - {source: tail, target: target}
traces:
  - id: trace
    replay_outputs: {r1: {out: one}, r2: {out: two}, tail: {out: tail}}
"#,
        )
        .expect_err("replay cycle");
        assert_eq!(
            error.to_string(),
            "cannot fold replay cycle through [r1, r2]"
        );
    }

    #[test]
    fn replay_fold_refuses_replay_node_min_start_delay() {
        let error = fold_source(
            r#"graph:
  state: {out: {}}
  nodes:
    source: {prompt: [x], output: out}
    replay: {node_type: replay, outputs: [out], min_start_delay_us: 1}
    target: {prompt: [x], output: out}
  edges:
    - {source: START, target: source}
    - {source: source, target: replay}
    - {source: replay, target: target}
traces:
  - id: trace
    replay_outputs: {replay: {out: replayed}}
"#,
        )
        .expect_err("replay node minimum-start delay");
        assert!(error.to_string().contains("cannot fold replay node"));
    }

    #[test]
    fn replay_fold_handles_a_long_acyclic_chain_without_path_copies() {
        const REPLAYS: usize = 256;
        let mut source = String::from(
            "graph:\n  state: {out: {}}\n  nodes:\n    source: {prompt: [x], output: out}\n    target: {prompt: [x], output: out}\n",
        );
        for index in 0..REPLAYS {
            source.push_str(&format!(
                "    r{index}: {{node_type: replay, outputs: [out]}}\n"
            ));
        }
        source.push_str("  edges:\n    - {source: START, target: source}\n");
        source.push_str("    - {source: source, target: r0}\n");
        for index in 0..REPLAYS - 1 {
            source.push_str(&format!(
                "    - {{source: r{index}, target: r{}}}\n",
                index + 1
            ));
        }
        source.push_str(&format!(
            "    - {{source: r{}, target: target}}\n",
            REPLAYS - 1
        ));
        source.push_str("traces:\n  - id: trace\n    replay_outputs:\n");
        for index in 0..REPLAYS {
            source.push_str(&format!("      r{index}: {{out: replayed}}\n"));
        }

        let folded = fold_source(&source).expect("long replay chain");
        assert!(
            folded
                .graph
                .edges
                .iter()
                .any(|edge| edge.source == "source" && edge.target == "target")
        );
    }

    #[test]
    fn replay_fold_uses_maximum_delay_across_reconvergent_paths() {
        let folded = fold_source(
            r#"graph:
  state: {out: {}}
  nodes:
    source: {prompt: [x], output: out}
    root: {node_type: replay, outputs: [out]}
    quick: {node_type: replay, outputs: [out], duration_ms: 0.005}
    slow: {node_type: replay, outputs: [out], duration_ms: 0.02}
    target: {prompt: [x], output: out}
  edges:
    - {source: START, target: source}
    - {source: source, target: root}
    - {source: root, target: slow}
    - {source: slow, target: target, delay_after_predecessor_us: 13}
    - {source: root, target: quick}
    - {source: quick, target: target}
traces:
  - id: trace
    replay_outputs: {root: {out: root}, quick: {out: quick}, slow: {out: slow}}
"#,
        )
        .expect("reconvergent replay DAG");
        let edge = folded
            .graph
            .edges
            .iter()
            .find(|edge| edge.source == "source" && edge.target == "target")
            .expect("collapsed edge");
        assert_eq!(edge.delay_after_predecessor_us, Some(33.0));
    }

    #[test]
    fn replay_fold_refuses_nonzero_start_completion_timing() {
        let error = fold_source(
            r#"graph:
  state: {out: {}}
  nodes:
    replay: {node_type: replay, outputs: [out], duration_ms: 1}
    target: {prompt: [x], output: out}
  edges:
    - {source: START, target: replay}
    - {source: replay, target: target}
traces:
  - id: trace
    replay_outputs: {replay: {out: replayed}}
"#,
        )
        .expect_err("START has no completion event");
        assert!(
            error
                .to_string()
                .contains("cannot fold nonzero replay completion delay from START")
        );
    }

    #[test]
    fn replay_fold_preserves_direct_edges_and_zero_start_delay() {
        let folded = fold_source(
            r#"graph:
  state: {out: {}}
  nodes:
    replay: {node_type: replay, outputs: [out]}
    direct: {prompt: [x], output: out}
    folded: {prompt: [x], output: out}
  edges:
    - {source: START, target: replay}
    - {source: replay, target: folded}
    - {source: START, target: direct}
    - {source: direct, target: END, delay_after_predecessor_us: -1, min_start_delay_us: 2, delay_after_predecessor_start_us: 3, delay_after_predecessor_first_token_us: 4}
traces:
  - id: trace
    replay_outputs: {replay: {out: replayed}}
"#,
        )
        .expect("zero-delay START replay path");

        let start_edge = folded
            .graph
            .edges
            .iter()
            .find(|edge| edge.source == START_NODE_ID && edge.target == "folded")
            .expect("folded START edge");
        assert_eq!(start_edge.delay_after_predecessor_us, Some(0.0));

        let direct_edge = folded
            .graph
            .edges
            .iter()
            .find(|edge| edge.source == "direct" && edge.target == "END")
            .expect("direct edge");
        assert_eq!(direct_edge.delay_after_predecessor_us, Some(-1.0));
        assert_eq!(direct_edge.min_start_delay_us, Some(2.0));
        assert_eq!(direct_edge.delay_after_predecessor_start_us, Some(3.0));
        assert_eq!(
            direct_edge.delay_after_predecessor_first_token_us,
            Some(4.0)
        );
    }

    #[test]
    fn replay_fold_refuses_a_composed_out_of_range_delay() {
        let error = fold_source(
            r#"graph:
  state: {out: {}}
  nodes:
    source: {prompt: [x], output: out}
    replay: {node_type: replay, outputs: [out], duration_ms: 5000000000000}
    target: {prompt: [x], output: out}
  edges:
    - {source: START, target: source}
    - {source: source, target: replay, delay_after_predecessor_us: 5000000000000000}
    - {source: replay, target: target}
traces:
  - id: trace
    replay_outputs: {replay: {out: replayed}}
"#,
        )
        .expect_err("composed delay exceeds signed nanosecond range");
        assert_eq!(
            error.to_string(),
            "folded replay completion delay must fit i64 nanoseconds"
        );
    }
}
