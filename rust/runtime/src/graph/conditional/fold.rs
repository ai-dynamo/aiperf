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

use std::collections::{BTreeMap, BTreeSet};

use serde_json::Value;

use crate::dataset::{SegmentPool, TextTokenizer};
use crate::graph::model::{ChannelSpec, GraphRecord, LlmNode, PromptItem, StaticEdge};
use crate::graph::segment::intern_message;
use crate::graph::validate::validate;
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
    let handle = intern_message(pool, &message, None, tokenizer)
        .map_err(|error| ConditionalError(format!("node {id:?} prompt segment: {error}")))?;
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
    let mut nodes: BTreeMap<String, LlmNode> = BTreeMap::new();

    for (id, node) in &taken.nodes {
        match node {
            AuthoredNode::Replay(replay) => {
                replay_ids.insert(id.clone());
                replay_duration_ms.insert(id.clone(), replay.duration_ms);
                let recorded = trace.replay_outputs.get(id).ok_or_else(|| {
                    ConditionalError(format!(
                        "trace {:?} is missing replay_outputs for fired replay node {id:?}",
                        trace.id
                    ))
                })?;
                for channel in &replay.outputs {
                    if !state.contains_key(channel) {
                        return Err(ConditionalError(format!(
                            "replay node {id:?} writes undeclared channel {channel:?}"
                        )));
                    }
                    let value = recorded.get(channel).ok_or_else(|| {
                        ConditionalError(format!(
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
                    LlmNode {
                        output: llm.output.clone(),
                        streaming: llm.streaming,
                        inputs: Vec::new(),
                        min_start_delay_us: llm.min_start_delay_us,
                        max_tokens: llm.max_tokens,
                        items,
                        metadata,
                    },
                );
            }
        }
    }

    let edges = emit_edges(taken, &replay_ids, &replay_duration_ms);

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
    if !validation.is_empty() {
        let detail = validation
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("; ");
        return Err(ConditionalError(format!(
            "trace {:?} lowered to an invalid graph: {detail}",
            trace.id
        )));
    }

    // Every seeded channel must be a declared state channel or the per-trace
    // store rejects it at construction.
    for channel in initial_state.keys() {
        if !state.contains_key(channel) {
            return Err(ConditionalError(format!(
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

/// Build the emitted static edges, skipping replay nodes by connecting each
/// non-replay source to the non-replay targets reachable through replay chains,
/// accumulating each skipped replay node's recorded latency onto the edge delay.
fn emit_edges(
    taken: &TakenGraph,
    replay_ids: &BTreeSet<String>,
    replay_duration_ms: &BTreeMap<String, f64>,
) -> Vec<StaticEdge> {
    let mut out_adjacency: BTreeMap<&str, Vec<&TakenEdge>> = BTreeMap::new();
    for edge in &taken.edges {
        out_adjacency
            .entry(edge.source.as_str())
            .or_default()
            .push(edge);
    }

    let delay_of = |edge: &TakenEdge| edge.delay_after_predecessor_us.unwrap_or(0.0);
    let duration_us = |node: &str| replay_duration_ms.get(node).copied().unwrap_or(0.0) * 1_000.0;

    let mut result: Vec<StaticEdge> = Vec::new();
    let mut seen: BTreeSet<(String, String)> = BTreeSet::new();
    for edge in &taken.edges {
        // Edges leaving a replay node are folded in via the DFS below; skip them
        // as independent edges.
        if replay_ids.contains(&edge.source) {
            continue;
        }
        if !replay_ids.contains(&edge.target) {
            // Direct non-replay -> non-replay/END edge: copy every anchor.
            if seen.insert((edge.source.clone(), edge.target.clone())) {
                result.push(StaticEdge {
                    source: edge.source.clone(),
                    target: edge.target.clone(),
                    delay_after_predecessor_us: edge.delay_after_predecessor_us,
                    min_start_delay_us: edge.min_start_delay_us,
                    delay_after_predecessor_start_us: edge.delay_after_predecessor_start_us,
                    delay_after_predecessor_first_token_us: edge
                        .delay_after_predecessor_first_token_us,
                });
            }
            continue;
        }
        // Fold: DFS through the replay chain rooted at edge.target, summing each
        // replay node's latency plus the intervening edge delays.
        let mut stack: Vec<(&str, f64)> = vec![(
            edge.target.as_str(),
            delay_of(edge) + duration_us(&edge.target),
        )];
        while let Some((node, accumulated)) = stack.pop() {
            for next in out_adjacency.get(node).into_iter().flatten() {
                if replay_ids.contains(&next.target) {
                    stack.push((
                        next.target.as_str(),
                        accumulated + delay_of(next) + duration_us(&next.target),
                    ));
                } else {
                    let total = accumulated + delay_of(next);
                    if seen.insert((edge.source.clone(), next.target.clone())) {
                        result.push(StaticEdge {
                            source: edge.source.clone(),
                            target: next.target.clone(),
                            delay_after_predecessor_us: (total > 0.0).then_some(total),
                            min_start_delay_us: None,
                            delay_after_predecessor_start_us: None,
                            delay_after_predecessor_first_token_us: None,
                        });
                    }
                }
            }
        }
    }
    result
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
                .metadata
                .get("terminal_for_user"),
            Some(&Value::Bool(true))
        );
    }
}
