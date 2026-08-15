// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native `aiperf.trace.v1` compiler.
//!
//! `aiperf.trace.v1` is a per-session, content-addressed recorded-trace format:
//! a deduplicating **segment pool** (`segments`, each a message with an explicit
//! `role` and its block `hash_ids`) plus **`inference_calls`** that reference
//! pooled segments by index and carry timing, usage, and the conversation graph
//! (`previous_ref`, `compaction`, hashed agent ids). One session = one trace;
//! each call = one leaf request. Unlike WEKA/Dynamo — which carry only opaque
//! block hashes and must *reconstruct* message boundaries heuristically — this
//! format supplies **ground-truth** message boundaries, injected straight into
//! the shared lowering via `RecordedRequest::explicit_tags`.

mod schema;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use crate::dataset::{SegmentPool, TextTokenizer};
use serde_json::Value;

use crate::graph::input::{GraphInputBundle, GraphInputMetadata};
use crate::graph::model::GraphTraceProgram;

use super::content::CorpusContentSynthesizer;
use super::source::load_aiperf_documents;
use super::trie::{BlockTag, RecordedRequest, graph_plan, lower_recorded_graph};
use super::{RecordedTraceError, RecordedTraceInputConfig};
use schema::{AIPerfCall, AIPerfTrace, parse_trace};

/// Parse, reconstruct, and lower an `aiperf.trace.v1` source exactly once.
pub async fn compile_aiperf_trace_input(
    config: RecordedTraceInputConfig,
    tokenizer: &dyn TextTokenizer,
) -> Result<GraphInputBundle, RecordedTraceError> {
    config.validate()?;
    if let Some(name) = config.load.options.keys().next() {
        return Err(RecordedTraceError(format!(
            "aiperf_trace Graph-IR input does not support loader option {name:?}"
        )));
    }
    let documents = load_aiperf_documents(&config.load).await?;
    if documents.is_empty() {
        return Err(RecordedTraceError(
            "aiperf_trace source contains no sessions".into(),
        ));
    }
    let source_is_single = documents.len() == 1;

    let mut parsed = Vec::new();
    let mut ids = HashSet::new();
    for document in documents {
        let trace = parse_trace(document)?;
        if !ids.insert(trace.id.clone()) {
            return Err(RecordedTraceError(format!(
                "aiperf_trace source contains duplicate session id {:?}",
                trace.id
            )));
        }
        parsed.push(trace);
        if let Some(limit) = config.root_limit
            && parsed.len() >= limit
        {
            break;
        }
    }

    let owned =
        CorpusContentSynthesizer::new(tokenizer, config.prompt_corpus, config.content_root_seed)?;
    let mut content = owned.as_synthesizer();
    let mut pool = SegmentPool::new();
    let mut programs = Vec::with_capacity(parsed.len());
    for trace in parsed {
        let requests = flatten_trace(&trace)?;
        // Each session is its own hash namespace (block ids are per-session salted).
        let hash_scope = Some(trace.id.as_str());
        let graph = lower_recorded_graph(
            requests,
            trace.block_size,
            config.idle_gap_cap_seconds,
            super::trie::IdleWarpMode::BusyPeriod,
            hash_scope,
            &trace.id,
            &mut content,
            &mut pool,
        )?;
        let mut plan = graph_plan(graph, trace.id);
        if !source_is_single {
            plan.trace.graph_ref = Some(plan.trace.id.clone());
        }
        programs.push(GraphTraceProgram::static_graph(plan));
    }
    programs.sort_by(|left, right| left.profiling.trace.id.cmp(&right.profiling.trace.id));
    let metadata = GraphInputMetadata {
        format: "aiperf_trace".into(),
        root_count: programs.len(),
        node_count: programs
            .iter()
            .map(|program| program.profiling.graph.llm_node_count())
            .sum(),
        warning_facts: Vec::new(),
    };
    Ok(GraphInputBundle {
        programs,
        segments: Arc::new(pool.freeze()),
        metadata,
    })
}

/// Flatten one session's `inference_calls` into normalized `RecordedRequest`s.
///
/// Calls are taken in file order (`aiperf.trace.v1` emits them time-sorted, and
/// its `previous_ref`/`compaction.prior_ref`/`response_refs` are indices into
/// that same order). Each call's prompt is the concatenation of its referenced
/// segments' block ids; per-block `(role, starts_message)` tags come straight
/// from the segment roles. `input_tokens` is the **exact** sum of the referenced
/// segments' token counts, and `block_lens` records each block's true length —
/// full `block_size` blocks plus each segment's partial-tail remainder — so the
/// reconstruction honors the real per-segment length rather than rounding every
/// tail up to a whole block.
fn flatten_trace(trace: &AIPerfTrace) -> Result<Vec<RecordedRequest>, RecordedTraceError> {
    let block_size = trace.block_size;
    let node_ids: Vec<String> = assign_node_ids(trace);
    let mut requests = Vec::with_capacity(trace.calls.len());

    for (index, call) in trace.calls.iter().enumerate() {
        let mut hash_ids = Vec::new();
        let mut explicit_tags = Vec::new();
        let mut block_lens = Vec::new();
        let mut token_sum = 0_usize;
        for &seg_ref in &call.segment_refs {
            let segment = trace.segments.get(seg_ref).ok_or_else(|| {
                RecordedTraceError(format!(
                    "session {:?} call {index}: segment_ref {seg_ref} out of range",
                    trace.id
                ))
            })?;
            let blocks = segment.hash_ids.len();
            if blocks == 0 {
                return Err(RecordedTraceError(format!(
                    "session {:?} call {index}: segment {seg_ref} has no hash_ids",
                    trace.id
                )));
            }
            // The final block of a segment is a partial tail: its length is the
            // remainder `tokens - (blocks - 1) * block_size`, which must fall in
            // `1..=block_size` (i.e. `blocks == ceil(tokens / block_size)`). Every
            // earlier block is a full `block_size`. This preserves the segment's
            // *exact* token count instead of rounding the tail up to a whole block.
            let tail = segment
                .tokens
                .checked_sub(blocks.saturating_sub(1).saturating_mul(block_size))
                .filter(|tail| (1..=block_size).contains(tail))
                .ok_or_else(|| {
                    RecordedTraceError(format!(
                        "session {:?} call {index}: segment {seg_ref} has {} tokens \
                         inconsistent with {blocks} block(s) of size {block_size}",
                        trace.id, segment.tokens
                    ))
                })?;
            token_sum = token_sum.saturating_add(segment.tokens);
            for (block, id) in segment.hash_ids.iter().enumerate() {
                hash_ids.push(*id);
                explicit_tags.push(BlockTag::from_authored(&segment.role, block == 0));
                block_lens.push(if block + 1 == blocks {
                    tail
                } else {
                    block_size
                });
            }
        }
        // Exact reconstructed input length: `Σ segment.tokens == Σ block_lens`.
        let input_tokens = token_sum;
        let output_tokens = call.output_tokens.unwrap_or_else(|| {
            call.response_refs
                .iter()
                .filter_map(|&r| trace.segments.get(r))
                .map(|seg| seg.tokens)
                .sum()
        });

        let chain_id = call
            .agent_id
            .map_or_else(|| trace.id.clone(), |agent| agent.to_string());
        let causal_parent_id = call
            .previous_ref
            .or_else(|| call.compaction.as_ref().and_then(|c| c.prior_ref))
            .and_then(|parent| node_ids.get(parent).cloned());
        let async_ancestors = call
            .parent_agent_id
            .map(|parent| HashSet::from([parent.to_string()]))
            .unwrap_or_default();

        let mut adapter_metadata = BTreeMap::new();
        adapter_metadata.insert("recorded_input_tokens".into(), Value::from(token_sum));
        if let Some(kind) = &call.request_kind {
            adapter_metadata.insert("request_kind".into(), Value::String(kind.clone()));
        }
        if let Some(compaction) = &call.compaction {
            adapter_metadata.insert("compaction".into(), Value::Bool(true));
            if let Some(prior) = compaction.prior_segments {
                adapter_metadata.insert("compaction_prior_segments".into(), Value::from(prior));
            }
        }

        requests.push(RecordedRequest {
            node_id: node_ids[index].clone(),
            chain_id,
            turn_index: turn_index_of(&node_ids[index]),
            order: index,
            hash_ids,
            input_tokens,
            output_tokens,
            start_seconds: call.ts_ms / 1_000.0,
            duration_seconds: call.e2e_latency_ms.unwrap_or(0.0) / 1_000.0,
            model: call.model.clone(),
            streaming: call.ttft_ms.is_some(),
            ttft_seconds: call.ttft_ms.map(|value| value / 1_000.0),
            causal_parent_id,
            async_ancestors,
            max_tokens: output_tokens.max(1),
            extra_headers: BTreeMap::new(),
            adapter_metadata,
            explicit_tags: Some(explicit_tags),
            block_lens: Some(block_lens),
        });
    }

    if requests.is_empty() {
        return Err(RecordedTraceError(format!(
            "aiperf_trace session {:?} has no inference calls",
            trace.id
        )));
    }
    Ok(requests)
}

/// Assign `"{chain}:{turn}"` node ids: `chain` = `agent_id` (else session id),
/// `turn` = per-chain counter in call order.
fn assign_node_ids(trace: &AIPerfTrace) -> Vec<String> {
    let mut turn_by_chain: HashMap<String, usize> = HashMap::new();
    trace
        .calls
        .iter()
        .map(|call: &AIPerfCall| {
            let chain = call
                .agent_id
                .map_or_else(|| trace.id.clone(), |agent| agent.to_string());
            let turn = turn_by_chain.entry(chain.clone()).or_insert(0);
            let node_id = format!("{chain}:{turn}");
            *turn += 1;
            node_id
        })
        .collect()
}

fn turn_index_of(node_id: &str) -> usize {
    node_id
        .rsplit_once(':')
        .and_then(|(_, turn)| turn.parse().ok())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use crate::dataset::{DatasetSource, LoadConfig, Payload, TiktokenTokenizer};
    use serde_json::json;

    use super::*;
    use crate::graph::model::{LlmNode, PromptItem};

    fn config(records: Value) -> RecordedTraceInputConfig {
        RecordedTraceInputConfig {
            load: LoadConfig::new(DatasetSource::Inline(records)),
            root_limit: None,
            max_context_length: None,
            max_osl: None,
            idle_gap_cap_seconds: Some(60.0),
            prompt_corpus: crate::graph::recorded::PromptCorpus::Sonnet,
            content_root_seed: 42,
        }
    }

    fn node<'a>(bundle: &'a GraphInputBundle, node_id: &str) -> &'a LlmNode {
        bundle.programs[0].profiling.graph.nodes[node_id]
            .as_llm()
            .unwrap()
    }

    fn session() -> Value {
        json!({
            "schema": "aiperf.trace.v1",
            "session_id": 42,
            "block_size": 16,
            "source_metadata": {"provider": "anthropic", "generator": "x", "models": ["m"]},
            "hash_id_salt": 42,
            "time_anchor_ms": 0,
            "segments": [
                {"role": "system", "kind": ["system"], "hash_ids": [1, 2], "tokens": 32},
                {"role": "user", "kind": ["text"], "hash_ids": [3], "tokens": 16},
                {"role": "assistant", "kind": ["response"], "hash_ids": [4], "tokens": 16},
                {"role": "tool", "kind": ["tool_result"], "hash_ids": [5], "tokens": 16}
            ],
            "inference_calls": [
                {"ts": 0.0, "model": "m", "ttft_ms": 100.0, "e2e_latency_ms": 200.0,
                 "segment_refs": [0, 1], "response_ref": 2, "usage": {"output_tokens": 8}},
                {"ts": 1000.0, "model": "m", "ttft_ms": 120.0, "e2e_latency_ms": 210.0,
                 "segment_refs": [0, 1, 2, 3], "usage": {"output_tokens": 8}}
            ],
            "role_counts": {},
            "meta": {}
        })
    }

    fn prompt_roles(bundle: &GraphInputBundle, node_id: &str) -> Vec<String> {
        node(bundle, node_id)
            .items
            .iter()
            .map(|item| {
                let PromptItem::Seg { seg } = item else {
                    panic!("recorded prompt must use dense segments");
                };
                match bundle.segments.get(*seg).unwrap() {
                    Payload::Message { role, .. } => role.as_str().to_string(),
                    other => panic!("expected a message payload, got {other:?}"),
                }
            })
            .collect()
    }

    #[tokio::test]
    async fn lowers_calls_with_flattened_hash_ids_and_timing() {
        let bundle = compile_aiperf_trace_input(config(session()), &TiktokenTokenizer::builtin())
            .await
            .unwrap();
        assert_eq!(bundle.metadata.format, "aiperf_trace");
        let graph = &bundle.programs[0].profiling.graph;
        assert_eq!(graph.nodes.len(), 2);
        // streaming + model carried through; call 1 continues call 0's prefix.
        assert!(graph.nodes["42:1"].as_llm().unwrap().streaming);
        assert_eq!(graph.nodes["42:0"].as_llm().unwrap().metadata["model"], "m");
        assert_eq!(
            graph.nodes["42:0"].as_llm().unwrap().metadata["arrival_offset_us"],
            0
        );
        assert_eq!(
            graph.nodes["42:1"].as_llm().unwrap().metadata["arrival_offset_us"],
            1_000_000
        );
    }

    #[tokio::test]
    async fn explicit_segment_roles_drive_the_pool_verbatim() {
        // The heuristic path can only ever emit user/assistant; our ground-truth
        // system + tool roles must survive into the SegmentPool messages exactly.
        let bundle = compile_aiperf_trace_input(config(session()), &TiktokenTokenizer::builtin())
            .await
            .unwrap();
        assert_eq!(prompt_roles(&bundle, "42:0"), ["system", "user"]);
        assert_eq!(
            prompt_roles(&bundle, "42:1"),
            ["system", "user", "assistant", "tool"]
        );
    }

    fn prompt_token_counts(bundle: &GraphInputBundle, node_id: &str) -> Vec<usize> {
        node(bundle, node_id)
            .items
            .iter()
            .map(|item| {
                let PromptItem::Seg { seg } = item else {
                    panic!("recorded prompt must use dense segments");
                };
                match bundle.segments.get(*seg).unwrap() {
                    Payload::Message { tokens, .. } => tokens.len(),
                    other => panic!("expected a message payload, got {other:?}"),
                }
            })
            .collect()
    }

    #[tokio::test]
    async fn partial_tail_segments_materialize_exact_token_counts() {
        // block_size 16; token counts are deliberately NOT multiples of 16, so
        // every segment ends on a partial tail. Each must materialize at its exact
        // length — 40 -> 16+16+8, NOT rounded up to a full 3*16 = 48.
        let records = json!({
            "schema": "aiperf.trace.v1", "session_id": 7, "block_size": 16,
            "segments": [
                {"role": "system", "hash_ids": [1, 2, 3], "tokens": 40},
                {"role": "user", "hash_ids": [4], "tokens": 10},
                {"role": "assistant", "hash_ids": [5], "tokens": 12},
                {"role": "tool", "hash_ids": [6], "tokens": 9}
            ],
            "inference_calls": [
                {"ts": 0.0, "model": "m", "segment_refs": [0, 1],
                 "response_ref": 2, "usage": {"output_tokens": 5}},
                {"ts": 1000.0, "model": "m", "segment_refs": [0, 1, 2, 3],
                 "previous_ref": 0, "usage": {"output_tokens": 5}}
            ]
        });
        let bundle = compile_aiperf_trace_input(config(records), &TiktokenTokenizer::builtin())
            .await
            .unwrap();
        // Per-message counts are the true segment token counts, partial tails intact.
        assert_eq!(prompt_token_counts(&bundle, "7:0"), [40, 10]);
        assert_eq!(prompt_token_counts(&bundle, "7:1"), [40, 10, 12, 9]);
        // Reported ISL is the exact sum (50 / 71), not the block-rounded
        // 4*16 = 64 / 6*16 = 96.
        assert_eq!(
            prompt_token_counts(&bundle, "7:0").iter().sum::<usize>(),
            50
        );
        assert_eq!(
            prompt_token_counts(&bundle, "7:1").iter().sum::<usize>(),
            71
        );
        // The shared-prefix messages still dedup to the same pooled handles across
        // calls (identical bytes => a KV cache hits), partial tail and all.
        let handles = |id: &str| {
            node(&bundle, id)
                .items
                .iter()
                .filter_map(|item| match item {
                    PromptItem::Seg { seg } => Some(*seg),
                    _ => None,
                })
                .collect::<Vec<_>>()
        };
        let (call0, call1) = (handles("7:0"), handles("7:1"));
        assert_eq!(call0, call1[..call0.len()]);
    }

    #[tokio::test]
    async fn response_refs_sum_multiple_segments_into_output_tokens() {
        // A response spanning two pooled segments (assistant text + a tool-call
        // segment): output tokens is their sum, not just the first.
        let records = json!({
            "schema": "aiperf.trace.v1", "session_id": 9, "block_size": 16,
            "segments": [
                {"role": "user", "hash_ids": [1], "tokens": 10},
                {"role": "assistant", "hash_ids": [2], "tokens": 12},
                {"role": "assistant", "hash_ids": [3], "tokens": 7}
            ],
            "inference_calls": [
                {"ts": 0.0, "model": "m", "segment_refs": [0],
                 "response_refs": [1, 2], "usage": {}}
            ]
        });
        let bundle = compile_aiperf_trace_input(config(records), &TiktokenTokenizer::builtin())
            .await
            .unwrap();
        // `max_tokens` equals both response segments: 12 + 7 = 19.
        assert_eq!(node(&bundle, "9:0").max_tokens, Some(19));
    }

    #[tokio::test]
    async fn subagent_call_forms_its_own_chain() {
        let mut records = session();
        // Tag call 1 as a subagent turn: its own chain + an async ancestor.
        records["inference_calls"][1]["agent_id"] = json!(777);
        records["inference_calls"][1]["parent_agent_id"] = json!(42);
        let bundle = compile_aiperf_trace_input(config(records), &TiktokenTokenizer::builtin())
            .await
            .unwrap();
        let graph = &bundle.programs[0].profiling.graph;
        assert!(graph.nodes.contains_key("42:0"));
        assert!(graph.nodes.contains_key("777:0"));
    }
}
