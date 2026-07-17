// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native WEKA trace compiler.

mod schema;

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use crate::dataset::{DatasetSource, SegmentPool, TextTokenizer};

use crate::graph::input::{GraphInputBundle, GraphInputMetadata};

use super::content::CorpusContentSynthesizer;
use super::source::load_weka_documents;
use super::trie::{RecordedRequest, graph_plan, lower_recorded_graph};
use super::{RecordedTraceError, RecordedTraceInputConfig};
use schema::{WekaEntry, WekaTrace, parse_trace};

/// Parse, select, reconstruct, and lower a WEKA source exactly once.
pub async fn compile_weka_trace_input(
    config: RecordedTraceInputConfig,
    tokenizer: &dyn TextTokenizer,
) -> Result<GraphInputBundle, RecordedTraceError> {
    config.validate()?;
    reject_loader_options(&config)?;
    let source_is_single = matches!(
        &config.load.source,
        DatasetSource::Path(path) if path.is_file()
    ) || matches!(&config.load.source, DatasetSource::Bytes(_))
        || matches!(&config.load.source, DatasetSource::Inline(value) if value.is_object());
    let documents = load_weka_documents(&config.load).await?;
    if documents.is_empty() {
        return Err(RecordedTraceError("WEKA source contains no traces".into()));
    }
    let selection_enabled =
        !source_is_single && (config.root_limit.is_some() || config.max_context_length.is_some());
    let mut parsed = Vec::new();
    let mut ids = HashSet::new();
    for document in &documents {
        let trace = parse_trace(document)?;
        if selection_enabled
            && config
                .max_context_length
                .is_some_and(|limit| peak_context(&trace, config.max_osl) > limit)
        {
            continue;
        }
        if !ids.insert(trace.id.clone()) {
            return Err(RecordedTraceError(format!(
                "WEKA source contains duplicate trace id {:?}",
                trace.id
            )));
        }
        parsed.push(trace);
        if selection_enabled && parsed.len() >= config.root_limit.unwrap_or(usize::MAX) {
            break;
        }
    }
    if parsed.is_empty() {
        return Err(RecordedTraceError(
            "WEKA selection rejected every trace".into(),
        ));
    }

    let mut content =
        CorpusContentSynthesizer::new(tokenizer, config.prompt_corpus, config.content_root_seed)?;
    let mut pool = SegmentPool::new();
    let mut plans = Vec::with_capacity(parsed.len());
    for trace in parsed {
        let requests = flatten_trace(&trace, config.max_osl)?;
        // A local WEKA hash namespace is scoped by trace id, not file hash.
        let hash_scope = (!trace.global_hash_scope).then_some(trace.id.as_str());
        let graph = lower_recorded_graph(
            requests,
            trace.block_size,
            config.idle_gap_cap_seconds,
            hash_scope,
            &trace.id,
            &mut content,
            &mut pool,
        )?;
        let mut plan = graph_plan(graph, trace.id);
        if !source_is_single {
            plan.trace.graph_ref = Some(plan.trace.id.clone());
        }
        plans.push(plan);
    }
    plans.sort_by(|left, right| left.trace.id.cmp(&right.trace.id));
    let metadata = GraphInputMetadata {
        format: "weka_trace".into(),
        root_count: plans.len(),
        node_count: plans.iter().map(|plan| plan.graph.nodes.len()).sum(),
    };
    Ok(GraphInputBundle {
        plans,
        segments: Arc::new(pool.freeze()),
        metadata,
    })
}

fn reject_loader_options(config: &RecordedTraceInputConfig) -> Result<(), RecordedTraceError> {
    if let Some(name) = config.load.options.keys().next() {
        return Err(RecordedTraceError(format!(
            "weka_trace Graph-IR input does not support loader option {name:?}"
        )));
    }
    Ok(())
}

fn peak_context(trace: &WekaTrace, max_osl: Option<usize>) -> usize {
    fn visit(entries: &[WekaEntry], max_osl: Option<usize>, top_level: bool) -> usize {
        entries
            .iter()
            .map(|entry| match entry {
                WekaEntry::Leaf(leaf) => {
                    let output = if top_level {
                        max_osl.map_or(leaf.output_tokens, |cap| leaf.output_tokens.min(cap))
                    } else {
                        leaf.output_tokens
                    };
                    leaf.input_tokens.saturating_add(output)
                }
                WekaEntry::Subagent(subagent) => visit(&subagent.requests, None, false),
            })
            .max()
            .unwrap_or(0)
    }
    visit(&trace.requests, max_osl, true)
}

fn flatten_trace(
    trace: &WekaTrace,
    max_osl: Option<usize>,
) -> Result<Vec<RecordedRequest>, RecordedTraceError> {
    let mut output = Vec::new();
    flatten_entries(
        &trace.requests,
        &trace.id,
        None,
        &HashSet::new(),
        true,
        max_osl,
        &mut output,
    )?;
    if output.is_empty() {
        return Err(RecordedTraceError(format!(
            "WEKA trace {:?} flattens to zero normal/streaming leaf requests",
            trace.id
        )));
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn flatten_entries(
    entries: &[WekaEntry],
    scope: &str,
    inherited_spawner: Option<String>,
    async_ancestors: &HashSet<String>,
    top_level: bool,
    max_osl: Option<usize>,
    output: &mut Vec<RecordedRequest>,
) -> Result<(), RecordedTraceError> {
    let mut previous = inherited_spawner;
    let mut turn_index = 0_usize;
    for entry in entries {
        match entry {
            WekaEntry::Leaf(leaf) => {
                let node_id = format!("{scope}:{turn_index}");
                let max_tokens = if top_level {
                    max_osl.map_or(leaf.output_tokens, |cap| leaf.output_tokens.min(cap))
                } else {
                    leaf.output_tokens
                }
                .max(1);
                output.push(RecordedRequest {
                    node_id: node_id.clone(),
                    chain_id: scope.to_string(),
                    turn_index,
                    order: output.len(),
                    hash_ids: leaf.hashes.clone(),
                    input_tokens: leaf.input_tokens,
                    output_tokens: leaf.output_tokens,
                    start_seconds: leaf.start_seconds,
                    duration_seconds: leaf.duration_seconds,
                    model: Some(leaf.model.clone()),
                    streaming: leaf.streaming,
                    ttft_seconds: leaf.ttft_seconds,
                    causal_parent_id: previous.clone(),
                    async_ancestors: async_ancestors.clone(),
                    max_tokens,
                    extra_headers: BTreeMap::new(),
                    adapter_metadata: BTreeMap::new(),
                    explicit_tags: None,
                    block_lens: None,
                });
                previous = Some(node_id);
                turn_index += 1;
            }
            WekaEntry::Subagent(subagent) => {
                let mut child_async = async_ancestors.clone();
                if subagent.status == "async_launched" {
                    child_async.insert(subagent.agent_id.clone());
                }
                flatten_entries(
                    &subagent.requests,
                    &subagent.agent_id,
                    previous.clone(),
                    &child_async,
                    false,
                    None,
                    output,
                )?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::dataset::{DatasetSource, LoadConfig, Payload, TiktokenTokenizer};
    use serde_json::json;

    use super::*;
    use crate::graph::recorded::PromptCorpus;

    #[tokio::test]
    async fn inline_sonnet_trace_lowers_nested_start_anchor_and_materialized_messages() {
        let config = RecordedTraceInputConfig {
            load: LoadConfig::new(DatasetSource::Inline(json!({
                "id": "root",
                "models": ["m"],
                "block_size": 16,
                "hash_id_scope": "global",
                "requests": [
                    {"t": 0, "type": "s", "model": "m", "in": 32, "out": 8,
                     "hash_ids": [1, 2], "api_time": 2, "ttft": 0.5},
                    {"t": 0.5, "type": "subagent", "agent_id": "child",
                     "subagent_type": "x", "status": "completed", "models": ["m"],
                     "requests": [
                         {"t": 0.5, "type": "n", "model": "m", "in": 16,
                          "out": 4, "hash_ids": [9], "api_time": 0.5}
                     ]}
                ]
            }))),
            root_limit: None,
            max_context_length: None,
            max_osl: None,
            idle_gap_cap_seconds: Some(60.0),
            prompt_corpus: PromptCorpus::Sonnet,
            content_root_seed: 42,
        };
        let bundle = compile_weka_trace_input(config, &TiktokenTokenizer::builtin())
            .await
            .unwrap();
        let graph = &bundle.plans[0].graph;
        assert_eq!(graph.nodes.len(), 2);
        let edge = graph
            .edges
            .iter()
            .find(|edge| edge.target == "child:0")
            .unwrap();
        assert_eq!(edge.source, "root:0");
        assert_eq!(edge.delay_after_predecessor_start_us, Some(500_000.0));
        let handle = match graph.nodes["root:0"].items[0] {
            crate::graph::model::PromptItem::Seg { seg } => seg,
            _ => panic!("recorded prompt must use dense segments"),
        };
        assert!(matches!(
            bundle.segments.get(handle).unwrap(),
            Payload::Message { .. }
        ));
    }

    #[tokio::test]
    async fn multi_trace_cap_stops_before_decoding_unselected_documents() {
        let selected = json!({
            "id": "selected",
            "models": ["m"],
            "block_size": 16,
            "hash_id_scope": "global",
            "requests": [
                {"t": 0, "type": "n", "model": "m", "in": 16, "out": 1,
                 "hash_ids": [1]}
            ]
        });
        let config = RecordedTraceInputConfig {
            load: LoadConfig::new(DatasetSource::Inline(json!([
                selected,
                {"id": "selected", "invalid_after_cap": true}
            ]))),
            root_limit: Some(1),
            max_context_length: None,
            max_osl: None,
            idle_gap_cap_seconds: Some(60.0),
            prompt_corpus: PromptCorpus::Sonnet,
            content_root_seed: 42,
        };

        let bundle = compile_weka_trace_input(config, &TiktokenTokenizer::builtin())
            .await
            .expect("the cap must stop the schema scan after one eligible trace");
        assert_eq!(bundle.plans.len(), 1);
        assert_eq!(bundle.plans[0].trace.id, "selected");
    }

    #[tokio::test]
    async fn multi_trace_selection_filters_before_applying_the_root_cap() {
        let trace = |id: &str, input_tokens: usize| {
            json!({
                "id": id,
                "models": ["m"],
                "block_size": 16,
                "hash_id_scope": "global",
                "requests": [
                    {"t": 0, "type": "n", "model": "m", "in": input_tokens,
                     "out": 1, "hash_ids": [1]}
                ]
            })
        };
        let config = RecordedTraceInputConfig {
            load: LoadConfig::new(DatasetSource::Inline(json!([
                trace("too-large", 64),
                trace("first-eligible", 16),
                {"id": "unread-after-cap", "invalid": true}
            ]))),
            root_limit: Some(1),
            max_context_length: Some(17),
            max_osl: None,
            idle_gap_cap_seconds: Some(60.0),
            prompt_corpus: PromptCorpus::Sonnet,
            content_root_seed: 42,
        };

        let bundle = compile_weka_trace_input(config, &TiktokenTokenizer::builtin())
            .await
            .expect("selection must filter before capping");
        assert_eq!(bundle.plans.len(), 1);
        assert_eq!(bundle.plans[0].trace.id, "first-eligible");
    }
}
