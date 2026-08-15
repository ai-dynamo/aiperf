// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native WEKA trace compiler.

mod schema;

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use rayon::prelude::*;
use serde_json::Value;
use serde_json::value::RawValue;

use crate::dataset::{DatasetSource, Handle, SegmentPool, TextTokenizer};

use crate::graph::input::{GraphInputBundle, GraphInputMetadata};
use crate::graph::model::{ExecutableGraphNode, GraphTracePlan, GraphTraceProgram, PromptItem};

use super::content::CorpusShared;
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
    let timing = std::env::var_os("AIPERF_WEKA_TIMING").is_some();
    let clock = || std::time::Instant::now();
    let mut mark = clock();
    macro_rules! phase {
        ($label:expr) => {
            if timing {
                let now = clock();
                eprintln!("[weka-timing] {}: {:.3?}", $label, now.duration_since(mark));
                #[allow(unused_assignments)]
                {
                    mark = now;
                }
            }
        };
    }

    let documents = load_weka_documents(&config.load).await?;
    if documents.is_empty() {
        return Err(RecordedTraceError("WEKA source contains no traces".into()));
    }
    phase!("load_documents");
    let selection_enabled =
        !source_is_single && (config.root_limit.is_some() || config.max_context_length.is_some());

    // A root/context cap makes selection order-dependent and must stop before
    // decoding unselected documents, so it stays a sequential scan. Otherwise
    // the whole source is taken and every document parses in parallel.
    let parsed = if selection_enabled {
        select_traces_sequential(&documents, &config)?
    } else {
        parse_all_traces_parallel(&documents)?
    };
    if parsed.is_empty() {
        return Err(RecordedTraceError(
            "WEKA selection rejected every trace".into(),
        ));
    }
    phase!("parse_traces");

    // Build the immutable corpus once, then lower every independent trace in
    // parallel: a trace's graph references only its own segments, so each
    // borrows the shared corpus and interns into a private pool + private block
    // cache. The fan-out is deterministic per trace (its seeds and intern order
    // are a pure function of its own content) and independent of the thread
    // count. One CoW corpus is shared by reference with no process or
    // serialization overhead.
    let shared = CorpusShared::new(tokenizer, config.prompt_corpus, config.content_root_seed)?;
    phase!("build_corpus");
    let max_osl = config.max_osl;
    let idle_gap = config.idle_gap_cap_seconds;
    let mut lowered = parsed
        .into_par_iter()
        .map(|trace| {
            let started = timing.then(std::time::Instant::now);
            let requests = flatten_trace(&trace, max_osl)?;
            // A local WEKA hash namespace is scoped by trace id, not file hash.
            let hash_scope = (!trace.global_hash_scope).then_some(trace.id.as_str());
            let mut content = shared.synthesizer();
            let mut pool = SegmentPool::new();
            let graph = lower_recorded_graph(
                requests,
                trace.block_size,
                idle_gap,
                // WEKA byte-exact parity with the Python `_IdleGapTimeWarp`
                // oracle: compress consecutive request-start gaps, not
                // busy-period gaps.
                super::trie::IdleWarpMode::BusyPeriod,
                hash_scope,
                &trace.id,
                &mut content,
                &mut pool,
            )?;
            let node_count = graph.nodes.len();
            let mut plan = graph_plan(graph, trace.id);
            if !source_is_single {
                plan.trace.graph_ref = Some(plan.trace.id.clone());
            }
            if let Some(started) = started {
                let elapsed = started.elapsed();
                if elapsed.as_secs_f64() > 0.4 {
                    eprintln!(
                        "[weka-timing]   trace {:?} lower {:.3?} ({node_count} nodes)",
                        plan.trace.id, elapsed
                    );
                }
            }
            Ok((plan, pool))
        })
        .collect::<Result<Vec<(GraphTracePlan, SegmentPool)>, RecordedTraceError>>()?;
    phase!("parallel_lower");

    // Restore the deterministic by-id order the sequential compiler produced,
    // then stitch each private pool into one store, shifting the segment handles
    // baked into each plan by that pool's arena offset in the merged store.
    lowered.sort_by(|(left, _), (right, _)| left.trace.id.cmp(&right.trace.id));
    let mut pool = SegmentPool::new();
    let mut programs = Vec::with_capacity(lowered.len());
    for (mut plan, local_pool) in lowered {
        let offset = pool
            .concat_disjoint(local_pool)
            .map_err(|error| RecordedTraceError(error.to_string()))?;
        shift_plan_handles(&mut plan, offset);
        programs.push(GraphTraceProgram::static_graph(plan));
    }
    phase!("merge_pools");

    let metadata = GraphInputMetadata {
        format: "weka_trace".into(),
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

/// Sequential selection scan for the capped path: parse documents in order,
/// drop any whose peak context exceeds the cap, reject duplicate ids, and stop
/// as soon as the root cap is met so unselected documents are never decoded.
fn select_traces_sequential(
    documents: &[Box<RawValue>],
    config: &RecordedTraceInputConfig,
) -> Result<Vec<WekaTrace>, RecordedTraceError> {
    let mut parsed = Vec::new();
    let mut ids = HashSet::new();
    for document in documents {
        let trace = parse_trace(document)?;
        if config
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
        if parsed.len() >= config.root_limit.unwrap_or(usize::MAX) {
            break;
        }
    }
    Ok(parsed)
}

/// Parse every document in parallel (whole source taken, no early-break to
/// preserve), then reject duplicate ids in a cheap sequential pass.
fn parse_all_traces_parallel(
    documents: &[Box<RawValue>],
) -> Result<Vec<WekaTrace>, RecordedTraceError> {
    // Process the largest documents first. Trace sizes are extremely skewed (a
    // ~150 MB outlier next to an ~800 KB median), and both parse and lower are
    // per-trace-atomic, so starting the biggest trace last would strand 31 cores
    // at the tail. Longest-processing-time-first ordering lets small traces
    // backfill behind the outlier. Output order is restored by the by-id sort
    // after lowering, so this reordering is invisible downstream.
    let mut ordered: Vec<&RawValue> = documents.iter().map(Box::as_ref).collect();
    ordered.sort_by_key(|document| std::cmp::Reverse(document.get().len()));
    let parsed = ordered
        .par_iter()
        .map(|document| parse_trace(document))
        .collect::<Result<Vec<WekaTrace>, RecordedTraceError>>()?;
    let mut ids = HashSet::with_capacity(parsed.len());
    for trace in &parsed {
        if !ids.insert(trace.id.as_str()) {
            return Err(RecordedTraceError(format!(
                "WEKA source contains duplicate trace id {:?}",
                trace.id
            )));
        }
    }
    Ok(parsed)
}

/// Shift every segment handle baked into a per-trace plan by `offset`.
///
/// A trace lowered into a private pool bakes local (0-based) handles into its
/// `PromptItem::Seg` items and into the `prompt_segment_handles` /
/// `extra_headers_handle` metadata. After [`SegmentPool::concat_disjoint`]
/// relocates that pool to `[offset, offset + len)` in the merged store, every
/// baked handle shifts by the same constant so it resolves against the merged
/// arena. The rendered wire is handle-numbering-invariant, so this reproduces
/// byte-identical reconstructed content.
fn shift_plan_handles(plan: &mut GraphTracePlan, offset: u32) {
    if offset == 0 {
        return;
    }
    let bump = u64::from(offset);
    for node in plan.graph.nodes.values_mut() {
        let ExecutableGraphNode::Llm(node) = node else {
            continue;
        };
        for item in &mut node.items {
            if let PromptItem::Seg { seg } = item {
                *seg = Handle::new(seg.index() + offset);
            }
        }
        if let Some(Value::Array(handles)) = node.metadata.get_mut("prompt_segment_handles") {
            for handle in handles.iter_mut() {
                if let Some(index) = handle.as_u64() {
                    *handle = Value::from(index + bump);
                }
            }
        }
        if let Some(handle) = node.metadata.get_mut("extra_headers_handle")
            && let Some(index) = handle.as_u64()
        {
            *handle = Value::from(index + bump);
        }
        if let Some(request) = &mut node.request {
            for handle in [&mut request.tools, &mut request.additional_body] {
                if let Some(handle) = handle {
                    *handle = Handle::new(handle.index() + offset);
                }
            }
        }
    }
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
        let graph = &bundle.programs[0].profiling.graph;
        assert_eq!(graph.nodes.len(), 2);
        let edge = graph
            .edges
            .iter()
            .find(|edge| edge.target == "child:0")
            .unwrap();
        assert_eq!(edge.source, "root:0");
        assert_eq!(edge.delay_after_predecessor_start_us, Some(500_000.0));
        let handle = match graph.nodes["root:0"].as_llm().unwrap().items[0] {
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
        assert_eq!(bundle.programs.len(), 1);
        assert_eq!(bundle.programs[0].profiling.trace.id, "selected");
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
        assert_eq!(bundle.programs.len(), 1);
        assert_eq!(bundle.programs[0].profiling.trace.id, "first-eligible");
    }
}
