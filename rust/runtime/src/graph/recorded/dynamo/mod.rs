// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native Dynamo request-trace compiler.

mod schema;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use crate::dataset::{SegmentPool, TextTokenizer};
use serde_json::value::RawValue;
use serde_json::{Map, Value};

use super::BlockHash;

use crate::graph::input::{GraphInputBundle, GraphInputMetadata};
use crate::graph::model::GraphTraceProgram;

use super::content::CorpusContentSynthesizer;
use super::source::load_dynamo_documents;
use super::trie::{RecordedGraphLowering, RecordedRequest, graph_plan, lower_recorded_graph};
use super::{RecordedTraceError, RecordedTraceInputConfig, rejected_peak_context_error};
use schema::{EventType, ReplayMetrics, RequestMetrics, TraceRecord, parse_record};

const DEFAULT_VIRTUAL_BLOCK_SIZE: usize = 16;

#[derive(Debug)]
struct SessionChain {
    id: String,
    parent: Option<String>,
    records: Vec<TraceRecord>,
}

#[derive(Debug, Hash, PartialEq, Eq)]
enum DedupKey {
    Request(String, String),
    Tool(EventType, String, String, i64, Option<String>),
}

/// Parse, deduplicate, select, and lower a Dynamo capture exactly once.
pub async fn compile_dynamo_trace_input(
    config: RecordedTraceInputConfig,
    tokenizer: &dyn TextTokenizer,
) -> Result<GraphInputBundle, RecordedTraceError> {
    config.validate()?;
    reject_loader_options(&config)?;
    let documents = load_dynamo_documents(&config.load).await?;
    let records = collect_records(documents)?;
    let chains = build_chains(records)?;
    validate_forest(&chains)?;
    let selected = select_trees(&chains, config.root_limit, config.max_context_length)?;
    // Resolve this authority across the complete capture, not just the selected
    // trees, before any per-tree build. A mixed-size tree cannot be hidden behind
    // a root or context filter and make the remaining trees silently executable.
    let block_size = resolve_block_size(&chains)?;
    let owned = CorpusContentSynthesizer::build_owned(
        tokenizer,
        config.prompt_corpus,
        config.content_root_seed,
    )?;
    let mut content = owned.as_synthesizer();
    let mut pool = SegmentPool::new();
    let mut programs = Vec::with_capacity(selected.len());
    for (root, session_ids) in selected {
        let requests = build_tree_requests(&chains, &session_ids, block_size)?;
        let graph = lower_recorded_graph(RecordedGraphLowering {
            requests,
            block_size,
            idle_gap_cap_seconds: config.idle_gap_cap_seconds,
            idle_warp_mode: super::trie::IdleWarpMode::BusyPeriod,
            hash_scope: None,
            tail_scope: &root,
            content: &mut content,
            pool: &mut pool,
        })?;
        let mut plan = graph_plan(graph, root);
        plan.trace.graph_ref = Some(plan.trace.id.clone());
        programs.push(GraphTraceProgram::static_graph(plan));
    }
    programs.sort_by(|left, right| left.profiling.trace.id.cmp(&right.profiling.trace.id));
    if programs.is_empty() {
        return Err(RecordedTraceError(
            "Dynamo selection contains no session trees".into(),
        ));
    }
    let metadata = GraphInputMetadata {
        format: "dynamo_trace".into(),
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

fn reject_loader_options(config: &RecordedTraceInputConfig) -> Result<(), RecordedTraceError> {
    if let Some(name) = config.load.options.keys().next() {
        return Err(RecordedTraceError(format!(
            "dynamo_trace Graph-IR input does not support loader option {name:?}"
        )));
    }
    Ok(())
}

fn collect_records(documents: Vec<Box<RawValue>>) -> Result<Vec<TraceRecord>, RecordedTraceError> {
    let mut seen = HashSet::<DedupKey>::new();
    let mut records = Vec::new();
    for (source_order, value) in documents.into_iter().enumerate() {
        let Some(record) = parse_record(&value, source_order)? else {
            continue;
        };
        let key = record
            .context
            .as_ref()
            .and_then(|context| match record.event_type {
                EventType::RequestEnd => record.request.as_ref().map(|request| {
                    DedupKey::Request(context.session_id.clone(), request.request_id.clone())
                }),
                event_type => record.tool.as_ref().map(|tool| {
                    DedupKey::Tool(
                        event_type,
                        context.session_id.clone(),
                        tool.tool_call_id.clone(),
                        record.event_time_ms,
                        tool.status.clone(),
                    )
                }),
            });
        if key.is_some_and(|key| !seen.insert(key)) {
            continue;
        }
        records.push(record);
    }
    if records.is_empty() {
        return Err(RecordedTraceError(
            "Dynamo trace contains no typed request/tool records".into(),
        ));
    }
    Ok(records)
}

fn build_chains(
    records: Vec<TraceRecord>,
) -> Result<BTreeMap<String, SessionChain>, RecordedTraceError> {
    if !records.iter().any(|record| record.context.is_some()) {
        return Err(RecordedTraceError(
            "Dynamo trace has no agent_context session identity; replay-only records cannot form a graph"
                .into(),
        ));
    }
    let mut parents = HashMap::<String, String>::new();
    let mut grouped = HashMap::<String, Vec<TraceRecord>>::new();
    for record in records {
        let Some(context) = record.context.as_ref() else {
            continue;
        };
        let session_id = context.session_id.clone();
        if !parents.contains_key(&session_id) {
            let candidate = match context
                .parent_trajectory_id
                .as_deref()
                .filter(|parent| !parent.is_empty())
            {
                Some(parent) => (parent != session_id).then_some(parent),
                None => context
                    .parent_session_id
                    .as_deref()
                    .filter(|parent| !parent.is_empty() && *parent != session_id),
            };
            if let Some(parent) = candidate {
                parents.insert(session_id.clone(), parent.to_string());
            }
        }
        grouped.entry(session_id).or_default().push(record);
    }
    let mut chains = BTreeMap::new();
    for (session_id, mut events) in grouped {
        events.sort_by_key(|record| (record.event_time_ms, record.source_order));
        for record in &events {
            if record.event_type == EventType::RequestEnd
                && let Some(request) = &record.request
            {
                validate_request_counts(request, &session_id)?;
            }
        }
        events.retain(|record| record.event_type == EventType::RequestEnd);
        if events.is_empty() {
            continue;
        }
        validate_session_id(&session_id)?;
        chains.insert(
            session_id.clone(),
            SessionChain {
                id: session_id.clone(),
                parent: parents.get(&session_id).cloned(),
                records: events,
            },
        );
    }
    if chains.is_empty() {
        return Err(RecordedTraceError(
            "Dynamo trace contains no request_end records with session identity".into(),
        ));
    }
    Ok(chains)
}

fn validate_request_counts(
    request: &RequestMetrics,
    session_id: &str,
) -> Result<(), RecordedTraceError> {
    for (field, value) in [
        ("input_tokens", request.input_tokens),
        ("output_tokens", request.output_tokens),
        ("cached_tokens", request.cached_tokens),
    ] {
        if value.is_some_and(|value| value < 0) {
            return Err(RecordedTraceError(format!(
                "Dynamo session {session_id:?} request {field} cannot be negative during native lowering"
            )));
        }
    }
    if request
        .replay
        .as_ref()
        .is_some_and(|replay| replay.input_length < 0)
    {
        return Err(RecordedTraceError(format!(
            "Dynamo session {session_id:?} replay input_length cannot be negative during native lowering"
        )));
    }
    Ok(())
}

fn validate_session_id(session_id: &str) -> Result<(), RecordedTraceError> {
    if session_id.is_empty() || session_id.contains(':') {
        return Err(RecordedTraceError(format!(
            "Dynamo session id {session_id:?} must be non-empty and cannot contain ':'"
        )));
    }
    Ok(())
}

fn validate_forest(chains: &BTreeMap<String, SessionChain>) -> Result<(), RecordedTraceError> {
    let max_depth = std::env::var("AIPERF_DYNAMO_MAX_SUBAGENT_DEPTH")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(16);
    if max_depth == 0 {
        return Err(RecordedTraceError(
            "AIPERF_DYNAMO_MAX_SUBAGENT_DEPTH must be positive".into(),
        ));
    }
    for session_id in chains.keys() {
        let mut current = session_id.as_str();
        let mut seen = HashSet::new();
        let mut depth = 1_usize;
        if depth > max_depth {
            return Err(RecordedTraceError(format!(
                "Dynamo session {session_id:?} exceeds maximum subagent depth {max_depth}"
            )));
        }
        seen.insert(current.to_string());
        while let Some(parent) = chains
            .get(current)
            .and_then(|chain| chain.parent.as_deref())
            .filter(|parent| chains.contains_key(*parent))
        {
            if !seen.insert(parent.to_string()) {
                return Err(RecordedTraceError(format!(
                    "Dynamo session parent cycle includes {parent:?}"
                )));
            }
            depth += 1;
            if depth > max_depth {
                return Err(RecordedTraceError(format!(
                    "Dynamo session {session_id:?} exceeds maximum subagent depth {max_depth}"
                )));
            }
            current = parent;
        }
    }
    Ok(())
}

fn select_trees(
    chains: &BTreeMap<String, SessionChain>,
    root_limit: Option<usize>,
    max_context: Option<usize>,
) -> Result<Vec<(String, Vec<String>)>, RecordedTraceError> {
    let mut trees = BTreeMap::<String, Vec<String>>::new();
    for session_id in chains.keys() {
        let mut root = session_id.as_str();
        while let Some(parent) = chains
            .get(root)
            .and_then(|chain| chain.parent.as_deref())
            .filter(|parent| chains.contains_key(*parent))
        {
            root = parent;
        }
        trees
            .entry(root.to_string())
            .or_default()
            .push(session_id.clone());
    }
    let mut selected = Vec::new();
    let mut scanned = 0;
    let mut smallest_rejected = None;
    for (root, mut sessions) in trees {
        sessions.sort();
        let peak = sessions
            .iter()
            .flat_map(|session| &chains[session].records)
            .map(|record| request_peak_context(record.request.as_ref()))
            .max()
            .unwrap_or(0);
        scanned += 1;
        if let Some(limit) = max_context
            && peak > limit
        {
            smallest_rejected =
                Some(smallest_rejected.map_or(peak, |smallest: usize| smallest.min(peak)));
            continue;
        }
        selected.push((root, sessions));
        if selected.len() >= root_limit.unwrap_or(usize::MAX) {
            break;
        }
    }
    if selected.is_empty() {
        if let (Some(limit), Some(smallest)) = (max_context, smallest_rejected) {
            return Err(rejected_peak_context_error(
                "Dynamo selection",
                scanned,
                root_limit,
                limit,
                smallest,
            ));
        }
        return Err(RecordedTraceError(
            "Dynamo selection rejected every session tree".into(),
        ));
    }
    Ok(selected)
}

fn request_peak_context(request: Option<&RequestMetrics>) -> usize {
    let input = request
        .and_then(|request| request.replay.as_ref())
        .map(|replay| usize::try_from(replay.input_length).expect("validated non-negative replay"))
        .or_else(|| {
            request
                .and_then(|request| request.input_tokens)
                .filter(|value| *value > 0)
                .map(|value| usize::try_from(value).expect("validated non-negative input"))
        })
        .unwrap_or(1);
    input.saturating_add(
        request
            .and_then(|request| request.output_tokens)
            .filter(|value| *value > 0)
            .map(|value| usize::try_from(value).expect("validated non-negative output"))
            .unwrap_or(0),
    )
}

/// Collapse a stream of replay records to their single shared block size, or
/// `fallback` when none carry one. A capture that mixes block sizes is rejected.
fn unique_block_size<'a>(
    replays: impl Iterator<Item = &'a ReplayMetrics>,
    fallback: usize,
) -> Result<usize, RecordedTraceError> {
    let mut found = None;
    for replay in replays {
        if found.is_some_and(|value| value != replay.block_size) {
            return Err(RecordedTraceError(format!(
                "Dynamo capture mixes replay block sizes {} and {}",
                found.unwrap(),
                replay.block_size
            )));
        }
        found = Some(replay.block_size);
    }
    Ok(found.unwrap_or(fallback))
}

fn resolve_block_size(
    chains: &BTreeMap<String, SessionChain>,
) -> Result<usize, RecordedTraceError> {
    unique_block_size(
        chains
            .values()
            .flat_map(|chain| &chain.records)
            .filter_map(|record| record.request.as_ref())
            .filter_map(|request| request.replay.as_ref()),
        DEFAULT_VIRTUAL_BLOCK_SIZE,
    )
}

#[derive(Debug)]
struct NormalizedTurn {
    session_id: String,
    parent_session_id: Option<String>,
    turn_index: usize,
    start_ms: i64,
    duration_ms: i64,
    input_tokens: usize,
    output_tokens: usize,
    hashes: Vec<BlockHash>,
    request: Option<RequestMetrics>,
    is_final: bool,
    virtual_fallback: bool,
}

fn build_tree_requests(
    chains: &BTreeMap<String, SessionChain>,
    session_ids: &[String],
    block_size: usize,
) -> Result<Vec<RecordedRequest>, RecordedTraceError> {
    let block_size = tree_block_size(chains, session_ids, block_size)?;
    let mut turns = Vec::<NormalizedTurn>::new();
    for session_id in session_ids {
        let chain = &chains[session_id];
        let request_records = &chain.records;
        for (turn_index, record) in request_records.iter().enumerate() {
            let request = record.request.clone();
            let input_tokens = request
                .as_ref()
                .and_then(|request| request.replay.as_ref())
                .map(|replay| {
                    usize::try_from(replay.input_length).expect("validated non-negative replay")
                })
                .or_else(|| {
                    request
                        .as_ref()
                        .and_then(|request| request.input_tokens)
                        .filter(|value| *value > 0)
                        .map(|value| usize::try_from(value).expect("validated non-negative input"))
                })
                .unwrap_or(1);
            let virtual_fallback = request
                .as_ref()
                .is_none_or(|request| request.replay.is_none());
            let output_tokens = request
                .as_ref()
                .and_then(|request| request.output_tokens)
                .filter(|value| *value > 0)
                .map(|value| usize::try_from(value).expect("validated non-negative output"))
                .unwrap_or(0);
            let duration_ms = request_duration_ms(record.event_time_ms, request.as_ref())?;
            let start_ms = request
                .as_ref()
                .and_then(|request| request.request_received_ms)
                .unwrap_or_else(|| record.event_time_ms.saturating_sub(duration_ms));
            turns.push(NormalizedTurn {
                session_id: chain.id.clone(),
                parent_session_id: chain.parent.clone(),
                turn_index,
                start_ms,
                duration_ms,
                input_tokens,
                output_tokens,
                hashes: Vec::new(),
                request,
                is_final: turn_index + 1 == request_records.len(),
                virtual_fallback,
            });
        }
    }
    let mut order = (0..turns.len()).collect::<Vec<_>>();
    order.sort_by_key(|index| {
        (
            turns[*index].start_ms,
            turns[*index].session_id.clone(),
            turns[*index].turn_index,
        )
    });
    let mut next_virtual = -1_i64;
    let mut virtual_previous = HashMap::<String, Vec<BlockHash>>::new();
    for &index in &order {
        let turn = &mut turns[index];
        if let Some(replay) = turn
            .request
            .as_ref()
            .and_then(|request| request.replay.as_ref())
        {
            let mut hashes = replay.hashes.clone();
            let input_length = usize::try_from(replay.input_length)
                .expect("validated non-negative replay input length");
            validate_replay_alignment(input_length, block_size, hashes.len())?;
            if !hashes.is_empty() && input_length < hashes.len().saturating_mul(block_size) {
                hashes.pop();
            }
            virtual_previous.insert(turn.session_id.clone(), hashes.clone());
            turn.hashes = hashes;
        } else {
            let full_blocks = turn.input_tokens / block_size;
            let mut hashes = virtual_previous
                .get(&turn.session_id)
                .cloned()
                .unwrap_or_default();
            hashes.truncate(full_blocks);
            while hashes.len() < full_blocks {
                hashes.push(BlockHash::from(next_virtual));
                next_virtual -= 1;
            }
            virtual_previous.insert(turn.session_id.clone(), hashes.clone());
            turn.hashes = hashes;
        }
    }
    let origin_ms = turns.iter().map(|turn| turn.start_ms).min().unwrap_or(0);
    let by_session = turns.iter().enumerate().fold(
        HashMap::<&str, Vec<usize>>::new(),
        |mut map, (index, turn)| {
            map.entry(&turn.session_id).or_default().push(index);
            map
        },
    );
    let causal = turns
        .iter()
        .map(|turn| {
            if turn.turn_index > 0 {
                return Some(format!("{}:{}", turn.session_id, turn.turn_index - 1));
            }
            let parent = turn.parent_session_id.as_deref()?;
            by_session.get(parent).and_then(|indices| {
                indices
                    .iter()
                    .copied()
                    .filter(|index| turns[*index].start_ms <= turn.start_ms)
                    .max_by_key(|index| (turns[*index].start_ms, turns[*index].turn_index))
                    .map(|index| format!("{}:{}", turns[index].session_id, turns[index].turn_index))
            })
        })
        .collect::<Vec<_>>();
    let mut requests = Vec::with_capacity(turns.len());
    for index in order {
        let turn = &turns[index];
        let mut headers = BTreeMap::new();
        headers.insert("x-dynamo-session-id".into(), turn.session_id.clone());
        if let Some(parent) = &turn.parent_session_id {
            headers.insert("x-dynamo-parent-session-id".into(), parent.clone());
        }
        if turn.is_final {
            headers.insert("x-dynamo-session-final".into(), "true".into());
        }
        let small_prompt =
            turn.hashes.len().min(turn.input_tokens / block_size) == 0 && turn.input_tokens > 0;
        let mut dynamo = Map::new();
        dynamo.insert("session_id".into(), Value::String(turn.session_id.clone()));
        dynamo.insert(
            "parent_session_id".into(),
            turn.parent_session_id
                .clone()
                .map_or(Value::Null, Value::String),
        );
        dynamo.insert("turn_index".into(), Value::from(turn.turn_index));
        dynamo.insert("small_prompt".into(), Value::Bool(small_prompt));
        let mut metadata = BTreeMap::new();
        metadata.insert("dynamo".into(), Value::Object(dynamo));
        metadata.insert(
            "expected".into(),
            serde_json::json!({
                "input_tokens": turn.request.as_ref().and_then(|request| request.input_tokens),
                "output_tokens": turn.request.as_ref().and_then(|request| request.output_tokens),
                "cache_read_tokens": turn.request.as_ref().and_then(|request| request.cached_tokens),
            }),
        );
        if turn.virtual_fallback {
            metadata.insert("virtual_hash_fallback".into(), Value::Bool(true));
        }
        requests.push(RecordedRequest {
            node_id: format!("{}:{}", turn.session_id, turn.turn_index),
            chain_id: turn.session_id.clone(),
            turn_index: turn.turn_index,
            order: requests.len(),
            hash_ids: turn.hashes.clone(),
            input_tokens: turn.input_tokens,
            output_tokens: turn.output_tokens,
            start_seconds: turn.start_ms.saturating_sub(origin_ms) as f64 / 1_000.0,
            duration_seconds: turn.duration_ms as f64 / 1_000.0,
            model: turn
                .request
                .as_ref()
                .and_then(|request| request.model.clone()),
            streaming: turn
                .request
                .as_ref()
                .is_some_and(|request| request.ttft_ms.is_some()),
            ttft_seconds: turn
                .request
                .as_ref()
                .and_then(|request| request.ttft_ms)
                .map(|value| value / 1_000.0),
            causal_parent_id: causal[index].clone(),
            async_ancestors: HashSet::new(),
            max_tokens: turn.output_tokens.max(1),
            extra_headers: headers,
            adapter_metadata: metadata,
            explicit_tags: None,
            block_lens: None,
        });
    }
    Ok(requests)
}

fn tree_block_size(
    chains: &BTreeMap<String, SessionChain>,
    session_ids: &[String],
    fallback: usize,
) -> Result<usize, RecordedTraceError> {
    unique_block_size(
        session_ids
            .iter()
            .flat_map(|session| &chains[session].records)
            .filter_map(|record| record.request.as_ref())
            .filter_map(|request| request.replay.as_ref()),
        fallback,
    )
}

fn validate_replay_alignment(
    input_length: usize,
    block_size: usize,
    hash_count: usize,
) -> Result<(), RecordedTraceError> {
    if hash_count == 0 {
        if input_length == 0 {
            return Ok(());
        }
    } else {
        let lower = (hash_count - 1).saturating_mul(block_size);
        let upper = hash_count.saturating_mul(block_size);
        if lower < input_length && input_length <= upper {
            return Ok(());
        }
    }
    Err(RecordedTraceError(format!(
        "Dynamo replay hash geometry is incompatible: input_length={input_length}, block_size={block_size}, hashes={hash_count}"
    )))
}

fn request_duration_ms(
    event_time_ms: i64,
    request: Option<&RequestMetrics>,
) -> Result<i64, RecordedTraceError> {
    if let Some(total) = request.and_then(|request| request.total_time_ms) {
        if !total.is_finite() || total < i64::MIN as f64 || total >= i64::MAX as f64 {
            return Err(RecordedTraceError(
                "Dynamo total_time_ms is outside the finite i64 range".into(),
            ));
        }
        return Ok((total.round_ties_even() as i64).max(0));
    }
    Ok(request
        .and_then(|request| request.request_received_ms)
        .map_or(0, |received| event_time_ms.saturating_sub(received).max(0)))
}

#[cfg(test)]
mod tests {
    use crate::dataset::{DatasetSource, LoadConfig, TiktokenTokenizer};
    use serde_json::json;

    use super::*;
    use crate::graph::recorded::PromptCorpus;

    /// Re-serialize `json!`-built records into raw tokens for `collect_records`.
    fn raws(values: Vec<Value>) -> Vec<Box<RawValue>> {
        values
            .iter()
            .map(|value| serde_json::value::to_raw_value(value).unwrap())
            .collect()
    }

    fn inline_config(records: Vec<Value>) -> RecordedTraceInputConfig {
        RecordedTraceInputConfig {
            load: LoadConfig::new(DatasetSource::Inline(Value::Array(records))),
            root_limit: None,
            max_context_length: None,
            max_osl: None,
            idle_gap_cap_seconds: Some(60.0),
            prompt_corpus: PromptCorpus::Sonnet,
            content_root_seed: 42,
        }
    }

    fn fallback_record(session: &str, request_id: &str, start_ms: i64, input: i64) -> Value {
        json!({
            "schema": "dynamo.request.trace.v1",
            "event_type": "request_end",
            "event_time_unix_ms": start_ms,
            "agent_context": {"session_id": session},
            "request": {
                "request_id": request_id,
                "input_tokens": input,
                "output_tokens": 1,
                "request_received_ms": start_ms,
                "total_time_ms": 0
            }
        })
    }

    #[tokio::test]
    async fn inline_tree_lowers_headers_partial_tail_and_streaming() {
        let records = vec![
            json!({
                "schema": "dynamo.request.trace.v1", "event_type": "request_end",
                "event_time_unix_ms": 1000,
                "agent_context": {"session_id": "root"},
                "request": {"request_id": "r0", "model": "m", "input_tokens": 21,
                    "output_tokens": 0, "ttft_ms": 0, "request_received_ms": 500,
                    "replay": {"trace_block_size": 16, "input_length": 21,
                        "input_sequence_hashes": [1, 999]}}
            }),
            json!({
                "schema": "dynamo.request.trace.v1", "event_type": "request_end",
                "event_time_unix_ms": 1300,
                "agent_context": {"session_id": "child", "parent_session_id": "root"},
                "request": {"request_id": "c0", "input_tokens": 7, "output_tokens": 4,
                    "request_received_ms": 900}
            }),
        ];
        let config = RecordedTraceInputConfig {
            load: LoadConfig::new(DatasetSource::Inline(Value::Array(records))),
            root_limit: None,
            max_context_length: None,
            max_osl: None,
            idle_gap_cap_seconds: Some(60.0),
            prompt_corpus: PromptCorpus::Sonnet,
            content_root_seed: 42,
        };
        let bundle = compile_dynamo_trace_input(config, &TiktokenTokenizer::builtin())
            .await
            .unwrap();
        let graph = &bundle.programs[0].profiling.graph;
        assert!(graph.nodes["root:0"].as_llm().unwrap().streaming);
        assert_eq!(graph.nodes["root:0"].as_llm().unwrap().max_tokens, Some(1));
        assert_eq!(
            graph.nodes["root:0"].as_llm().unwrap().metadata["theoretical_prefix_cache_total_blocks"],
            1
        );
        assert_eq!(
            graph.nodes["child:0"].as_llm().unwrap().metadata["dynamo"]["small_prompt"],
            true
        );
    }

    #[tokio::test]
    async fn request_end_without_optional_request_payload_lowers_as_fallback_turn() {
        let config = RecordedTraceInputConfig {
            load: LoadConfig::new(DatasetSource::Inline(json!([{
                "schema": "dynamo.request.trace.v1",
                "event_type": "request_end",
                "event_time_unix_ms": 1000,
                "agent_context": {"session_id": "root"},
                "request": null
            }]))),
            root_limit: None,
            max_context_length: None,
            max_osl: None,
            idle_gap_cap_seconds: Some(60.0),
            prompt_corpus: PromptCorpus::Sonnet,
            content_root_seed: 42,
        };

        let bundle = compile_dynamo_trace_input(config, &TiktokenTokenizer::builtin())
            .await
            .expect("request is optional in the Dynamo v1 trace schema");
        let node = bundle.programs[0].profiling.graph.nodes["root:0"]
            .as_llm()
            .unwrap();
        assert_eq!(node.max_tokens, Some(1));
        assert_eq!(node.metadata["input_tokens"], 1);
        assert_eq!(node.metadata["expected"]["input_tokens"], Value::Null);
        assert_eq!(node.metadata["expected"]["output_tokens"], Value::Null);
        assert_eq!(node.metadata["virtual_hash_fallback"], true);
    }

    #[test]
    fn virtual_hashes_follow_global_start_order_and_do_not_resurrect_after_shrink() {
        let mut child = fallback_record("b", "b0", 1_000, 16);
        child["agent_context"]["parent_session_id"] = json!("a");
        let records = vec![
            fallback_record("a", "a0", 0, 32),
            child,
            fallback_record("a", "a1", 2_000, 16),
            fallback_record("a", "a2", 3_000, 32),
        ];
        let chains = build_chains(collect_records(raws(records)).unwrap()).unwrap();
        let requests = build_tree_requests(&chains, &["a".into(), "b".into()], 16).unwrap();
        let hashes = requests
            .iter()
            .map(|request| {
                (
                    request.node_id.as_str(),
                    request
                        .hash_ids
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            hashes,
            [
                ("a:0", vec!["-1".into(), "-2".into()]),
                ("b:0", vec!["-3".into()]),
                ("a:1", vec!["-1".into()]),
                ("a:2", vec!["-1".into(), "-4".into()]),
            ]
        );
    }

    #[tokio::test]
    async fn mixed_block_sizes_fail_even_when_selection_would_hide_one_tree() {
        let mut first = fallback_record("a", "a0", 0, 16);
        first["request"]["replay"] = json!({
            "trace_block_size": 16,
            "input_length": 16,
            "input_sequence_hashes": [1]
        });
        let mut hidden = fallback_record("z", "z0", 1, 32);
        hidden["request"]["replay"] = json!({
            "trace_block_size": 32,
            "input_length": 32,
            "input_sequence_hashes": [2]
        });
        let mut config = inline_config(vec![first, hidden]);
        config.root_limit = Some(1);
        let error = compile_dynamo_trace_input(config, &TiktokenTokenizer::builtin())
            .await
            .err()
            .expect("mixed block sizes must fail before selected-tree build");
        assert!(error.to_string().contains("mixes replay block sizes"));
    }

    #[test]
    fn first_non_self_parent_from_any_event_is_the_single_tree_authority() {
        let records = vec![
            json!({
                "schema": "dynamo.request.trace.v1", "event_type": "tool_start",
                "event_time_unix_ms": 0,
                "agent_context": {
                    "session_id": "child", "parent_trajectory_id": "root"
                },
                "tool": {"tool_call_id": "t", "tool_class": "shell"}
            }),
            fallback_record("child", "c0", 1, 16),
            fallback_record("root", "r0", 0, 16),
            json!({
                "schema": "dynamo.request.trace.v1", "event_type": "tool_start",
                "event_time_unix_ms": 2,
                "agent_context": {
                    "session_id": "child", "parent_trajectory_id": "other"
                },
                "tool": {"tool_call_id": "t2", "tool_class": "shell"}
            }),
        ];
        let chains = build_chains(collect_records(raws(records)).unwrap()).unwrap();
        assert_eq!(chains["child"].parent.as_deref(), Some("root"));
        let trees = select_trees(&chains, None, None).unwrap();
        assert_eq!(
            trees,
            [("root".into(), vec!["child".into(), "root".into()])]
        );
    }

    #[test]
    fn authoritative_self_trajectory_does_not_use_fallback_parent() {
        let mut child = fallback_record("child", "c0", 1, 16);
        child["agent_context"] = json!({
            "session_id": "child",
            "parent_trajectory_id": "child",
            "parent_session_id": "root"
        });
        let records = vec![fallback_record("root", "r0", 0, 16), child];
        let chains = build_chains(collect_records(raws(records)).unwrap()).unwrap();
        assert_eq!(chains["child"].parent, None);
        assert_eq!(
            select_trees(&chains, None, None).unwrap(),
            [
                ("child".into(), vec!["child".into()]),
                ("root".into(), vec!["root".into()]),
            ]
        );
    }

    #[test]
    fn all_context_rejected_trees_report_the_smallest_peak() {
        let records = vec![
            fallback_record("larger", "l0", 0, 120),
            fallback_record("smaller", "s0", 1, 40),
        ];
        let chains = build_chains(collect_records(raws(records)).unwrap()).unwrap();
        let error = select_trees(&chains, None, Some(20))
            .expect_err("every session tree exceeds the context cap");
        assert!(
            error.0.contains("No eligible traces in Dynamo selection after filter-then-cap (scanned 2, --max-context-length=20"),
            "{}",
            error.0
        );
        assert!(
            error.0.contains("Smallest trace requires 41 tokens; raise --max-context-length to at least that (e.g. --max-context-length 41) to admit any trace."),
            "{}",
            error.0
        );
    }

    #[test]
    fn tool_only_session_scope_is_dropped_before_node_id_validation() {
        let records = vec![
            json!({
                "schema": "dynamo.request.trace.v1", "event_type": "tool_start",
                "event_time_unix_ms": 0,
                "agent_context": {"session_id": "tool:only"},
                "tool": {"tool_call_id": "t", "tool_class": "shell"}
            }),
            fallback_record("root", "r0", 1, 16),
        ];
        let chains = build_chains(collect_records(raws(records)).unwrap()).unwrap();
        assert_eq!(
            chains.keys().map(String::as_str).collect::<Vec<_>>(),
            ["root"]
        );
    }
}
