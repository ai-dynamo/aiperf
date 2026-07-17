// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared LCP-trie lowering used by both recorded adapters.

mod messages;
mod parents;
mod timing;

pub(crate) use messages::BlockTag;

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::dataset::SegmentPool;
use serde_json::Value;

use crate::graph::model::{
    ChannelRequirement, ChannelSpec, ChannelType, Count, GraphRecord, LlmNode, PromptItem,
    ReducerName,
};

use super::content::RecordedContentSynthesizer;
use super::{BlockHash, RecordedTraceError};

/// One format-normalized recorded inference request.
#[derive(Debug, Clone)]
pub(crate) struct RecordedRequest {
    pub node_id: String,
    pub chain_id: String,
    pub turn_index: usize,
    pub order: usize,
    pub hash_ids: Vec<BlockHash>,
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub start_seconds: f64,
    pub duration_seconds: f64,
    pub model: Option<String>,
    pub streaming: bool,
    pub ttft_seconds: Option<f64>,
    pub causal_parent_id: Option<String>,
    pub async_ancestors: HashSet<String>,
    pub max_tokens: usize,
    pub extra_headers: BTreeMap<String, String>,
    pub adapter_metadata: BTreeMap<String, Value>,
    /// Ground-truth per-block `(role, starts_message)` tags supplied by the
    /// `aiperf_trace` adapter, which knows exact message boundaries. `None` for
    /// WEKA/Dynamo, which fall back to the token-geometry heuristic.
    pub explicit_tags: Option<Vec<BlockTag>>,
    /// Ground-truth per-block token lengths, aligned 1:1 with `hash_ids`, supplied
    /// by the `aiperf_trace` adapter. Every entry is `block_size` except a
    /// message's final block, which carries its exact **partial-tail** length
    /// (`tokens - (blocks - 1) * block_size`). This makes reconstruction honor the
    /// real per-segment token count instead of rounding each tail up to a full
    /// block. `None` for WEKA/Dynamo, whose blocks are uniformly `block_size` with
    /// any remainder handled as a single prompt-level tail. When present,
    /// `Σ block_lens == input_tokens`.
    pub block_lens: Option<Vec<usize>>,
}

impl RecordedRequest {
    pub(crate) fn raw_end(&self) -> f64 {
        self.start_seconds + self.duration_seconds
    }
}

#[derive(Debug, Clone)]
struct TrieNode {
    request: RecordedRequest,
    content_parent: Option<usize>,
    warped_start: f64,
    rank: usize,
}

impl TrieNode {
    fn end(&self) -> f64 {
        self.warped_start + self.request.duration_seconds
    }
}

/// Lower one trace/tree through the frozen common trie algorithm.
pub(crate) fn lower_recorded_graph(
    requests: Vec<RecordedRequest>,
    block_size: usize,
    idle_gap_cap_seconds: Option<f64>,
    hash_scope: Option<&str>,
    tail_scope: &str,
    content: &mut dyn RecordedContentSynthesizer,
    pool: &mut SegmentPool,
) -> Result<GraphRecord, RecordedTraceError> {
    if requests.is_empty() {
        return Err(RecordedTraceError(
            "recorded trace contains zero inference requests".into(),
        ));
    }
    if block_size == 0 {
        return Err(RecordedTraceError(
            "recorded trace block_size must be positive".into(),
        ));
    }
    let mut nodes = requests
        .into_iter()
        .map(|request| TrieNode {
            warped_start: request.start_seconds,
            request,
            content_parent: None,
            rank: 0,
        })
        .collect::<Vec<_>>();
    parents::resolve_content_parents(&mut nodes);
    timing::apply_idle_warp(&mut nodes, idle_gap_cap_seconds);
    timing::compute_ranks(&mut nodes);
    let mut edges = timing::build_interval_edges(&nodes);
    timing::apply_start_anchors(&nodes, &mut edges);

    let (tags, _) = messages::assign_block_tags(&nodes, block_size)?;
    let mut state = BTreeMap::new();
    let mut graph_nodes = BTreeMap::new();
    let mut all_edges = Vec::new();
    let prefix_cache = theoretical_prefix_cache(&nodes);
    // Reusing interned shared-prefix messages avoids quadratic tokenization and
    // BLAKE3 hashing on deep traces.
    let mut message_cache = messages::PromptMessageCache::new();

    for (index, node) in nodes.iter().enumerate() {
        let prompt = messages::emit_prompt(
            node,
            &tags[index],
            block_size,
            hash_scope,
            tail_scope,
            content,
            pool,
            &mut message_cache,
        )?;
        let response_tokens = content.tail_tokens(
            node.request.output_tokens,
            &format!("{tail_scope}:{}:response", node.request.node_id),
        );
        let response_content = content.decode(&response_tokens)?;
        let response_parent = prompt.last().copied();
        messages::intern_message(
            pool,
            response_parent,
            "assistant",
            &response_content,
            &response_tokens,
        )?;

        let incoming = edges.remove(&node.request.node_id).unwrap_or_default();
        let inputs = incoming
            .iter()
            .filter(|edge| {
                edge.source != crate::graph::model::START_NODE_ID
                    && edge.delay_after_predecessor_start_us.is_none()
            })
            .map(|edge| ChannelRequirement {
                channel: format!("{}_out", edge.source),
                count: Count::N(1),
            })
            .collect::<Vec<_>>();
        let is_start_root = incoming
            .iter()
            .any(|edge| edge.source == crate::graph::model::START_NODE_ID);
        all_edges.extend(incoming);

        let mut metadata = node.request.adapter_metadata.clone();
        metadata.insert(
            "conversation_id".into(),
            Value::String(node.request.chain_id.clone()),
        );
        metadata.insert("turn_index".into(), Value::from(node.request.turn_index));
        metadata.insert(
            "input_tokens".into(),
            Value::from(node.request.input_tokens),
        );
        metadata.insert(
            "recorded_output_tokens".into(),
            Value::from(node.request.output_tokens),
        );
        metadata.entry("expected".into()).or_insert_with(|| {
            serde_json::json!({
                "input_tokens": node.request.input_tokens,
                "output_tokens": node.request.output_tokens,
                "cache_read_tokens": null,
            })
        });
        let arrival_offset_us = (node.warped_start * 1_000_000.0).round_ties_even();
        if !(0.0..u64::MAX as f64).contains(&arrival_offset_us) {
            return Err(RecordedTraceError(format!(
                "node {:?}: warped arrival offset is outside u64 microseconds",
                node.request.node_id
            )));
        }
        metadata.insert(
            "arrival_offset_us".into(),
            Value::from(arrival_offset_us as u64),
        );
        metadata.insert(
            "prompt_segment_handles".into(),
            Value::Array(
                prompt
                    .iter()
                    .map(|handle| Value::from(handle.index()))
                    .collect(),
            ),
        );
        if let Some(model) = &node.request.model {
            metadata.insert("model".into(), Value::String(model.clone()));
        }
        let (hit_blocks, total_blocks) = prefix_cache[&node.request.node_id];
        if total_blocks > 0 {
            metadata.insert(
                "theoretical_prefix_cache_hit_blocks".into(),
                Value::from(hit_blocks),
            );
            metadata.insert(
                "theoretical_prefix_cache_total_blocks".into(),
                Value::from(total_blocks),
            );
        }
        if !node.request.extra_headers.is_empty() {
            let wire = serde_json::to_vec(&node.request.extra_headers)?;
            let handle = pool.intern_raw(prompt.last().copied(), wire)?;
            metadata.insert("extra_headers_handle".into(), Value::from(handle.index()));
        }

        let output = format!("{}_out", node.request.node_id);
        state.insert(
            output.clone(),
            ChannelSpec {
                channel_type: ChannelType::Text,
                reducer: ReducerName::Overwrite,
            },
        );
        graph_nodes.insert(
            node.request.node_id.clone(),
            LlmNode {
                output,
                streaming: node.request.streaming,
                inputs,
                min_start_delay_us: is_start_root.then_some(node.warped_start * 1_000_000.0),
                max_tokens: Some(node.request.max_tokens.max(1)),
                items: prompt
                    .into_iter()
                    .map(|seg| PromptItem::Seg { seg })
                    .collect(),
                metadata,
            },
        );
    }

    let graph = GraphRecord {
        version: Some("2.0".into()),
        system: None,
        state,
        nodes: graph_nodes,
        edges: all_edges,
    };
    let errors = crate::graph::validate::validate(&graph);
    if !errors.is_empty() {
        return Err(RecordedTraceError(format!(
            "recorded graph failed structural validation: {}",
            errors
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("; ")
        )));
    }
    Ok(graph)
}

fn theoretical_prefix_cache(nodes: &[TrieNode]) -> HashMap<String, (usize, usize)> {
    let mut order = (0..nodes.len()).collect::<Vec<_>>();
    order.sort_by(|left, right| {
        nodes[*left]
            .request
            .start_seconds
            .total_cmp(&nodes[*right].request.start_seconds)
            .then_with(|| nodes[*left].request.order.cmp(&nodes[*right].request.order))
    });
    let mut seen = HashSet::<BlockHash>::new();
    let mut stats = HashMap::with_capacity(nodes.len());
    for index in order {
        let request = &nodes[index].request;
        let hit = request
            .hash_ids
            .iter()
            .take_while(|hash| seen.contains(*hash))
            .count();
        stats.insert(request.node_id.clone(), (hit, request.hash_ids.len()));
        seen.extend(request.hash_ids.iter().cloned());
    }
    stats
}

pub(crate) fn graph_plan(
    graph: GraphRecord,
    trace_id: String,
) -> crate::graph::model::GraphTracePlan {
    crate::graph::model::GraphTracePlan {
        graph,
        trace: crate::graph::model::TraceRecord {
            id: trace_id,
            graph_ref: None,
            initial_state: BTreeMap::new(),
        },
        arrival_offset_ns: None,
    }
}
