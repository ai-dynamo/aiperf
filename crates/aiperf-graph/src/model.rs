// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Graph-IR data model: the runtime-relevant subset of a parsed graph workload.
//!
//! Only the fields the dataflow runtime consumes are modeled; other prompt /
//! materialization fields in the input JSON are ignored (serde skips unknown
//! fields). Every non-required field is `#[serde(default)]`, so an input that
//! omits defaulted fields still decodes.

use serde::Deserialize;
use std::collections::BTreeMap;

/// Reserved virtual entry node id (valid edge source only).
pub const START_NODE_ID: &str = "START";
/// Reserved virtual exit node id (valid edge target only).
pub const END_NODE_ID: &str = "END";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
pub enum ChannelType {
    #[serde(rename = "text")]
    #[default]
    Text,
    #[serde(rename = "messages")]
    Messages,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
pub enum ReducerName {
    #[serde(rename = "overwrite")]
    #[default]
    Overwrite,
    #[serde(rename = "add_messages")]
    AddMessages,
}

/// A state-channel declaration (`ChannelSpec`).
#[derive(Debug, Clone, Deserialize)]
pub struct ChannelSpec {
    #[serde(rename = "type", default)]
    pub channel_type: ChannelType,
    #[serde(default)]
    pub reducer: ReducerName,
}

impl Default for ChannelSpec {
    fn default() -> Self {
        ChannelSpec {
            channel_type: ChannelType::Text,
            reducer: ReducerName::Overwrite,
        }
    }
}

/// Required arrival count on a channel input: `count: int` or `count: "all"`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(untagged)]
pub enum Count {
    N(i64),
    Word(String),
}

impl Count {
    /// True for the `"all"` sentinel (resolved to the static producer count).
    pub fn is_all(&self) -> bool {
        matches!(self, Count::Word(w) if w == "all")
    }

    /// The explicit integer count, or `None` for `"all"`.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Count::N(n) => Some(*n),
            Count::Word(_) => None,
        }
    }
}

impl Default for Count {
    fn default() -> Self {
        Count::N(1)
    }
}

/// AND-fan-in input requirement on a node (`ChannelRequirement`).
#[derive(Debug, Clone, Deserialize)]
pub struct ChannelRequirement {
    pub channel: String,
    #[serde(default)]
    pub count: Count,
}

/// Unconditional edge (`StaticEdge`). The `edge_type` tag is ignored on decode.
#[derive(Debug, Clone, Deserialize)]
pub struct StaticEdge {
    pub source: String,
    pub target: String,
    #[serde(default)]
    pub delay_after_predecessor_us: Option<f64>,
    #[serde(default)]
    pub min_start_delay_us: Option<f64>,
    #[serde(default)]
    pub delay_after_predecessor_start_us: Option<f64>,
    #[serde(default)]
    pub delay_after_predecessor_first_token_us: Option<f64>,
}

/// LLM node (`LlmNode`). Prompt/materialization fields are skipped on decode.
#[derive(Debug, Clone, Deserialize)]
pub struct LlmNode {
    pub output: String,
    #[serde(default = "default_true")]
    pub streaming: bool,
    #[serde(default)]
    pub inputs: Vec<ChannelRequirement>,
    #[serde(default)]
    pub min_start_delay_us: Option<f64>,
    /// Generation cap mapped to the endpoint's token field.
    #[serde(default)]
    pub max_tokens: Option<usize>,
    /// Prompt-assembly program: static segment ids interleaved with dynamic
    /// splice slots. Empty for a node whose prompt is supplied another way.
    #[serde(default)]
    pub items: Vec<PromptItem>,
    #[serde(default)]
    pub metadata: BTreeMap<String, serde_json::Value>,
}

/// One step of a node's prompt-assembly program.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum PromptItem {
    /// A static, content-addressed segment id (walked through the SegmentStore).
    Seg { seg: String },
    /// Splice the dynamic reply captured on `splice` (a predecessor's output
    /// channel) at this position.
    Splice { splice: String },
}

fn default_true() -> bool {
    true
}

impl LlmNode {
    /// Channels this node writes: `[output]`.
    pub fn write_channels(&self) -> Vec<&str> {
        vec![self.output.as_str()]
    }
}

/// Topology record (`GraphRecord`).
#[derive(Debug, Clone, Default, Deserialize)]
pub struct GraphRecord {
    #[serde(default)]
    pub version: Option<String>,
    #[serde(default)]
    pub system: Option<String>,
    #[serde(default)]
    pub state: BTreeMap<String, ChannelSpec>,
    #[serde(default)]
    pub nodes: BTreeMap<String, LlmNode>,
    #[serde(default)]
    pub edges: Vec<StaticEdge>,
}

/// Per-trace data (`TraceRecord`).
#[derive(Debug, Clone, Deserialize)]
pub struct TraceRecord {
    pub id: String,
    #[serde(default)]
    pub graph_ref: Option<String>,
    #[serde(default)]
    pub initial_state: BTreeMap<String, serde_json::Value>,
}

/// Parsed graph workload (`ParsedGraph`).
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ParsedGraph {
    #[serde(default)]
    pub graph: GraphRecord,
    #[serde(default)]
    pub graphs: BTreeMap<String, GraphRecord>,
    #[serde(default)]
    pub traces: Vec<TraceRecord>,
    /// Present (non-null) iff a segment pool backs this workload. Only its
    /// presence matters to the runtime (`_handle_node_exception` sentinel
    /// branch); the harness emits a truthy placeholder.
    #[serde(default)]
    pub segment_pool: Option<serde_json::Value>,
}

impl ParsedGraph {
    /// True when a segment pool backs this workload (`segment_pool is not None`).
    pub fn has_segment_pool(&self) -> bool {
        self.segment_pool.is_some()
    }

    /// The top-level graph a trace runs against (`resolve_trace_graph`).
    pub fn resolve_trace_graph(&self, trace: &TraceRecord) -> &GraphRecord {
        match &trace.graph_ref {
            None => &self.graph,
            Some(reference) => self.graphs.get(reference).unwrap_or(&self.graph),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialize_minimal_graph() {
        let json = r#"{
            "graph": {
                "state": {"out": {"type": "text", "reducer": "overwrite"}},
                "nodes": {"n0": {"node_type": "llm", "prompt": ["hi"], "output": "out"}},
                "edges": [
                    {"edge_type": "static", "source": "START", "target": "n0"},
                    {"edge_type": "static", "source": "n0", "target": "END"}
                ]
            },
            "traces": [{"id": "t-1", "initial_state": {}}]
        }"#;
        let parsed: ParsedGraph = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.traces.len(), 1);
        assert_eq!(parsed.traces[0].id, "t-1");
        let node = parsed.graph.nodes.get("n0").unwrap();
        assert_eq!(node.output, "out");
        assert!(node.streaming);
        assert_eq!(node.write_channels(), vec!["out"]);
        assert_eq!(parsed.graph.edges.len(), 2);
        assert!(!parsed.has_segment_pool());
    }

    #[test]
    fn count_all_and_default() {
        let req: ChannelRequirement =
            serde_json::from_str(r#"{"channel": "c", "count": "all"}"#).unwrap();
        assert!(req.count.is_all());
        let req2: ChannelRequirement = serde_json::from_str(r#"{"channel": "c"}"#).unwrap();
        assert_eq!(req2.count.as_int(), Some(1));
        let req3: ChannelRequirement =
            serde_json::from_str(r#"{"channel": "c", "count": 3}"#).unwrap();
        assert_eq!(req3.count.as_int(), Some(3));
    }
}
