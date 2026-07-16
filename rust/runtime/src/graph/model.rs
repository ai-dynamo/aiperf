// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Graph-IR data model: the runtime-relevant subset of a parsed graph workload.
//!
//! Only the fields the dataflow runtime consumes are modeled; other prompt /
//! materialization fields in the input JSON are ignored (serde skips unknown
//! fields). Every non-required field is `#[serde(default)]`, so an input that
//! omits defaulted fields still decodes.

use crate::dataset::Handle;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Reserved virtual entry node id (valid edge source only).
pub const START_NODE_ID: &str = "START";
/// Reserved virtual exit node id (valid edge target only).
pub const END_NODE_ID: &str = "END";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ChannelType {
    #[serde(rename = "text")]
    #[default]
    Text,
    #[serde(rename = "messages")]
    Messages,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ReducerName {
    #[serde(rename = "overwrite")]
    #[default]
    Overwrite,
    #[serde(rename = "add_messages")]
    AddMessages,
}

/// A state-channel declaration (`ChannelSpec`).
///
/// The derived `Default` matches the field defaults exactly: `ChannelType`
/// defaults to `Text` and `ReducerName` to `Overwrite`.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ChannelSpec {
    #[serde(rename = "type", default)]
    pub channel_type: ChannelType,
    #[serde(default)]
    pub reducer: ReducerName,
}

/// Required arrival count on a channel input: `count: int` or `count: "all"`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelRequirement {
    pub channel: String,
    #[serde(default)]
    pub count: Count,
}

/// Unconditional edge (`StaticEdge`). The `edge_type` tag is ignored on decode.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PromptItem {
    /// A static, dense segment handle (resolved through the shared SegmentStore).
    Seg { seg: Handle },
    /// An authored pre-serialized message array retained as one raw segment.
    ///
    /// Dataset formats such as `dag_jsonl` deliberately retain the complete
    /// array wire. The graph materializer splits it into its exact object
    /// slices once per prompt without inventing a second segment arena.
    RawMessages { raw_messages: Handle },
    /// A text-only dataset segment projected into a message with `role`.
    ///
    /// Shared system and user-context prompts use the text segment domain, so
    /// this variant lets graph lowering reuse their real dense handles.
    Text { text: Handle, role: String },
    /// Splice the dynamic reply captured on `splice` (a predecessor's output
    /// channel) at this position.
    Splice { splice: String },
}

fn default_true() -> bool {
    true
}

impl LlmNode {
    /// Channels this node writes: `[output]`.
    ///
    /// Returns an iterator so per-fire callers (`finalize_node`) do not heap-
    /// allocate a single-element `Vec` on the completion path.
    pub fn write_channels(&self) -> impl Iterator<Item = &str> {
        std::iter::once(self.output.as_str())
    }

    /// Dynamic channel keys this node's prompt splices in (`PromptItem::Splice`).
    ///
    /// These are the *only* channels a node reads while materializing its prompt:
    /// `Seg`/`RawMessages`/`Text` items resolve through the segment store and never
    /// consult channel state. The executor uses this to reduce just these channels
    /// from the gated snapshot instead of the whole store (per-node materialize is
    /// otherwise O(channels × history) in a wide graph).
    pub fn splice_channels(&self) -> Vec<&str> {
        self.items
            .iter()
            .filter_map(|item| match item {
                PromptItem::Splice { splice } => Some(splice.as_str()),
                _ => None,
            })
            .collect()
    }
}

/// Topology record (`GraphRecord`).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceRecord {
    pub id: String,
    #[serde(default)]
    pub graph_ref: Option<String>,
    #[serde(default)]
    pub initial_state: BTreeMap<String, serde_json::Value>,
}

/// Complete data-plane command for one root trace.
///
/// Placement boundaries move this value as one unit. They never distribute
/// individual node turns, because fan-out, joins, firing gates, and dynamic
/// reply splices are trace-local state owned by one executor. Dense segment
/// handles refer to the immutable segment catalog installed in the selected
/// backend; a remote implementation can provision that catalog once and send
/// this serde-ready command for every trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphTracePlan {
    /// Resolved graph for this trace instance.
    pub graph: GraphRecord,
    /// Per-trace identity and initial channel state.
    pub trace: TraceRecord,
    /// Optional arrival offset from workload start.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arrival_offset_ns: Option<i64>,
}

/// Parsed graph workload (`ParsedGraph`).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ParsedGraph {
    #[serde(default)]
    pub graph: GraphRecord,
    #[serde(default)]
    pub graphs: BTreeMap<String, GraphRecord>,
    #[serde(default)]
    pub traces: Vec<TraceRecord>,
}

impl ParsedGraph {
    /// The top-level graph a trace runs against (`resolve_trace_graph`).
    pub fn resolve_trace_graph(&self, trace: &TraceRecord) -> &GraphRecord {
        match &trace.graph_ref {
            None => &self.graph,
            Some(reference) => self.graphs.get(reference).unwrap_or_else(|| {
                // A named graph_ref that resolves to nothing is a config error;
                // the fallback keeps the trace runnable but must not do so
                // silently, or a mistyped ref masquerades as the default graph.
                tracing::warn!(
                    graph_ref = %reference,
                    "unknown graph_ref; falling back to top-level graph"
                );
                &self.graph
            }),
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
        assert_eq!(node.write_channels().collect::<Vec<_>>(), vec!["out"]);
        assert_eq!(parsed.graph.edges.len(), 2);
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
