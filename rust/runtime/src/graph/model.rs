// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Graph-IR data model: the runtime-relevant subset of a parsed graph workload.
//!
//! Only the fields the dataflow runtime consumes are modeled; other prompt /
//! materialization fields in the input JSON are ignored (serde skips unknown
//! fields). Every non-required field is `#[serde(default)]`, so an input that
//! omits defaulted fields still decodes.

use crate::dataset::Handle;
use crate::graph::driver::{ReplayTraceMetadata, TraceDriverSpec, TraceEnvironmentSpec};
use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize};
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

impl std::fmt::Display for Count {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::N(count) => write!(f, "{count}"),
            Self::Word(word) => write!(f, "{word:?}"),
        }
    }
}

/// Why an authored channel count cannot be used by the graph runtime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CountValidationError {
    /// Integer counts must not be negative.
    Negative(i64),
    /// The target platform cannot represent the integer count.
    OutOfRange(i64),
    /// The only supported word count is the `"all"` sentinel.
    UnknownWord(String),
}

impl std::fmt::Display for CountValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Negative(count) => write!(f, "{count} (counts must be non-negative)"),
            Self::OutOfRange(count) => write!(f, "{count} (count is too large for this platform)"),
            Self::UnknownWord(word) => write!(f, "{word:?} (expected an integer or \"all\")"),
        }
    }
}

impl std::error::Error for CountValidationError {}

impl Count {
    /// Resolve an authored count, retaining `None` for the `"all"` sentinel.
    pub fn validate(&self) -> Result<Option<usize>, CountValidationError> {
        match self {
            Self::N(count) if *count >= 0 => usize::try_from(*count)
                .map(Some)
                .map_err(|_| CountValidationError::OutOfRange(*count)),
            Self::N(count) => Err(CountValidationError::Negative(*count)),
            Self::Word(word) if word == "all" => Ok(None),
            Self::Word(word) => Err(CountValidationError::UnknownWord(word.clone())),
        }
    }

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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
    /// Optional request fields retained separately from prompt construction.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request: Option<LlmRequestSpec>,
    #[serde(default)]
    pub metadata: BTreeMap<String, serde_json::Value>,
}

/// Optional, request-level fields retained with an LLM graph node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmRequestSpec {
    /// Serialized tools payload selected for this call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Handle>,
    /// Explicit model override for this call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Serialized additional request-body payload.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub additional_body: Option<Handle>,
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
    pub fn write_channels(&self) -> std::iter::Once<&str> {
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

/// Predetermined tool work carried between LLM graph turns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolNode {
    /// Channel receiving this tool invocation's observation.
    pub output: String,
    /// Completed source commands to replay in order.
    pub commands: Vec<String>,
    /// Optional upper bound for the complete tool node.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_ns: Option<u64>,
}

/// One executable graph node.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExecutableGraphNode {
    /// A measured inference request.
    Llm(LlmNode),
    /// A non-inference tool observation producer.
    Tool(ToolNode),
}

impl<'de> Deserialize<'de> for ExecutableGraphNode {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        let Some(object) = value.as_object() else {
            return Err(D::Error::custom("graph node must be an object"));
        };
        match object.get("kind") {
            None => serde_json::from_value(value)
                .map(ExecutableGraphNode::Llm)
                .map_err(D::Error::custom),
            Some(serde_json::Value::String(kind)) if kind == "llm" => serde_json::from_value(value)
                .map(ExecutableGraphNode::Llm)
                .map_err(D::Error::custom),
            Some(serde_json::Value::String(kind)) if kind == "tool" => {
                serde_json::from_value(value)
                    .map(ExecutableGraphNode::Tool)
                    .map_err(D::Error::custom)
            }
            Some(serde_json::Value::String(kind)) => Err(D::Error::custom(format!(
                "unknown graph node kind {kind:?}"
            ))),
            Some(_) => Err(D::Error::custom("graph node kind must be a string")),
        }
    }
}

impl ExecutableGraphNode {
    /// The channel written by this node.
    pub fn output(&self) -> &str {
        match self {
            ExecutableGraphNode::Llm(node) => &node.output,
            ExecutableGraphNode::Tool(node) => &node.output,
        }
    }

    /// Channels this node observes before it can run.
    pub fn read_channels(&self) -> Vec<&str> {
        match self {
            ExecutableGraphNode::Llm(node) => node
                .inputs
                .iter()
                .map(|requirement| requirement.channel.as_str())
                .chain(node.splice_channels())
                .collect(),
            ExecutableGraphNode::Tool(_) => Vec::new(),
        }
    }

    /// Input requirements used by static firing gates.
    pub fn input_requirements(&self) -> &[ChannelRequirement] {
        match self {
            ExecutableGraphNode::Llm(node) => &node.inputs,
            ExecutableGraphNode::Tool(_) => &[],
        }
    }

    /// Channels this node produces once it completes.
    pub fn write_channels(&self) -> std::iter::Once<&str> {
        match self {
            ExecutableGraphNode::Llm(node) => node.write_channels(),
            ExecutableGraphNode::Tool(node) => std::iter::once(node.output.as_str()),
        }
    }

    /// Static inference credits consumed by this node.
    pub const fn static_request_count(&self) -> u64 {
        match self {
            ExecutableGraphNode::Llm(_) => 1,
            ExecutableGraphNode::Tool(_) => 0,
        }
    }

    /// Borrow the inference node when this is an LLM node.
    pub const fn as_llm(&self) -> Option<&LlmNode> {
        match self {
            ExecutableGraphNode::Llm(node) => Some(node),
            ExecutableGraphNode::Tool(_) => None,
        }
    }

    /// Borrow LLM metadata when this is an inference node.
    pub fn metadata(&self) -> Option<&BTreeMap<String, serde_json::Value>> {
        self.as_llm().map(|node| &node.metadata)
    }

    /// Borrow the inference node mutably when this is an LLM node.
    pub fn as_llm_mut(&mut self) -> Option<&mut LlmNode> {
        match self {
            ExecutableGraphNode::Llm(node) => Some(node),
            ExecutableGraphNode::Tool(_) => None,
        }
    }
}

impl From<LlmNode> for ExecutableGraphNode {
    fn from(node: LlmNode) -> Self {
        Self::Llm(node)
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
    pub nodes: BTreeMap<String, ExecutableGraphNode>,
    #[serde(default)]
    pub edges: Vec<StaticEdge>,
}

impl GraphRecord {
    /// Number of static inference requests in this topology.
    pub fn llm_node_count(&self) -> usize {
        self.nodes
            .values()
            .filter(|node| matches!(node, ExecutableGraphNode::Llm(_)))
            .count()
    }

    /// Number of all executable graph nodes, including tool nodes.
    pub fn total_node_count(&self) -> usize {
        self.nodes.len()
    }
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

/// Complete placement-owned trace command including optional replay context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphTraceProgram {
    /// Profiling graph trace.
    pub profiling: GraphTracePlan,
    /// Optional trace-local warmup graph.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub warmup: Option<GraphTracePlan>,
    /// Optional selected environment recipe.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub environment: Option<TraceEnvironmentSpec>,
    /// Optional recorded-replay source facts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replay: Option<ReplayTraceMetadata>,
    /// Registered trace program driver.
    pub driver: TraceDriverSpec,
}

impl GraphTraceProgram {
    /// Wrap a generic graph trace in the built-in static driver.
    pub fn static_graph(profiling: GraphTracePlan) -> Self {
        Self {
            profiling,
            warmup: None,
            environment: None,
            replay: None,
            driver: TraceDriverSpec::static_graph(),
        }
    }

    /// Whether this program can use the legacy static graph executor.
    pub fn is_static_graph_program(&self) -> bool {
        self.driver.is_static_graph()
            && self.warmup.is_none()
            && self.environment.is_none()
            && self.replay.is_none()
    }
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
        let node = node.as_llm().unwrap();
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

    #[test]
    fn graph_trace_program_serde_decodes_legacy_llm_nodes_and_rejects_unknown_node_kinds() {
        let legacy = r#"{
            "nodes": {"call": {"output": "reply"}},
            "edges": []
        }"#;
        let graph: GraphRecord = serde_json::from_str(legacy).unwrap();
        assert!(matches!(graph.nodes["call"], ExecutableGraphNode::Llm(_)));

        let unknown = r#"{
            "nodes": {"call": {"kind": "unsupported", "output": "reply"}},
            "edges": []
        }"#;
        assert!(serde_json::from_str::<GraphRecord>(unknown).is_err());
    }

    #[test]
    fn graph_trace_program_tool_node_writes_a_channel_without_consuming_request_credit() {
        let graph = GraphRecord {
            nodes: BTreeMap::from([
                (
                    "call".into(),
                    ExecutableGraphNode::Llm(LlmNode {
                        output: "reply".into(),
                        streaming: true,
                        inputs: Vec::new(),
                        min_start_delay_us: None,
                        max_tokens: Some(1),
                        items: Vec::new(),
                        request: None,
                        metadata: BTreeMap::new(),
                    }),
                ),
                (
                    "tool".into(),
                    ExecutableGraphNode::Tool(ToolNode {
                        output: "observation".into(),
                        commands: vec!["pwd".into()],
                        timeout_ns: None,
                    }),
                ),
            ]),
            ..GraphRecord::default()
        };

        assert_eq!(graph.llm_node_count(), 1);
        assert_eq!(graph.total_node_count(), 2);
        assert_eq!(
            graph.nodes["tool"].write_channels().collect::<Vec<_>>(),
            vec!["observation"]
        );
        assert_eq!(graph.nodes["tool"].static_request_count(), 0);
    }
}
