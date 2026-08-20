// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Authored conditional-graph document model and strict YAML/JSON decode.
//!
//! The authored format is a pre-IR surface: a `graph` block (state channels,
//! LLM and replay nodes, static and conditional edges) plus a `traces` block
//! (per-trace pinned branches, recorded replay outputs, and initial channel
//! state). It is decoded strictly — a raw layer with `deny_unknown_fields`
//! rejects typos, then an explicit conversion enforces per-node-kind field rules
//! that serde cannot express on a tagged enum. Nothing here executes; the
//! compiler in the parent module resolves, prunes, and folds each trace into the
//! flat Graph-IR.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{self, Display};

use serde::Deserialize;
use serde_json::Value;

use crate::graph::model::{ChannelRequirement, ChannelType, Count, ReducerName};

/// Focused authored-graph parse or lowering failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConditionalError(pub String);

impl ConditionalError {
    pub fn message(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl Display for ConditionalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for ConditionalError {}

impl From<serde_json::Error> for ConditionalError {
    fn from(error: serde_json::Error) -> Self {
        Self::message(error.to_string())
    }
}

impl From<serde_yaml::Error> for ConditionalError {
    fn from(error: serde_yaml::Error) -> Self {
        Self::message(error.to_string())
    }
}

// ---------------------------------------------------------------------------
// Clean model (the compiler consumes these)
// ---------------------------------------------------------------------------

/// A decoded authored conditional-graph document.
#[derive(Debug, Clone, PartialEq)]
pub struct AuthoredGraphDoc {
    pub graph: AuthoredGraph,
    pub traces: Vec<AuthoredTrace>,
}

/// The authored graph topology, shared by every trace before resolution.
#[derive(Debug, Clone, PartialEq)]
pub struct AuthoredGraph {
    pub state: BTreeMap<String, AuthoredChannelSpec>,
    pub nodes: BTreeMap<String, AuthoredNode>,
    pub edges: Vec<AuthoredEdge>,
}

/// A state-channel declaration. Typed authored kinds (`json`/`image`) collapse
/// to the flat core's `text`/`messages` — no runtime type is preserved.
#[derive(Debug, Clone, PartialEq)]
pub struct AuthoredChannelSpec {
    pub channel_type: ChannelType,
    pub reducer: ReducerName,
}

/// An authored node: a dispatching LLM node or a non-dispatching replay node.
#[derive(Debug, Clone, PartialEq)]
pub enum AuthoredNode {
    Llm(AuthoredLlmNode),
    Replay(AuthoredReplayNode),
}

/// A dispatching LLM node.
#[derive(Debug, Clone, PartialEq)]
pub struct AuthoredLlmNode {
    pub prompt: Vec<PromptGrammarItem>,
    pub output: String,
    pub inputs: Vec<ChannelRequirement>,
    pub streaming: bool,
    pub endpoint: Option<String>,
    pub max_tokens: Option<usize>,
    pub metadata: BTreeMap<String, Value>,
    pub terminal_for_user: bool,
    pub min_start_delay_us: Option<f64>,
}

/// A non-dispatching node whose recorded `outputs` fold into `initial_state`.
#[derive(Debug, Clone, PartialEq)]
pub struct AuthoredReplayNode {
    pub outputs: Vec<String>,
    pub duration_ms: f64,
    pub metadata: BTreeMap<String, Value>,
    pub min_start_delay_us: Option<f64>,
}

/// One authored edge: unconditional or model-independent conditional.
#[derive(Debug, Clone, PartialEq)]
pub enum AuthoredEdge {
    Static(AuthoredStaticEdge),
    Conditional(AuthoredConditionalEdge),
}

/// An unconditional edge; mirrors `graph::model::StaticEdge` delay anchors.
#[derive(Debug, Clone, PartialEq)]
pub struct AuthoredStaticEdge {
    pub source: String,
    pub target: String,
    pub delay_after_predecessor_us: Option<f64>,
    pub min_start_delay_us: Option<f64>,
    pub delay_after_predecessor_start_us: Option<f64>,
    pub delay_after_predecessor_first_token_us: Option<f64>,
}

/// A conditional edge whose taken branch is resolved per trace at lowering.
#[derive(Debug, Clone, PartialEq)]
pub struct AuthoredConditionalEdge {
    pub source: String,
    pub branches: BTreeMap<String, BranchTargets>,
    pub branch_weights: Option<BTreeMap<String, f64>>,
    pub delay_after_predecessor_us: Option<f64>,
    pub min_start_delay_us: Option<f64>,
}

/// One or more successor ids reached by a branch key.
#[derive(Debug, Clone, PartialEq)]
pub enum BranchTargets {
    One(String),
    Many(Vec<String>),
}

impl BranchTargets {
    /// The successor ids for this branch, in authored order.
    pub fn targets(&self) -> Vec<&str> {
        match self {
            BranchTargets::One(target) => vec![target.as_str()],
            BranchTargets::Many(targets) => targets.iter().map(String::as_str).collect(),
        }
    }
}

/// One authored prompt-grammar item.
#[derive(Debug, Clone, PartialEq)]
pub enum PromptGrammarItem {
    /// A bare `@channel` reference: splice the channel's value at this position.
    ChannelRef(String),
    /// Literal text projected into a message with the default `user` role.
    Text(String),
    /// A `{role, content}` message; content parts may themselves be `@channel`
    /// references interleaved with literal text.
    Message {
        role: String,
        content: Vec<MessagePart>,
    },
}

/// One part of a `{role, content}` message.
#[derive(Debug, Clone, PartialEq)]
pub enum MessagePart {
    /// A `@channel` splice reference.
    ChannelRef(String),
    /// Literal text.
    Text(String),
}

/// Per-trace resolution inputs and recorded state.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthoredTrace {
    pub id: String,
    #[serde(default)]
    pub initial_state: BTreeMap<String, Value>,
    #[serde(default)]
    pub selected_branches: BTreeMap<String, String>,
    #[serde(default)]
    pub branch_distributions: Option<BTreeMap<String, BTreeMap<String, f64>>>,
    #[serde(default)]
    pub replay_outputs: BTreeMap<String, BTreeMap<String, Value>>,
    #[serde(default)]
    pub arrival_time: Option<f64>,
    #[serde(default)]
    pub tags: Vec<String>,
}

// ---------------------------------------------------------------------------
// Raw decode layer (strict; converted into the clean model)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RawDoc {
    graph: RawGraph,
    #[serde(default)]
    traces: Vec<AuthoredTrace>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawGraph {
    #[serde(default)]
    state: BTreeMap<String, RawChannelSpec>,
    #[serde(default)]
    nodes: BTreeMap<String, RawNode>,
    #[serde(default)]
    edges: Vec<RawEdge>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawChannelSpec {
    #[serde(rename = "type", default)]
    channel_type: Option<String>,
    #[serde(default)]
    reducer: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawNode {
    #[serde(default = "default_node_type")]
    node_type: String,
    #[serde(default)]
    prompt: Option<Vec<PromptGrammarItem>>,
    #[serde(default)]
    output: Option<String>,
    #[serde(default)]
    inputs: RawInputs,
    #[serde(default)]
    outputs: Option<Vec<String>>,
    #[serde(default)]
    streaming: Option<bool>,
    #[serde(default)]
    endpoint: Option<String>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    duration_ms: Option<f64>,
    #[serde(default)]
    metadata: BTreeMap<String, Value>,
    #[serde(default)]
    terminal_for_user: bool,
    #[serde(default)]
    min_start_delay_us: Option<f64>,
    // Accepted (engine-prediction comparators in authored fixtures) but not
    // consumed by lowering; declared so strict decode does not reject it.
    #[serde(default)]
    #[allow(dead_code)]
    expected: Option<Value>,
}

fn default_node_type() -> String {
    "llm".to_string()
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawChannelRequirement {
    channel: String,
    #[serde(default = "default_raw_channel_count")]
    count: RawChannelCount,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum RawChannelCount {
    N(i64),
    Word(String),
}

fn default_raw_channel_count() -> RawChannelCount {
    RawChannelCount::N(1)
}

#[derive(Default)]
enum RawInputs {
    #[default]
    Absent,
    Present(Vec<RawChannelRequirement>),
}

impl<'de> Deserialize<'de> for RawInputs {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Vec::<RawChannelRequirement>::deserialize(deserializer).map(Self::Present)
    }
}

impl RawInputs {
    fn into_llm_requirements(self) -> Vec<RawChannelRequirement> {
        match self {
            Self::Absent => Vec::new(),
            Self::Present(inputs) => inputs,
        }
    }

    fn is_present(&self) -> bool {
        matches!(self, Self::Present(_))
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawEdge {
    source: String,
    #[serde(default)]
    target: Option<String>,
    #[serde(default)]
    branches: Option<BTreeMap<String, BranchTargets>>,
    #[serde(default)]
    branch_weights: Option<BTreeMap<String, f64>>,
    #[serde(default)]
    delay_after_predecessor_us: Option<f64>,
    #[serde(default)]
    min_start_delay_us: Option<f64>,
    #[serde(default)]
    delay_after_predecessor_start_us: Option<f64>,
    #[serde(default)]
    delay_after_predecessor_first_token_us: Option<f64>,
}

// `BranchTargets` and the prompt grammar decode as untagged shapes: a scalar vs
// a sequence, and a scalar vs a `{role, content}` map. Untagged is correct here
// because the shapes are disjoint by JSON/YAML type; the strict field checks
// that `deny_unknown_fields` provides live on the enclosing structs.
impl<'de> Deserialize<'de> for BranchTargets {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Raw {
            One(String),
            Many(Vec<String>),
        }
        Ok(match Raw::deserialize(deserializer)? {
            Raw::One(target) => BranchTargets::One(target),
            Raw::Many(targets) => BranchTargets::Many(targets),
        })
    }
}

impl<'de> Deserialize<'de> for PromptGrammarItem {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Raw {
            Scalar(String),
            Message { role: String, content: RawContent },
        }
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum RawContent {
            One(String),
            Many(Vec<String>),
        }
        Ok(match Raw::deserialize(deserializer)? {
            Raw::Scalar(text) => scalar_prompt_item(text),
            Raw::Message { role, content } => {
                let parts = match content {
                    RawContent::One(text) => vec![message_part(text)],
                    RawContent::Many(texts) => texts.into_iter().map(message_part).collect(),
                };
                PromptGrammarItem::Message {
                    role,
                    content: parts,
                }
            }
        })
    }
}

fn scalar_prompt_item(text: String) -> PromptGrammarItem {
    match text.strip_prefix('@') {
        Some(channel) => PromptGrammarItem::ChannelRef(channel.to_string()),
        None => PromptGrammarItem::Text(text),
    }
}

fn message_part(text: String) -> MessagePart {
    match text.strip_prefix('@') {
        Some(channel) => MessagePart::ChannelRef(channel.to_string()),
        None => MessagePart::Text(text),
    }
}

// ---------------------------------------------------------------------------
// Parse entry point + raw → clean conversion
// ---------------------------------------------------------------------------

/// Strictly decode an authored conditional-graph document from YAML or JSON.
///
/// YAML is a superset of JSON, so one `serde_yaml` pass accepts both wire forms.
pub fn parse_authored_graph(bytes: &[u8]) -> Result<AuthoredGraphDoc, ConditionalError> {
    convert_authored_graph(decode_authored_graph(bytes).map_err(ConditionalError::from)?)
}

pub(super) fn decode_authored_graph(bytes: &[u8]) -> Result<RawDoc, serde_yaml::Error> {
    serde_yaml::from_slice(bytes)
}

pub(super) fn convert_authored_graph(raw: RawDoc) -> Result<AuthoredGraphDoc, ConditionalError> {
    let mut nodes = BTreeMap::new();
    for (id, node) in raw.graph.nodes {
        nodes.insert(id.clone(), convert_node(&id, node)?);
    }
    let mut state = BTreeMap::new();
    for (name, spec) in raw.graph.state {
        state.insert(name.clone(), convert_channel_spec(&name, spec)?);
    }
    let edges = raw
        .graph
        .edges
        .into_iter()
        .map(convert_edge)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(AuthoredGraphDoc {
        graph: AuthoredGraph {
            state,
            nodes,
            edges,
        },
        traces: raw.traces,
    })
}

fn convert_channel_spec(
    name: &str,
    spec: RawChannelSpec,
) -> Result<AuthoredChannelSpec, ConditionalError> {
    let channel_type = match spec.channel_type.as_deref() {
        None | Some("text") | Some("json") => ChannelType::Text,
        Some("messages") => ChannelType::Messages,
        // Image content rides as message parts / segment bytes; there is no
        // runtime image channel type to preserve.
        Some("image") => ChannelType::Messages,
        Some(other) => {
            return Err(ConditionalError::message(format!(
                "channel {name:?} has unsupported type {other:?} (text|messages|json|image)"
            )));
        }
    };
    let reducer = match spec.reducer.as_deref() {
        None | Some("overwrite") => ReducerName::Overwrite,
        Some("add_messages") => ReducerName::AddMessages,
        Some(other) => {
            return Err(ConditionalError::message(format!(
                "channel {name:?} has unsupported reducer {other:?} (overwrite|add_messages)"
            )));
        }
    };
    Ok(AuthoredChannelSpec {
        channel_type,
        reducer,
    })
}

fn convert_node(id: &str, node: RawNode) -> Result<AuthoredNode, ConditionalError> {
    match node.node_type.as_str() {
        "llm" => {
            if node.outputs.is_some() {
                return Err(ConditionalError::message(format!(
                    "llm node {id:?} must use scalar `output`, not `outputs`"
                )));
            }
            if node.duration_ms.is_some() {
                return Err(ConditionalError::message(format!(
                    "llm node {id:?} cannot set `duration_ms` (replay-only)"
                )));
            }
            let prompt = node.prompt.ok_or_else(|| {
                ConditionalError::message(format!("llm node {id:?} is missing `prompt`"))
            })?;
            let output = node.output.ok_or_else(|| {
                ConditionalError::message(format!("llm node {id:?} is missing `output`"))
            })?;
            Ok(AuthoredNode::Llm(AuthoredLlmNode {
                prompt,
                output,
                inputs: convert_input_requirements(id, node.inputs.into_llm_requirements())?,
                streaming: node.streaming.unwrap_or(true),
                endpoint: node.endpoint,
                max_tokens: node.max_tokens,
                metadata: node.metadata,
                terminal_for_user: node.terminal_for_user,
                min_start_delay_us: node.min_start_delay_us,
            }))
        }
        "replay" => {
            if node.inputs.is_present() {
                return Err(ConditionalError::message(format!(
                    "replay node {id:?} cannot set `inputs`"
                )));
            }
            if node.prompt.is_some() {
                return Err(ConditionalError::message(format!(
                    "replay node {id:?} cannot set `prompt`"
                )));
            }
            if node.output.is_some() {
                return Err(ConditionalError::message(format!(
                    "replay node {id:?} must use `outputs`, not scalar `output`"
                )));
            }
            let outputs = node.outputs.ok_or_else(|| {
                ConditionalError::message(format!("replay node {id:?} is missing `outputs`"))
            })?;
            if outputs.is_empty() {
                return Err(ConditionalError::message(format!(
                    "replay node {id:?} declares no output channels"
                )));
            }
            Ok(AuthoredNode::Replay(AuthoredReplayNode {
                outputs,
                duration_ms: node.duration_ms.unwrap_or(0.0),
                metadata: node.metadata,
                min_start_delay_us: node.min_start_delay_us,
            }))
        }
        other => Err(ConditionalError::message(format!(
            "node {id:?} has unknown node_type {other:?} (llm|replay)"
        ))),
    }
}

fn convert_input_requirements(
    node_id: &str,
    inputs: Vec<RawChannelRequirement>,
) -> Result<Vec<ChannelRequirement>, ConditionalError> {
    inputs
        .into_iter()
        .enumerate()
        .map(|(index, input)| {
            let count = match input.count {
                RawChannelCount::N(count) => Count::N(count),
                RawChannelCount::Word(word) if word == "all" => Count::Word(word),
                RawChannelCount::Word(word) => {
                    return Err(ConditionalError::message(format!(
                        "llm node {node_id:?} inputs[{index}].count must be an integer or \"all\", got {word:?}"
                    )));
                }
            };
            Ok(ChannelRequirement {
                channel: input.channel,
                count,
            })
        })
        .collect()
}

fn convert_edge(edge: RawEdge) -> Result<AuthoredEdge, ConditionalError> {
    match (edge.target, edge.branches) {
        (Some(target), None) => Ok(AuthoredEdge::Static(AuthoredStaticEdge {
            source: edge.source,
            target,
            delay_after_predecessor_us: edge.delay_after_predecessor_us,
            min_start_delay_us: edge.min_start_delay_us,
            delay_after_predecessor_start_us: edge.delay_after_predecessor_start_us,
            delay_after_predecessor_first_token_us: edge.delay_after_predecessor_first_token_us,
        })),
        (None, Some(branches)) => {
            if branches.is_empty() {
                return Err(ConditionalError::message(format!(
                    "conditional edge from {:?} declares no branches",
                    edge.source
                )));
            }
            Ok(AuthoredEdge::Conditional(AuthoredConditionalEdge {
                source: edge.source,
                branches,
                branch_weights: edge.branch_weights,
                delay_after_predecessor_us: edge.delay_after_predecessor_us,
                min_start_delay_us: edge.min_start_delay_us,
            }))
        }
        (Some(_), Some(_)) => Err(ConditionalError::message(format!(
            "edge from {:?} sets both `target` and `branches`",
            edge.source
        ))),
        (None, None) => Err(ConditionalError::message(format!(
            "edge from {:?} sets neither `target` nor `branches`",
            edge.source
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DOC: &str = r#"
graph:
  state:
    messages: {type: messages, reducer: add_messages}
    intent: {type: text}
    raw_results: {type: json}
  nodes:
    route:
      node_type: llm
      prompt:
        - {role: system, content: "Classify intent."}
        - "@messages"
      output: intent
      streaming: false
    tool_exec:
      node_type: replay
      outputs: [raw_results, brand_cands]
      duration_ms: 80
  edges:
    - {source: START, target: route}
    - {source: route, branches: {shopping: plan, non_shopping: END}}
traces:
  - id: t-shopping
    selected_branches: {route: shopping}
    initial_state: {query_text: "find cheaper"}
    replay_outputs:
      tool_exec: {raw_results: [], brand_cands: "Nike, Adidas"}
"#;

    #[test]
    fn public_error_tuple_source_compatibility() {
        let conditional = ConditionalError("conditional".to_owned());
        assert_eq!(conditional.0, "conditional");
    }

    #[test]
    fn decodes_minimal_conditional_doc() {
        let doc = parse_authored_graph(DOC.as_bytes()).unwrap();
        assert_eq!(doc.traces.len(), 1);
        assert_eq!(doc.traces[0].id, "t-shopping");
        assert_eq!(
            doc.traces[0]
                .selected_branches
                .get("route")
                .map(String::as_str),
            Some("shopping")
        );
        // route is an LLM node with a static prompt message then a channel ref.
        let AuthoredNode::Llm(route) = &doc.graph.nodes["route"] else {
            panic!("route must be an llm node");
        };
        assert_eq!(route.output, "intent");
        assert!(!route.streaming);
        assert_eq!(route.prompt.len(), 2);
        assert!(matches!(route.prompt[1], PromptGrammarItem::ChannelRef(ref c) if c == "messages"));
        // tool_exec is a replay node with two output channels.
        let AuthoredNode::Replay(tool) = &doc.graph.nodes["tool_exec"] else {
            panic!("tool_exec must be a replay node");
        };
        assert_eq!(tool.outputs, vec!["raw_results", "brand_cands"]);
        assert_eq!(tool.duration_ms, 80.0);
        // json channel collapses to Text.
        assert_eq!(
            doc.graph.state["raw_results"].channel_type,
            ChannelType::Text
        );
        assert_eq!(
            doc.graph.state["messages"].channel_type,
            ChannelType::Messages
        );
    }

    #[test]
    fn static_vs_conditional_edge_discrimination() {
        let doc = parse_authored_graph(DOC.as_bytes()).unwrap();
        assert!(matches!(&doc.graph.edges[0], AuthoredEdge::Static(e) if e.target == "route"));
        let AuthoredEdge::Conditional(cond) = &doc.graph.edges[1] else {
            panic!("second edge must be conditional");
        };
        assert_eq!(cond.source, "route");
        assert_eq!(cond.branches["shopping"], BranchTargets::One("plan".into()));
        assert_eq!(
            cond.branches["non_shopping"],
            BranchTargets::One("END".into())
        );
    }

    #[test]
    fn rejects_unknown_top_level_field() {
        let bad = format!("{DOC}\nforeign: true\n");
        let err = parse_authored_graph(bad.as_bytes()).unwrap_err();
        assert!(err.to_string().contains("foreign") || err.to_string().contains("unknown"));
    }

    #[test]
    fn rejects_unknown_node_field() {
        let bad = DOC.replace("streaming: false", "streaming: false\n      bogus_field: 1");
        let err = parse_authored_graph(bad.as_bytes()).unwrap_err();
        assert!(err.to_string().contains("bogus_field") || err.to_string().contains("unknown"));
    }

    #[test]
    fn rejects_unknown_llm_input_field() {
        let bad = DOC.replace(
            "output: intent",
            "output: intent\n      inputs: [{channel: messages, count: 1, typo: true}]",
        );
        let err = parse_authored_graph(bad.as_bytes()).unwrap_err();
        assert!(err.to_string().contains("typo") || err.to_string().contains("unknown"));
    }

    #[test]
    fn rejects_non_all_llm_input_count_words() {
        let bad = DOC.replace(
            "output: intent",
            "output: intent\n      inputs: [{channel: messages, count: any}]",
        );
        let err = parse_authored_graph(bad.as_bytes()).unwrap_err();
        assert!(err.to_string().contains("all"));
    }

    #[test]
    fn rejects_explicit_null_llm_input_count() {
        let bad = DOC.replace(
            "output: intent",
            "output: intent\n      inputs: [{channel: messages, count: null}]",
        );
        assert!(parse_authored_graph(bad.as_bytes()).is_err());
    }

    #[test]
    fn defaults_an_absent_llm_input_count_to_one() {
        let source = DOC.replace(
            "output: intent",
            "output: intent\n      inputs: [{channel: messages}]",
        );
        let doc = parse_authored_graph(source.as_bytes()).expect("absent count is valid");
        let AuthoredNode::Llm(route) = &doc.graph.nodes["route"] else {
            panic!("route must be an llm node")
        };
        assert_eq!(route.inputs[0].count.as_int(), Some(1));
    }

    #[test]
    fn rejects_replay_inputs_instead_of_dropping_them() {
        let bad = DOC.replace(
            "outputs: [raw_results, brand_cands]",
            "outputs: [raw_results, brand_cands]\n      inputs: [{channel: messages, count: 1}]",
        );
        let err = parse_authored_graph(bad.as_bytes()).unwrap_err();
        assert!(err.to_string().contains("replay") && err.to_string().contains("inputs"));
    }

    #[test]
    fn rejects_empty_replay_inputs_instead_of_treating_them_as_absent() {
        let bad = DOC.replace(
            "outputs: [raw_results, brand_cands]",
            "outputs: [raw_results, brand_cands]\n      inputs: []",
        );
        let err = parse_authored_graph(bad.as_bytes()).unwrap_err();
        assert!(err.to_string().contains("replay") && err.to_string().contains("inputs"));
    }

    #[test]
    fn rejects_llm_node_with_replay_fields() {
        let bad = DOC.replace("output: intent", "output: intent\n      duration_ms: 5");
        let err = parse_authored_graph(bad.as_bytes()).unwrap_err();
        assert!(err.to_string().contains("duration_ms"));
    }

    #[test]
    fn json_and_yaml_parse_identically() {
        let yaml_doc = parse_authored_graph(DOC.as_bytes()).unwrap();
        // Re-encode the YAML-decoded doc's source as JSON and confirm it decodes
        // to an equal document (YAML is a JSON superset through serde_yaml).
        let json = serde_json::json!({
            "graph": {
                "state": {
                    "messages": {"type": "messages", "reducer": "add_messages"},
                    "intent": {"type": "text"},
                    "raw_results": {"type": "json"}
                },
                "nodes": {
                    "route": {
                        "node_type": "llm",
                        "prompt": [{"role": "system", "content": "Classify intent."}, "@messages"],
                        "output": "intent",
                        "streaming": false
                    },
                    "tool_exec": {
                        "node_type": "replay",
                        "outputs": ["raw_results", "brand_cands"],
                        "duration_ms": 80
                    }
                },
                "edges": [
                    {"source": "START", "target": "route"},
                    {"source": "route", "branches": {"shopping": "plan", "non_shopping": "END"}}
                ]
            },
            "traces": [
                {
                    "id": "t-shopping",
                    "selected_branches": {"route": "shopping"},
                    "initial_state": {"query_text": "find cheaper"},
                    "replay_outputs": {"tool_exec": {"raw_results": [], "brand_cands": "Nike, Adidas"}}
                }
            ]
        });
        let json_bytes = serde_json::to_vec(&json).unwrap();
        let json_doc = parse_authored_graph(&json_bytes).unwrap();
        assert_eq!(yaml_doc, json_doc);
    }
}
