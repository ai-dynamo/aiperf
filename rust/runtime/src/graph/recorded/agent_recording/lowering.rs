// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pure lowering from validated recorded-agent DTOs into executable Graph-IR.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{self, Display};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use bytes::Bytes;
use serde_json::value::RawValue;
use serde_json::{Map, Value};

use crate::dataset::SegmentPool;
use crate::graph::driver::{ReplayTraceMetadata, TraceDriverSpec};
use crate::graph::input::{GraphInputBundle, GraphInputMetadata};
use crate::graph::materialize::validate_additional_body;
use crate::graph::model::{
    ChannelSpec, ChannelType, END_NODE_ID, ExecutableGraphNode, GraphRecord, GraphTracePlan,
    GraphTraceProgram, LlmNode, LlmRequestSpec, PromptItem, ReducerName, START_NODE_ID, StaticEdge,
    ToolNode, TraceRecord,
};

use super::discovery::{ValidatedRecordedAgentCorpus, ValidatedRecordedAgentTrace};
use super::schema::{RecordedAgentEvent, ReplayTaskIdentity};

const DEFAULT_FALLBACK_MAX_TOKENS: usize = 32_768;

/// Pure resolved request policy for one replay task.
#[derive(Clone, Debug)]
pub struct ReplayRequestProfile {
    /// Stable profile identifier carried in replay provenance.
    pub identity: String,
    /// Resolved streaming behavior.
    pub streaming: bool,
    /// Fallback generation cap when source completion usage is absent or zero.
    pub fallback_max_tokens: usize,
    /// Whether eligible source tool commands become executable tool nodes.
    pub execute_tools: bool,
    /// Whether the recorded model is allowed to override the run model.
    pub use_recorded_model: bool,
    /// Whether recorded temperature and top-p may replace profile fields.
    pub use_recorded_sampling: bool,
    /// Whether this is the standard comparable scenario, which never takes
    /// sampling from the recording.
    pub is_standard_scenario: bool,
    /// Non-owned request fields selected by profile resolution.
    pub additional_body: Map<String, Value>,
}

/// Resolve typed request policy without reading recordings or external state.
pub trait ReplayRequestProfileResolver: Send + Sync {
    /// Resolve one profile for a validated replay task identity.
    fn resolve(
        &self,
        task: &ReplayTaskIdentity,
    ) -> Result<ReplayRequestProfile, RecordedAgentLoweringError>;
}

/// Stock pure profile resolver used by the native `agent_recording` adapter.
#[derive(Clone, Debug)]
pub struct BuiltinReplayRequestProfileResolver {
    streaming: bool,
    fallback_max_tokens: usize,
    execute_tools: bool,
    use_recorded_model: bool,
    use_recorded_sampling: bool,
    is_standard_scenario: bool,
    unknown_adapter_warning_emitted: Arc<AtomicBool>,
}

impl Default for BuiltinReplayRequestProfileResolver {
    fn default() -> Self {
        Self {
            streaming: true,
            fallback_max_tokens: DEFAULT_FALLBACK_MAX_TOKENS,
            execute_tools: false,
            use_recorded_model: false,
            use_recorded_sampling: false,
            is_standard_scenario: false,
            unknown_adapter_warning_emitted: Arc::new(AtomicBool::new(false)),
        }
    }
}

impl BuiltinReplayRequestProfileResolver {
    /// Construct the stock resolver from already validated runner options.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        streaming: bool,
        fallback_max_tokens: usize,
        execute_tools: bool,
        use_recorded_model: bool,
        use_recorded_sampling: bool,
        is_standard_scenario: bool,
    ) -> Result<Self, RecordedAgentLoweringError> {
        if fallback_max_tokens == 0 {
            return Err(RecordedAgentLoweringError(
                "recorded-agent fallback_max_tokens must be positive".into(),
            ));
        }
        Ok(Self {
            streaming,
            fallback_max_tokens,
            execute_tools,
            use_recorded_model,
            use_recorded_sampling,
            is_standard_scenario,
            unknown_adapter_warning_emitted: Arc::new(AtomicBool::new(false)),
        })
    }
}

impl ReplayRequestProfileResolver for BuiltinReplayRequestProfileResolver {
    fn resolve(
        &self,
        task: &ReplayTaskIdentity,
    ) -> Result<ReplayRequestProfile, RecordedAgentLoweringError> {
        let mut additional_body = Map::new();
        match task.adapter.as_str() {
            "swebench" => {
                additional_body.insert("temperature".into(), Value::from(0.7));
                additional_body.insert("top_p".into(), Value::from(0.8));
                additional_body.insert("top_k".into(), Value::from(20));
                additional_body.insert("min_p".into(), Value::from(0));
                additional_body.insert("parallel_tool_calls".into(), Value::Bool(true));
            }
            "pinchbench" => {}
            adapter => {
                if !self
                    .unknown_adapter_warning_emitted
                    .swap(true, Ordering::Relaxed)
                {
                    tracing::warn!(adapter, task_id = %task.task_id, "recorded-agent task has no inferred request family profile");
                }
            }
        }
        Ok(ReplayRequestProfile {
            identity: format!("recorded-agent:{}", task.adapter),
            streaming: self.streaming,
            fallback_max_tokens: self.fallback_max_tokens,
            execute_tools: self.execute_tools,
            use_recorded_model: self.use_recorded_model,
            use_recorded_sampling: self.use_recorded_sampling,
            is_standard_scenario: self.is_standard_scenario,
            additional_body,
        })
    }
}

/// Lowering error with source trace and event context where available.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordedAgentLoweringError(pub String);

impl Display for RecordedAgentLoweringError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for RecordedAgentLoweringError {}

impl From<crate::dataset::DatasetError> for RecordedAgentLoweringError {
    fn from(error: crate::dataset::DatasetError) -> Self {
        Self(error.to_string())
    }
}

impl From<serde_json::Error> for RecordedAgentLoweringError {
    fn from(error: serde_json::Error) -> Self {
        Self(error.to_string())
    }
}

/// Lower a fully validated recording corpus without filesystem, clock, or process effects.
pub fn lower_recorded_agent_corpus(
    corpus: &ValidatedRecordedAgentCorpus,
    resolver: &dyn ReplayRequestProfileResolver,
    pool: &mut SegmentPool,
) -> Result<GraphInputBundle, RecordedAgentLoweringError> {
    let mut programs = Vec::with_capacity(corpus.traces.len());
    for (ordinal, trace) in corpus.traces.iter().enumerate() {
        programs.push(lower_trace(trace, ordinal, corpus, resolver, pool)?);
    }
    let metadata = GraphInputMetadata {
        format: "agent_recording".into(),
        root_count: programs.len(),
        node_count: programs
            .iter()
            .map(|program| program.profiling.graph.llm_node_count())
            .sum(),
    };
    Ok(GraphInputBundle {
        programs,
        segments: Arc::new(pool.clone().freeze()),
        metadata,
    })
}

fn lower_trace(
    trace: &ValidatedRecordedAgentTrace,
    ordinal: usize,
    corpus: &ValidatedRecordedAgentCorpus,
    resolver: &dyn ReplayRequestProfileResolver,
    pool: &mut SegmentPool,
) -> Result<GraphTraceProgram, RecordedAgentLoweringError> {
    let identity = trace
        .identity
        .clone()
        .unwrap_or_else(|| ReplayTaskIdentity {
            adapter: trace
                .recording
                .metadata
                .benchmark
                .clone()
                .unwrap_or_else(|| "unknown".into()),
            family: "standalone".into(),
            task_id: trace.trace_id.clone(),
            primary_role: None,
        });
    let profile = resolver.resolve(&identity)?;
    if profile.fallback_max_tokens == 0 {
        return Err(trace_error(
            trace,
            None,
            "resolved request profile has a zero fallback_max_tokens",
        ));
    }
    validate_additional_body(&profile.additional_body, "request profile")
        .map_err(|error| RecordedAgentLoweringError(error.to_string()))?;
    let manifest_extra = corpus
        .manifest
        .as_ref()
        .map(|manifest| &manifest.defaults.extra_request_body);
    let fallback_max_tokens = corpus
        .manifest
        .as_ref()
        .map(|manifest| usize::try_from(manifest.defaults.fallback_max_output_tokens))
        .transpose()
        .map_err(|_| {
            trace_error(
                trace,
                None,
                "manifest fallback_max_output_tokens exceeds usize",
            )
        })?
        .unwrap_or(profile.fallback_max_tokens);
    if let Some(extra) = manifest_extra {
        validate_additional_body(extra, "manifest extra_request_body")
            .map_err(|error| RecordedAgentLoweringError(error.to_string()))?;
    }

    let mut graph = GraphRecord::default();
    let mut previous_llm: Option<RecordedLlm> = None;
    let mut previous_node: Option<String> = None;
    let mut pending_tools = Vec::new();
    let mut llm_count = 0_usize;
    let mut tool_count = 0_usize;
    let mut target_output_tokens = Vec::new();
    let mut completed_tool_count = 0_u64;

    for event in &trace.recording.events {
        if is_completed_tool_call(event) {
            completed_tool_count = completed_tool_count.saturating_add(1);
            if previous_llm.is_some() {
                let command = event
                    .action
                    .as_ref()
                    .and_then(|action| action.get("command"))
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        trace_error(trace, Some(event), "completed tool call lacks command")
                    })?;
                pending_tools.push(command.to_owned());
            }
            continue;
        }
        if event.event_type != "model_call" {
            continue;
        }
        let request = event.provider_request.as_ref().ok_or_else(|| {
            trace_error(
                trace,
                Some(event),
                "model call lacks provider_request after validation",
            )
        })?;
        let messages = request.messages.as_ref().ok_or_else(|| {
            trace_error(
                trace,
                Some(event),
                "model call lacks provider_request.messages after validation",
            )
        })?;
        let node_id = format!("llm_{llm_count}");
        llm_count += 1;

        if let (Some(previous), Some(previous_node_id)) = (&previous_llm, &previous_node) {
            if profile.execute_tools && !pending_tools.is_empty() {
                tool_count += 1;
                let tool_id = format!("tool_{tool_count}");
                add_tool_node(&mut graph, &tool_id, std::mem::take(&mut pending_tools));
                add_edge(&mut graph, previous_node_id, &tool_id, None);
                add_edge(&mut graph, &tool_id, &node_id, None);
            } else {
                let gap_us = relative_gap_us(previous.event, event);
                add_edge(&mut graph, previous_node_id, &node_id, gap_us);
                pending_tools.clear();
            }
        } else {
            add_edge(&mut graph, START_NODE_ID, &node_id, None);
        }

        let node = lower_llm_node(
            trace,
            event,
            messages,
            &profile,
            manifest_extra,
            fallback_max_tokens,
            pool,
        )?;
        target_output_tokens.push(completion_tokens(event));
        graph.state.insert(
            node.output.clone(),
            ChannelSpec {
                channel_type: ChannelType::Messages,
                reducer: ReducerName::AddMessages,
            },
        );
        graph
            .nodes
            .insert(node_id.clone(), ExecutableGraphNode::Llm(node));
        previous_llm = Some(RecordedLlm { event });
        previous_node = Some(node_id);
    }

    let Some(previous_node) = previous_node else {
        return Err(trace_error(
            trace,
            None,
            "recording contains no model calls",
        ));
    };
    let terminal_node = if profile.execute_tools && !pending_tools.is_empty() {
        tool_count += 1;
        let tool_id = format!("tool_{tool_count}");
        add_tool_node(&mut graph, &tool_id, pending_tools);
        add_edge(&mut graph, &previous_node, &tool_id, None);
        tool_id
    } else {
        previous_node
    };
    add_edge(&mut graph, &terminal_node, END_NODE_ID, None);

    Ok(GraphTraceProgram {
        profiling: GraphTracePlan {
            graph,
            trace: TraceRecord {
                id: trace.trace_id.clone(),
                graph_ref: None,
                initial_state: BTreeMap::new(),
            },
            arrival_offset_ns: None,
        },
        warmup: None,
        environment: None,
        replay: Some(ReplayTraceMetadata {
            manifest_ordinal: ordinal,
            identity,
            source_digest: trace.digest.clone(),
            normalization_target_digest: None,
            target_output_tokens,
            expected_llm_node_count: llm_count as u64,
            expected_tool_node_count: completed_tool_count,
            request_profile_identity: profile.identity,
            comparability_annotations: BTreeMap::new(),
        }),
        driver: TraceDriverSpec {
            kind: "recorded_replay".into(),
            data: BTreeMap::new(),
        },
    })
}

struct RecordedLlm<'a> {
    event: &'a RecordedAgentEvent,
}

fn lower_llm_node(
    trace: &ValidatedRecordedAgentTrace,
    event: &RecordedAgentEvent,
    messages: &[Box<RawValue>],
    profile: &ReplayRequestProfile,
    manifest_extra: Option<&Map<String, Value>>,
    fallback_max_tokens: usize,
    pool: &mut SegmentPool,
) -> Result<LlmNode, RecordedAgentLoweringError> {
    let mut parent = None;
    let mut items = Vec::with_capacity(messages.len());
    for message in messages {
        let value: Value = serde_json::from_str(message.get()).map_err(|error| {
            trace_error(
                trace,
                Some(event),
                &format!("provider_request.messages entry is invalid JSON: {error}"),
            )
        })?;
        let object = value.as_object().ok_or_else(|| {
            trace_error(
                trace,
                Some(event),
                "provider_request.messages entry is not an object",
            )
        })?;
        let role = object
            .get("role")
            .and_then(Value::as_str)
            .filter(|role| !role.is_empty())
            .ok_or_else(|| {
                trace_error(
                    trace,
                    Some(event),
                    "provider_request.messages entry lacks role",
                )
            })?;
        let wire = Bytes::copy_from_slice(message.get().as_bytes());
        let handle = pool.intern_message(parent, role, wire, Vec::<u32>::new())?;
        parent = Some(handle);
        items.push(PromptItem::Seg { seg: handle });
    }

    let request = event.provider_request.as_ref().ok_or_else(|| {
        trace_error(
            trace,
            Some(event),
            "model call lacks provider_request after validation",
        )
    })?;
    let tools = request
        .tools
        .as_ref()
        .map(|tools| intern_tools(pool, parent, tools, trace, event))
        .transpose()?;
    let additional_body = request_body(event, profile, manifest_extra)?
        .map(|body| {
            let parent = tools.or(parent);
            pool.intern_raw(parent, body)
                .map_err(RecordedAgentLoweringError::from)
        })
        .transpose()?;
    let max_tokens = usize::try_from(completion_tokens(event))
        .ok()
        .filter(|tokens| *tokens > 0)
        .unwrap_or(fallback_max_tokens);
    Ok(LlmNode {
        output: format!("llm_output_{}", event.id),
        streaming: profile.streaming,
        inputs: Vec::new(),
        min_start_delay_us: None,
        max_tokens: Some(max_tokens),
        items,
        request: Some(LlmRequestSpec {
            tools,
            model: profile
                .use_recorded_model
                .then(|| request.model.clone())
                .flatten(),
            additional_body,
        }),
        metadata: BTreeMap::new(),
    })
}

fn intern_tools(
    pool: &mut SegmentPool,
    parent: Option<crate::dataset::Handle>,
    tools: &RawValue,
    trace: &ValidatedRecordedAgentTrace,
    event: &RecordedAgentEvent,
) -> Result<crate::dataset::Handle, RecordedAgentLoweringError> {
    let value: Value = serde_json::from_str(tools.get()).map_err(|error| {
        trace_error(
            trace,
            Some(event),
            &format!("provider_request.tools is invalid JSON: {error}"),
        )
    })?;
    let Some(entries) = value.as_array() else {
        return Err(trace_error(
            trace,
            Some(event),
            "provider_request.tools must be a JSON array",
        ));
    };
    if entries.iter().any(|entry| !entry.is_object()) {
        return Err(trace_error(
            trace,
            Some(event),
            "provider_request.tools entries must be JSON objects",
        ));
    }
    Ok(pool.intern_raw(parent, Bytes::copy_from_slice(tools.get().as_bytes()))?)
}

fn request_body(
    event: &RecordedAgentEvent,
    profile: &ReplayRequestProfile,
    manifest_extra: Option<&Map<String, Value>>,
) -> Result<Option<Bytes>, RecordedAgentLoweringError> {
    let request = event
        .provider_request
        .as_ref()
        .ok_or_else(|| RecordedAgentLoweringError("model call lacks provider_request".into()))?;
    let mut fields = profile.additional_body.clone();
    if profile.use_recorded_sampling && !profile.is_standard_scenario {
        if let Some(temperature) = request.temperature {
            fields.insert("temperature".into(), Value::from(temperature));
        }
        if let Some(top_p) = request.top_p {
            fields.insert("top_p".into(), Value::from(top_p));
        }
    }
    if let Some(extra) = manifest_extra {
        fields.extend(extra.clone());
    }
    validate_additional_body(&fields, "lowered additional request body")
        .map_err(|error| RecordedAgentLoweringError(error.to_string()))?;
    (!fields.is_empty())
        .then(|| {
            serde_json::to_vec(&fields)
                .map(Bytes::from)
                .map_err(Into::into)
        })
        .transpose()
}

fn add_tool_node(graph: &mut GraphRecord, node_id: &str, commands: Vec<String>) {
    let output = format!("{node_id}_output");
    graph.state.insert(
        output.clone(),
        ChannelSpec {
            channel_type: ChannelType::Text,
            reducer: ReducerName::Overwrite,
        },
    );
    graph.nodes.insert(
        node_id.into(),
        ExecutableGraphNode::Tool(ToolNode {
            output,
            commands,
            timeout_ns: None,
        }),
    );
}

fn add_edge(
    graph: &mut GraphRecord,
    source: &str,
    target: &str,
    delay_after_predecessor_us: Option<f64>,
) {
    graph.edges.push(StaticEdge {
        source: source.into(),
        target: target.into(),
        delay_after_predecessor_us,
        min_start_delay_us: None,
        delay_after_predecessor_start_us: None,
        delay_after_predecessor_first_token_us: None,
    });
}

fn relative_gap_us(previous: &RecordedAgentEvent, current: &RecordedAgentEvent) -> Option<f64> {
    let current_start =
        current.timestamp - current.duration_ns.unwrap_or_default() as f64 / 1_000_000_000.0;
    let gap_us = (current_start - previous.timestamp) * 1_000_000.0;
    (gap_us.is_finite() && gap_us > 0.0).then_some(gap_us)
}

fn completion_tokens(event: &RecordedAgentEvent) -> u64 {
    event
        .response_message
        .as_ref()
        .and_then(|response| response.pointer("/extra/response/usage/completion_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0)
}

fn is_completed_tool_call(event: &RecordedAgentEvent) -> bool {
    event.event_type == "tool_call"
        && event
            .action
            .as_ref()
            .and_then(|action| action.get("command"))
            .and_then(Value::as_str)
            .is_some()
        && event.error.as_ref().is_none_or(is_completed_control_flow)
}

fn is_completed_control_flow(error: &Value) -> bool {
    error
        .get("type")
        .and_then(Value::as_str)
        .is_some_and(|kind| {
            matches!(
                kind,
                "InterruptAgentFlow"
                    | "Submitted"
                    | "LimitsExceeded"
                    | "ReplayExhausted"
                    | "UserInterruption"
                    | "FormatError"
            )
        })
}

fn trace_error(
    trace: &ValidatedRecordedAgentTrace,
    event: Option<&RecordedAgentEvent>,
    detail: &str,
) -> RecordedAgentLoweringError {
    match event {
        Some(event) => RecordedAgentLoweringError(format!(
            "{}: trace {:?} event {} type {:?}: {detail}",
            trace.path.display(),
            trace.trace_id,
            event.id,
            event.event_type
        )),
        None => RecordedAgentLoweringError(format!(
            "{}: trace {:?}: {detail}",
            trace.path.display(),
            trace.trace_id
        )),
    }
}
