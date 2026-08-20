// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pure lowering from imported session history into non-executable Graph-IR.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use serde_json::Value;

use crate::dataset::{SegmentPool, TextTokenizer};
use crate::graph::driver::{ReplayTraceMetadata, TraceDriverSpec};
use crate::graph::input::{GraphInputBundle, GraphInputMetadata, GraphInputWarning};
use crate::graph::model::{
    ChannelSpec, ChannelType, END_NODE_ID, ExecutableGraphNode, GraphRecord, GraphTracePlan,
    GraphTraceProgram, LlmNode, LlmRequestSpec, PromptItem, ReducerName, START_NODE_ID, StaticEdge,
    TraceRecord,
};

use super::{ImportedAgentError, ImportedAgentSession, ImportedAgentSource};
use crate::graph::recorded::agent_recording::{ReplayRequestProfileResolver, ReplayTaskIdentity};

/// Lower parsed imported sessions into isolated linear recorded-replay programs.
pub fn lower_imported_agent_sessions(
    sessions: &[ImportedAgentSession],
    resolver: &dyn ReplayRequestProfileResolver,
    tokenizer: &dyn TextTokenizer,
    pool: &mut SegmentPool,
) -> Result<GraphInputBundle, ImportedAgentError> {
    let mut ordered = sessions.iter().collect::<Vec<_>>();
    ordered.sort_by(|left, right| {
        left.session_id
            .cmp(&right.session_id)
            .then_with(|| left.source_path.cmp(&right.source_path))
    });
    let mut programs = Vec::with_capacity(ordered.len());
    let mut warnings = BTreeSet::new();
    for (ordinal, session) in ordered.into_iter().enumerate() {
        programs.push(lower_session(
            session,
            ordinal,
            resolver,
            tokenizer,
            pool,
            &mut warnings,
        )?);
    }
    Ok(GraphInputBundle {
        metadata: GraphInputMetadata {
            format: "agent_recording".into(),
            root_count: programs.len(),
            node_count: programs
                .iter()
                .map(|program| program.profiling.graph.llm_node_count())
                .sum(),
            warning_facts: warnings.into_iter().collect(),
        },
        programs,
        segments: Arc::new(pool.clone().freeze()),
    })
}

fn lower_session(
    session: &ImportedAgentSession,
    ordinal: usize,
    resolver: &dyn ReplayRequestProfileResolver,
    tokenizer: &dyn TextTokenizer,
    pool: &mut SegmentPool,
    warnings: &mut BTreeSet<GraphInputWarning>,
) -> Result<GraphTraceProgram, ImportedAgentError> {
    let source = source_label(session.source);
    let family = if session.parent.is_some() {
        "subagent"
    } else {
        "session"
    };
    let identity = ReplayTaskIdentity {
        adapter: source.into(),
        family: family.into(),
        task_id: session.session_id.clone(),
        primary_role: None,
    };
    let profile = resolver
        .resolve(&identity)
        .map_err(|_| session_error(session, "could not resolve imported request profile"))?;
    if profile.fallback_max_tokens == 0 {
        return Err(session_error(
            session,
            "resolved imported request profile has a zero fallback_max_tokens",
        ));
    }
    if profile.execute_tools || profile.use_recorded_sampling || profile.is_standard_scenario {
        return Err(session_error(
            session,
            "imported sessions reject executable tools, recorded sampling, and standard scenario",
        ));
    }
    if !profile.additional_body.is_empty() {
        return Err(session_error(
            session,
            "imported sessions reject synthesized additional request body",
        ));
    }
    warnings.extend(profile.warning_facts.iter().cloned());

    let mut graph = GraphRecord::default();
    let mut previous_node = None;
    let mut target_output_tokens = Vec::with_capacity(session.calls.len());
    for (index, call) in session.calls.iter().enumerate() {
        let node_id = format!("llm_{index}");
        let mut parent = None;
        let mut items = Vec::with_capacity(call.request_messages.len());
        let mut input_tokens = 0_u64;
        for message in &call.request_messages {
            if message.role.is_empty() {
                return Err(session_error(
                    session,
                    "imported request message has an empty role",
                ));
            }
            let wire: serde_json::Value = serde_json::from_slice(&message.wire).map_err(|_| {
                session_error(session, "imported request message is not valid JSON")
            })?;
            if wire.get("role").and_then(serde_json::Value::as_str) != Some(&message.role) {
                return Err(session_error(
                    session,
                    "imported request message role disagrees with serialized wire",
                ));
            }
            let message_text = std::str::from_utf8(&message.wire).map_err(|_| {
                session_error(session, "imported request message is not valid UTF-8")
            })?;
            let message_tokens = tokenizer.encode(message_text).map_err(|_| {
                session_error(session, "could not tokenize imported request message")
            })?;
            input_tokens = input_tokens
                .checked_add(u64::try_from(message_tokens.len()).map_err(|_| {
                    session_error(session, "imported request message token count exceeds u64")
                })?)
                .ok_or_else(|| {
                    session_error(session, "imported request input token count exceeds u64")
                })?;
            let handle = pool
                .intern_message(
                    parent,
                    message.role.as_str(),
                    message.wire.clone(),
                    message_tokens.into_boxed_slice(),
                )
                .map_err(|_| session_error(session, "could not intern imported request message"))?;
            parent = Some(handle);
            items.push(PromptItem::Seg { seg: handle });
        }
        if items.is_empty() {
            return Err(session_error(
                session,
                "imported model call has no reconstructed request messages",
            ));
        }
        graph.state.insert(
            format!("{node_id}_output"),
            ChannelSpec {
                channel_type: ChannelType::Messages,
                reducer: ReducerName::AddMessages,
            },
        );
        graph.nodes.insert(
            node_id.clone(),
            ExecutableGraphNode::Llm(LlmNode {
                output: format!("{node_id}_output"),
                streaming: profile.streaming,
                inputs: Vec::new(),
                min_start_delay_us: None,
                max_tokens: Some(profile.fallback_max_tokens),
                items,
                request: Some(LlmRequestSpec {
                    tools: None,
                    model: profile
                        .use_recorded_model
                        .then(|| call.model.clone().or_else(|| session.model.clone()))
                        .flatten(),
                    additional_body: None,
                }),
                metadata: BTreeMap::from([("input_tokens".into(), Value::from(input_tokens))]),
            }),
        );
        match previous_node.as_deref() {
            Some(previous) => {
                add_edge(&mut graph, previous, &node_id, call.delay_after_previous_us)
            }
            None => add_edge(&mut graph, START_NODE_ID, &node_id, None),
        }
        previous_node = Some(node_id);
        target_output_tokens.push(0);
    }
    let previous_node = previous_node.ok_or_else(|| {
        session_error(
            session,
            "imported session contains no inferred model calls after parsing",
        )
    })?;
    add_edge(&mut graph, &previous_node, END_NODE_ID, None);
    let mut annotations = BTreeMap::from([
        ("source_format".into(), Value::String(source.into())),
        ("request_wire_exact".into(), Value::Bool(false)),
        ("tool_schema_available".into(), Value::Bool(false)),
        ("output_tokens_available".into(), Value::Bool(false)),
        ("model_latency_available".into(), Value::Bool(false)),
        ("reasoning_included".into(), Value::Bool(false)),
        (
            "tool_results_complete".into(),
            Value::Bool(session.tool_results_complete),
        ),
        (
            "subagent_topology".into(),
            Value::String(
                if session.parent.is_some() {
                    "sibling"
                } else {
                    "none"
                }
                .into(),
            ),
        ),
        (
            "ignored_record_count".into(),
            Value::from(session.ignored_record_count),
        ),
        (
            "omitted_reasoning_count".into(),
            Value::from(session.omitted_reasoning_count),
        ),
        ("cwd_present".into(), Value::Bool(session.cwd_present)),
        (
            "git_branch_present".into(),
            Value::Bool(session.git_branch_present),
        ),
    ]);
    if let Some(parent) = &session.parent {
        annotations.insert(
            "parent_session_id".into(),
            Value::String(parent.session_id.clone()),
        );
        annotations.insert(
            "parent_tool_use_id".into(),
            Value::String(parent.tool_use_id.clone()),
        );
    }
    Ok(GraphTraceProgram {
        profiling: GraphTracePlan {
            graph,
            trace: TraceRecord {
                id: session.session_id.clone(),
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
            source_digest: session.source_digest.clone(),
            normalization_target_digest: None,
            target_output_tokens,
            expected_llm_node_count: session.calls.len() as u64,
            expected_tool_node_count: session.completed_tool_count,
            request_profile_identity: profile.identity,
            comparability_annotations: annotations,
        }),
        driver: TraceDriverSpec::recorded_replay(),
    })
}

fn source_label(source: ImportedAgentSource) -> &'static str {
    match source {
        ImportedAgentSource::Codex => "codex",
        ImportedAgentSource::ClaudeCode => "claude_code",
    }
}

fn session_error(session: &ImportedAgentSession, detail: &'static str) -> ImportedAgentError {
    ImportedAgentError::new(
        &session.source_path,
        0,
        source_label(session.source),
        "message",
        detail,
    )
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
