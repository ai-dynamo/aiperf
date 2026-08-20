// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Lowering and adapter integration coverage for imported agent sessions.

use std::fs;
use std::path::{Path, PathBuf};

use aiperf_runtime::config::model::dataset::RecordedAgentSourceFormat;
use aiperf_runtime::dataset::{Payload, TiktokenTokenizer};
use aiperf_runtime::engine::graph_input::{
    BuiltinRunnerGraphInputAdapterResolver, CacheBustTarget, GraphInputAdapter, GraphInputContext,
    RecordedAgentRunnerGraphInputAdapter, prepare_local_graph_inspection_input,
};
use aiperf_runtime::graph::model::{ChannelType, ExecutableGraphNode, PromptItem, ReducerName};
use aiperf_runtime::graph::recorded::agent_recording::{
    BuiltinReplayRequestProfileResolver, ImportedAgentSession, ImportedAgentSource,
    ImportedModelCall, ImportedSessionFamily, RawJsonMessage, discover_imported_agent_read_set,
    lower_imported_agent_sessions, parse_imported_agent_sessions,
};
use aiperf_runtime::graph::segment::SegmentPool;
use bytes::Bytes;
use serde_json::json;

fn fixture(path: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/recorded_agent_session_import")
        .join(path)
}

fn edge_pairs(program: &aiperf_runtime::graph::model::GraphTraceProgram) -> Vec<(&str, &str)> {
    program
        .profiling
        .graph
        .edges
        .iter()
        .map(|edge| (edge.source.as_str(), edge.target.as_str()))
        .collect()
}

fn llm_nodes(
    program: &aiperf_runtime::graph::model::GraphTraceProgram,
) -> Vec<&aiperf_runtime::graph::model::LlmNode> {
    program
        .profiling
        .graph
        .nodes
        .values()
        .filter_map(|node| match node {
            ExecutableGraphNode::Llm(node) => Some(node),
            ExecutableGraphNode::Tool(_) => None,
        })
        .collect()
}

fn lowered_message_wires(
    bundle: &aiperf_runtime::graph::input::GraphInputBundle,
) -> Vec<Vec<Bytes>> {
    llm_nodes(&bundle.programs[0])
        .into_iter()
        .map(|node| {
            node.items
                .iter()
                .map(|item| {
                    let PromptItem::Seg { seg } = item else {
                        panic!("imported request must lower to message segments");
                    };
                    let Payload::Message { wire, .. } =
                        bundle.segments.get(*seg).expect("lowered message")
                    else {
                        panic!("imported request item must be a message");
                    };
                    wire.clone()
                })
                .collect()
        })
        .collect()
}

fn builtin_tokenizer() -> TiktokenTokenizer {
    TiktokenTokenizer::builtin()
}

#[test]
fn imported_codex_sessions_lower_to_linear_recorded_replay_graphs() {
    let read_set = discover_imported_agent_read_set(
        &fixture("codex/linear.jsonl"),
        None,
        RecordedAgentSourceFormat::Codex,
        None,
    )
    .expect("discover Codex fixture");
    let sessions = parse_imported_agent_sessions(&read_set).expect("parse Codex fixture");
    let mut pool = SegmentPool::new();
    let tokenizer = builtin_tokenizer();
    let resolver = BuiltinReplayRequestProfileResolver::new(true, 123, false, false, false, false)
        .expect("valid resolver");

    let bundle = lower_imported_agent_sessions(&sessions, &resolver, &tokenizer, &mut pool)
        .expect("lower imported Codex session");
    let program = &bundle.programs[0];
    let replay = program.replay.as_ref().expect("replay metadata");

    assert_eq!(program.profiling.trace.id, sessions[0].session_id);
    assert_eq!(
        edge_pairs(program),
        [("START", "llm_0"), ("llm_0", "llm_1"), ("llm_1", "END")]
    );
    assert!(
        program
            .profiling
            .graph
            .nodes
            .values()
            .all(|node| node.as_llm().is_some())
    );
    assert!(program.environment.is_none() && program.warmup.is_none());
    assert_eq!(program.driver.kind, "recorded_replay");
    assert_eq!(replay.target_output_tokens, vec![0, 0]);
    assert_eq!(bundle.metadata.format, "agent_recording");
    assert_eq!(replay.identity.adapter, "codex");
    assert_eq!(replay.identity.family, "session");
    assert_eq!(replay.expected_tool_node_count, 0);
    assert_eq!(replay.request_profile_identity, "recorded-agent:codex");
    assert_eq!(
        replay.comparability_annotations["request_wire_exact"],
        false
    );
    assert_eq!(
        replay.comparability_annotations["tool_results_complete"],
        true
    );

    let nodes = llm_nodes(program);
    assert_eq!(nodes.len(), 2);
    for node in &nodes {
        let input_tokens = node
            .items
            .iter()
            .map(|item| match item {
                PromptItem::Seg { seg } => bundle
                    .segments
                    .get(*seg)
                    .expect("lowered message segment")
                    .token_count()
                    .expect("lowered message has token ids")
                    as u64,
                PromptItem::RawMessages { .. }
                | PromptItem::Text { .. }
                | PromptItem::Splice { .. } => panic!("imported request has only segment items"),
            })
            .sum::<u64>();
        assert_eq!(node.metadata["input_tokens"], input_tokens);
        let request = node.request.as_ref().expect("request policy");
        assert_eq!(node.max_tokens, Some(123));
        assert!(node.streaming);
        assert!(request.tools.is_none());
        assert!(request.additional_body.is_none());
        assert!(request.model.is_none());
        assert!(
            node.items
                .iter()
                .all(|item| matches!(item, PromptItem::Seg { .. }))
        );
        let channel = program
            .profiling
            .graph
            .state
            .get(&node.output)
            .expect("node output channel");
        assert_eq!(channel.channel_type, ChannelType::Messages);
        assert_eq!(channel.reducer, ReducerName::AddMessages);
    }

    let first_handles = nodes[0]
        .items
        .iter()
        .map(|item| match item {
            PromptItem::Seg { seg } => *seg,
            _ => unreachable!("only segments were asserted"),
        })
        .collect::<Vec<_>>();
    let second_handles = nodes[1]
        .items
        .iter()
        .map(|item| match item {
            PromptItem::Seg { seg } => *seg,
            _ => unreachable!("only segments were asserted"),
        })
        .collect::<Vec<_>>();
    assert_eq!(first_handles, second_handles[..first_handles.len()]);
    let Payload::Message { wire, .. } = bundle.segments.get(first_handles[0]).expect("first wire")
    else {
        panic!("imported request history must use message segments");
    };
    assert_eq!(
        wire,
        "{\"role\":\"system\",\"content\":\"You are Codex…\"}".as_bytes()
    );
}

#[test]
fn imported_tool_delay_applies_only_to_the_next_llm_edge() {
    let read_set = discover_imported_agent_read_set(
        &fixture("codex/with_tools.jsonl"),
        None,
        RecordedAgentSourceFormat::Codex,
        None,
    )
    .expect("discover Codex fixture");
    let sessions = parse_imported_agent_sessions(&read_set).expect("parse Codex fixture");
    let mut pool = SegmentPool::new();
    let tokenizer = builtin_tokenizer();
    let resolver = BuiltinReplayRequestProfileResolver::default();
    let bundle = lower_imported_agent_sessions(&sessions, &resolver, &tokenizer, &mut pool)
        .expect("lower imported Codex session");
    let program = &bundle.programs[0];

    assert_eq!(
        edge_pairs(program),
        [
            ("START", "llm_0"),
            ("llm_0", "llm_1"),
            ("llm_1", "llm_2"),
            ("llm_2", "END"),
        ]
    );
    let delayed = program
        .profiling
        .graph
        .edges
        .iter()
        .filter(|edge| edge.delay_after_predecessor_us.is_some())
        .collect::<Vec<_>>();
    assert_eq!(delayed.len(), 1);
    assert_eq!(delayed[0].source, "llm_0");
    assert_eq!(delayed[0].target, "llm_1");
    assert_eq!(delayed[0].delay_after_predecessor_us, Some(250_000.0));
    assert_eq!(
        program
            .replay
            .as_ref()
            .expect("replay")
            .expected_tool_node_count,
        1
    );
}

#[test]
fn imported_request_history_legacy_call_vec_lowers_and_preserves_completed_tool_count() {
    let session = ImportedAgentSession {
        session_id: "interrupted".into(),
        source: ImportedAgentSource::Codex,
        source_path: fixture("codex/linear.jsonl"),
        source_digest: "digest".into(),
        model: None,
        system_prompt: None,
        cwd_present: false,
        git_branch_present: false,
        parent: None,
        request_history: Default::default(),
        calls: vec![ImportedModelCall {
            source_id: "call".into(),
            request_messages: vec![RawJsonMessage {
                role: "user".into(),
                wire: Bytes::from_static(b"{\"role\":\"user\",\"content\":\"x\"}"),
            }],
            model: None,
            delay_after_previous_us: None,
            tool_schema_available: false,
            output_tokens: None,
        }],
        observed_tool_count: 1,
        completed_tool_count: 0,
        ignored_record_count: 0,
        omitted_reasoning_count: 0,
        tool_results_complete: false,
    };
    let bundle = lower_imported_agent_sessions(
        &[session],
        &BuiltinReplayRequestProfileResolver::default(),
        &builtin_tokenizer(),
        &mut SegmentPool::new(),
    )
    .expect("lower interrupted session");
    assert_eq!(
        llm_nodes(&bundle.programs[0])[0].items.len(),
        1,
        "legacy ImportedModelCall.request_messages must remain the fallback",
    );
    assert_eq!(
        bundle.programs[0]
            .replay
            .as_ref()
            .expect("replay")
            .expected_tool_node_count,
        0
    );
}

#[test]
fn imported_request_history_shared_and_legacy_lower_in_exact_wire_order() {
    let read_set = discover_imported_agent_read_set(
        &fixture("codex/linear.jsonl"),
        None,
        RecordedAgentSourceFormat::Codex,
        None,
    )
    .expect("discover Codex fixture");
    let shared = parse_imported_agent_sessions(&read_set).expect("parse Codex fixture")[0].clone();
    let mut legacy = shared.clone();
    for (call_index, call) in legacy.calls.iter_mut().enumerate() {
        call.request_messages = shared
            .request_messages(call_index)
            .expect("shared request history")
            .to_vec();
    }
    legacy.request_history = Default::default();

    let tokenizer = builtin_tokenizer();
    let resolver = BuiltinReplayRequestProfileResolver::default();
    let shared_bundle =
        lower_imported_agent_sessions(&[shared], &resolver, &tokenizer, &mut SegmentPool::new())
            .expect("lower shared request history");
    let legacy_bundle =
        lower_imported_agent_sessions(&[legacy], &resolver, &tokenizer, &mut SegmentPool::new())
            .expect("lower legacy request history");

    assert_eq!(
        lowered_message_wires(&shared_bundle),
        lowered_message_wires(&legacy_bundle),
    );
}

#[test]
fn imported_lowering_rejects_empty_or_mismatched_message_roles() {
    for (role, wire) in [
        ("", b"{\"role\":\"user\",\"content\":\"x\"}".as_slice()),
        (
            "assistant",
            b"{\"role\":\"user\",\"content\":\"x\"}".as_slice(),
        ),
    ] {
        let session = ImportedAgentSession {
            session_id: "roles".into(),
            source: ImportedAgentSource::Codex,
            source_path: fixture("codex/linear.jsonl"),
            source_digest: "digest".into(),
            model: None,
            system_prompt: None,
            cwd_present: false,
            git_branch_present: false,
            parent: None,
            request_history: Default::default(),
            observed_tool_count: 0,
            completed_tool_count: 0,
            ignored_record_count: 0,
            omitted_reasoning_count: 0,
            tool_results_complete: true,
            calls: vec![ImportedModelCall {
                source_id: "call".into(),
                request_messages: vec![RawJsonMessage {
                    role: role.into(),
                    wire: Bytes::copy_from_slice(wire),
                }],
                model: None,
                delay_after_previous_us: None,
                tool_schema_available: false,
                output_tokens: None,
            }],
        };
        assert!(
            lower_imported_agent_sessions(
                &[session],
                &BuiltinReplayRequestProfileResolver::default(),
                &builtin_tokenizer(),
                &mut SegmentPool::new(),
            )
            .is_err()
        );
    }
}

#[test]
fn imported_adapter_rejects_tools_sampling_and_standard_scenario() {
    let adapter = RecordedAgentRunnerGraphInputAdapter;
    let tokenizer = TiktokenTokenizer::builtin();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");
    for (graph_extra, extra) in [
        (json!({"execute_tools": true}), json!({})),
        (json!({}), json!({"use_recorded_sampling": true})),
        (json!({}), json!({"standard_scenario": true})),
    ] {
        let mut input = json!({
            "type": "file", "format": "agent_recording", "path": fixture("codex/linear.jsonl"),
            "sampling": "sequential", "graph": {"source_format": "codex"},
        });
        input["graph"]
            .as_object_mut()
            .expect("graph")
            .extend(graph_extra.as_object().expect("object").clone());
        input
            .as_object_mut()
            .expect("object")
            .extend(extra.as_object().expect("object").clone());
        let raw = serde_json::value::to_raw_value(&input).expect("raw graph input");
        assert!(
            runtime
                .block_on(adapter.load(
                    &raw,
                    &GraphInputContext {
                        tokenizer: &tokenizer,
                        run_random_seed: Some(7),
                        endpoint_id: "chat"
                    },
                ))
                .is_err()
        );
    }
}

#[test]
fn imported_adapter_rejects_tool_and_pinch_images_without_tool_execution() {
    let adapter = RecordedAgentRunnerGraphInputAdapter;
    let tokenizer = TiktokenTokenizer::builtin();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");
    for graph_extra in [
        json!({"tool_image": "registry.invalid/tool:latest"}),
        json!({"pinch_image": "registry.invalid/pinch:latest"}),
    ] {
        let mut input = json!({
            "type": "file", "format": "agent_recording", "path": fixture("codex/linear.jsonl"),
            "sampling": "sequential", "graph": {"source_format": "codex"},
        });
        input["graph"]
            .as_object_mut()
            .expect("graph")
            .extend(graph_extra.as_object().expect("object").clone());
        let raw = serde_json::value::to_raw_value(&input).expect("raw graph input");
        let error = runtime
            .block_on(adapter.load(
                &raw,
                &GraphInputContext {
                    tokenizer: &tokenizer,
                    run_random_seed: Some(7),
                    endpoint_id: "chat",
                },
            ))
            .expect_err("imported source must reject image authority");
        assert!(format!("{error:#}").contains("imported recorded-agent sessions reject"));
    }
}

#[test]
fn imported_lowering_emits_complete_metadata_without_adapter_warning() {
    let read_set = discover_imported_agent_read_set(
        &fixture("codex/linear.jsonl"),
        None,
        RecordedAgentSourceFormat::Codex,
        None,
    )
    .expect("discover Codex");
    let sessions = parse_imported_agent_sessions(&read_set).expect("parse Codex");
    let mut pool = SegmentPool::new();
    let tokenizer = builtin_tokenizer();
    let bundle = lower_imported_agent_sessions(
        &sessions,
        &BuiltinReplayRequestProfileResolver::new(true, 321, false, true, false, false)
            .expect("resolver"),
        &tokenizer,
        &mut pool,
    )
    .expect("lower Codex");
    let replay = bundle.programs[0].replay.as_ref().expect("replay");
    assert_eq!(bundle.metadata.warning_facts.len(), 0);
    assert_eq!(
        bundle.programs[0].profiling.graph.nodes["llm_0"]
            .as_llm()
            .expect("llm")
            .request
            .as_ref()
            .expect("request")
            .model
            .as_deref(),
        sessions[0].model.as_deref()
    );
    for (key, value) in [
        ("source_format", json!("codex")),
        ("request_wire_exact", json!(false)),
        ("tool_schema_available", json!(false)),
        ("output_tokens_available", json!(false)),
        ("model_latency_available", json!(false)),
        ("reasoning_included", json!(false)),
        ("tool_results_complete", json!(true)),
        ("subagent_topology", json!("none")),
        ("ignored_record_count", json!(4)),
        ("omitted_reasoning_count", json!(1)),
        ("cwd_present", json!(true)),
        ("git_branch_present", json!(true)),
    ] {
        assert_eq!(replay.comparability_annotations[key], value);
    }
}

#[test]
fn recorded_agent_auto_directory_is_rejected() {
    let error = discover_imported_agent_read_set(
        &fixture("codex"),
        None,
        RecordedAgentSourceFormat::Auto,
        None,
    )
    .expect_err("auto directories require explicit source format");
    assert!(error.to_string().contains("explicit source_format"));
}

#[test]
fn recorded_agent_adapter_auto_directory_requires_explicit_source() {
    let adapter = RecordedAgentRunnerGraphInputAdapter;
    let tokenizer = TiktokenTokenizer::builtin();
    let raw = serde_json::value::to_raw_value(&json!({
        "type": "file", "format": "agent_recording", "path": fixture("codex"),
        "sampling": "sequential", "graph": {"source_format": "auto"},
    }))
    .expect("raw graph input");
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");
    let error = runtime
        .block_on(adapter.load(
            &raw,
            &GraphInputContext {
                tokenizer: &tokenizer,
                run_random_seed: Some(7),
                endpoint_id: "chat",
            },
        ))
        .expect_err("auto directory must fail before strict discovery");
    assert!(error.to_string().contains("explicit source_format"));
}

#[test]
fn recorded_agent_adapter_mini_swe_rejects_jsonl() {
    let adapter = RecordedAgentRunnerGraphInputAdapter;
    let tokenizer = TiktokenTokenizer::builtin();
    let raw = serde_json::value::to_raw_value(&json!({
        "type": "file", "format": "agent_recording", "path": fixture("codex/linear.jsonl"),
        "sampling": "sequential", "graph": {"source_format": "mini_swe_agent"},
    }))
    .expect("raw graph input");
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");
    let error = runtime
        .block_on(adapter.load(
            &raw,
            &GraphInputContext {
                tokenizer: &tokenizer,
                run_random_seed: Some(7),
                endpoint_id: "chat",
            },
        ))
        .expect_err("Mini-SWE must reject JSONL before strict discovery");
    assert!(error.to_string().contains("rejects JSONL"));
}

#[test]
fn recorded_agent_adapter_auto_sniffs_codex_jsonl() {
    let adapter = RecordedAgentRunnerGraphInputAdapter;
    let tokenizer = TiktokenTokenizer::builtin();
    let raw = serde_json::value::to_raw_value(&json!({
        "type": "file", "format": "agent_recording", "path": fixture("codex/linear.jsonl"),
        "sampling": "sequential", "graph": {"source_format": "auto"},
    }))
    .expect("raw graph input");
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");
    let prepared = runtime
        .block_on(adapter.load(
            &raw,
            &GraphInputContext {
                tokenizer: &tokenizer,
                run_random_seed: Some(7),
                endpoint_id: "chat",
            },
        ))
        .expect("bounded auto Codex import");
    assert_eq!(prepared.bundle.metadata.format, "agent_recording");
    assert_eq!(
        prepared.bundle.programs[0]
            .replay
            .as_ref()
            .expect("replay")
            .identity
            .adapter,
        "codex"
    );
}

#[test]
fn recorded_agent_adapter_auto_rejects_ambiguous_jsonl() {
    let temporary = tempfile::tempdir().expect("temporary source root");
    let source = temporary.path().join("ambiguous.jsonl");
    fs::write(
        &source,
        r#"{"type":"session_meta","payload":{},"sessionId":"ambiguous","parentUuid":null}"#,
    )
    .expect("write ambiguous source");
    let adapter = RecordedAgentRunnerGraphInputAdapter;
    let tokenizer = TiktokenTokenizer::builtin();
    let raw = serde_json::value::to_raw_value(&json!({
        "type": "file", "format": "agent_recording", "path": source,
        "sampling": "sequential", "graph": {"source_format": "auto"},
    }))
    .expect("raw graph input");
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");
    let error = runtime
        .block_on(adapter.load(
            &raw,
            &GraphInputContext {
                tokenizer: &tokenizer,
                run_random_seed: Some(7),
                endpoint_id: "chat",
            },
        ))
        .expect_err("ambiguous Auto source must fail");
    assert!(format!("{error:#}").contains("ambiguous source markers"));
}

#[test]
fn recorded_agent_adapter_auto_uses_bounded_import_sniff_before_parser_tail_error() {
    let temporary = tempfile::tempdir().expect("temporary source root");
    let source = temporary.path().join("bounded.jsonl");
    let first_record = fs::read_to_string(fixture("codex/linear.jsonl"))
        .expect("read fixture")
        .lines()
        .next()
        .expect("first fixture record")
        .to_owned();
    let mut contents = std::iter::repeat_n(first_record.as_str(), 20)
        .collect::<Vec<_>>()
        .join("\n");
    contents.push_str("\n{malformed-secret-tail\n");
    fs::write(&source, contents).expect("write bounded source");
    let adapter = RecordedAgentRunnerGraphInputAdapter;
    let tokenizer = TiktokenTokenizer::builtin();
    let raw = serde_json::value::to_raw_value(&json!({
        "type": "file", "format": "agent_recording", "path": source,
        "sampling": "sequential", "graph": {"source_format": "auto"},
    }))
    .expect("raw graph input");
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");
    let error = runtime
        .block_on(adapter.load(
            &raw,
            &GraphInputContext {
                tokenizer: &tokenizer,
                run_random_seed: Some(7),
                endpoint_id: "chat",
            },
        ))
        .expect_err("parser must report the tail after bounded Auto sniffing");
    let message = format!("{error:#}");
    assert!(message.contains("parsing imported recorded-agent session input"));
    assert!(message.contains("invalid JSON"));
    assert!(!message.contains("decoding recorded-agent input"));
}

#[test]
fn recorded_agent_adapter_auto_sniffs_claude_before_endpoint_preflight() {
    let adapter = RecordedAgentRunnerGraphInputAdapter;
    let tokenizer = TiktokenTokenizer::builtin();
    let raw = serde_json::value::to_raw_value(&json!({
        "type": "file", "format": "agent_recording", "path": fixture("claude_code/linear.jsonl"),
        "sampling": "sequential", "graph": {"source_format": "auto"},
    }))
    .expect("raw graph input");
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");
    let chat = runtime.block_on(adapter.load(
        &raw,
        &GraphInputContext {
            tokenizer: &tokenizer,
            run_random_seed: Some(7),
            endpoint_id: "chat",
        },
    ));
    assert!(
        chat.expect_err("Auto Claude rejects chat")
            .to_string()
            .contains("messages")
    );
    assert!(
        runtime
            .block_on(adapter.load(
                &raw,
                &GraphInputContext {
                    tokenizer: &tokenizer,
                    run_random_seed: Some(7),
                    endpoint_id: "messages",
                },
            ))
            .is_ok()
    );
}

#[tokio::test]
async fn local_graph_inspection_forwards_selected_endpoint_identity() {
    let resolver = BuiltinRunnerGraphInputAdapterResolver::new();
    let tokenizer = TiktokenTokenizer::builtin();
    let chat = prepare_local_graph_inspection_input(
        &resolver,
        &fixture("claude_code/linear.jsonl"),
        "agent_recording",
        &tokenizer,
        "chat",
        None,
        7,
    )
    .await;
    assert!(
        chat.expect_err("Claude chat preflight")
            .to_string()
            .contains("messages")
    );
    assert!(
        prepare_local_graph_inspection_input(
            &resolver,
            &fixture("claude_code/linear.jsonl"),
            "agent_recording",
            &tokenizer,
            "messages",
            None,
            7,
        )
        .await
        .is_ok()
    );
}

#[test]
fn imported_claude_subagent_metadata_retains_parent_identity() {
    let read_set = discover_imported_agent_read_set(
        &fixture("claude_code/with_subagent"),
        None,
        RecordedAgentSourceFormat::ClaudeCode,
        None,
    )
    .expect("discover Claude fixture");
    assert_eq!(read_set.source, ImportedAgentSource::ClaudeCode);
    assert!(
        read_set
            .files
            .iter()
            .any(|file| file.family == ImportedSessionFamily::Subagent)
    );
    let sessions = parse_imported_agent_sessions(&read_set).expect("parse Claude fixture");
    let mut pool = SegmentPool::new();
    let tokenizer = builtin_tokenizer();
    let bundle = lower_imported_agent_sessions(
        &sessions,
        &BuiltinReplayRequestProfileResolver::default(),
        &tokenizer,
        &mut pool,
    )
    .expect("lower imported Claude sessions");
    let subagent = bundle
        .programs
        .iter()
        .find(|program| program.replay.as_ref().expect("replay").identity.family == "subagent")
        .expect("subagent program");
    let annotations = &subagent
        .replay
        .as_ref()
        .expect("replay")
        .comparability_annotations;
    assert_eq!(annotations["parent_session_id"], "sess-main");
    assert_eq!(annotations["parent_tool_use_id"], "toolu_task_01");
}

#[test]
fn imported_adapter_selects_sources_once_and_enforces_claude_messages_endpoint() {
    let adapter = RecordedAgentRunnerGraphInputAdapter;
    let tokenizer = TiktokenTokenizer::builtin();
    let raw = serde_json::value::to_raw_value(&json!({
        "type": "file",
        "format": "agent_recording",
        "path": fixture("claude_code/linear.jsonl"),
        "sampling": "sequential",
        "graph": {"source_format": "claude_code"}
    }))
    .expect("raw graph input");
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");
    let chat = runtime.block_on(adapter.load(
        &raw,
        &GraphInputContext {
            tokenizer: &tokenizer,
            run_random_seed: Some(77),
            endpoint_id: "chat",
        },
    ));
    assert!(
        chat.expect_err("Claude requires messages endpoint")
            .to_string()
            .contains("messages")
    );
    let prepared = runtime
        .block_on(adapter.load(
            &raw,
            &GraphInputContext {
                tokenizer: &tokenizer,
                run_random_seed: Some(77),
                endpoint_id: "messages",
            },
        ))
        .expect("Claude messages import");
    assert_eq!(prepared.random_seed, None);
    assert!(!prepared.allow_dataset_wrap);
    assert_eq!(prepared.default_output_tokens, 32_768);
    assert_eq!(prepared.cache_bust_target, CacheBustTarget::None);
}
