// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Lowering and adapter integration coverage for imported agent sessions.

use std::path::{Path, PathBuf};

use aiperf_runtime::config::model::dataset::RecordedAgentSourceFormat;
use aiperf_runtime::dataset::{Payload, TiktokenTokenizer};
use aiperf_runtime::engine::graph_input::{
    CacheBustTarget, GraphInputAdapter, GraphInputContext, RecordedAgentRunnerGraphInputAdapter,
};
use aiperf_runtime::graph::model::{ChannelType, ExecutableGraphNode, PromptItem, ReducerName};
use aiperf_runtime::graph::recorded::agent_recording::{
    BuiltinReplayRequestProfileResolver, ImportedAgentSource, ImportedSessionFamily,
    discover_imported_agent_read_set, lower_imported_agent_sessions, parse_imported_agent_sessions,
};
use aiperf_runtime::graph::segment::SegmentPool;
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
    let resolver = BuiltinReplayRequestProfileResolver::new(true, 123, false, false, false, false)
        .expect("valid resolver");

    let bundle = lower_imported_agent_sessions(&sessions, &resolver, &mut pool)
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
    let resolver = BuiltinReplayRequestProfileResolver::default();
    let bundle = lower_imported_agent_sessions(&sessions, &resolver, &mut pool)
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
    let bundle = lower_imported_agent_sessions(
        &sessions,
        &BuiltinReplayRequestProfileResolver::default(),
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
