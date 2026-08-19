// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Claude Code JSONL normalization contracts for imported recorded-agent sessions.

use std::fs;
use std::path::{Path, PathBuf};

use aiperf_runtime::graph::recorded::agent_recording::{
    ImportedAgentReadSet, ImportedAgentSource, ImportedAgentSourceFile, ImportedSessionFamily,
    parse_claude_session, parse_imported_agent_sessions,
};
use serde_json::Value;
use tempfile::tempdir;

fn fixture(path: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/recorded_agent_session_import/claude_code")
        .join(path)
}

fn claude_file(path: PathBuf, family: ImportedSessionFamily) -> ImportedAgentSourceFile {
    ImportedAgentSourceFile {
        relative_path: path.file_name().expect("fixture file name").into(),
        path,
        family,
    }
}

fn session_file(path: PathBuf) -> ImportedAgentSourceFile {
    claude_file(path, ImportedSessionFamily::Session)
}

fn messages(
    call: &aiperf_runtime::graph::recorded::agent_recording::ImportedModelCall,
) -> Vec<Value> {
    call.request_messages
        .iter()
        .map(|message| serde_json::from_slice(&message.wire).expect("canonical JSON wire"))
        .collect()
}

fn write(root: &Path, name: &str, body: &str) -> ImportedAgentSourceFile {
    let path = root.join(name);
    fs::write(&path, body).expect("write source fixture");
    session_file(path)
}

fn claude_read_set(
    root: PathBuf,
    files: Vec<(PathBuf, ImportedSessionFamily)>,
) -> ImportedAgentReadSet {
    ImportedAgentReadSet {
        selected_path: root.clone(),
        root,
        source: ImportedAgentSource::ClaudeCode,
        files: files
            .into_iter()
            .map(|(path, family)| claude_file(path, family))
            .collect(),
    }
}

fn fixture_read_set(name: &str) -> ImportedAgentReadSet {
    let root = fixture(&format!("../adversarial/claude_code/subagents/{name}"));
    let mut files = vec![(root.join("main.jsonl"), ImportedSessionFamily::Session)];
    if name == "duplicate_final_session" {
        files.push((root.join("other.jsonl"), ImportedSessionFamily::Session));
    } else {
        files.push((
            root.join("main/subagents/agent-aaa.jsonl"),
            ImportedSessionFamily::Subagent,
        ));
        if name == "duplicate_subagent" {
            files.push((
                root.join("main/subagents/agent-bbb.jsonl"),
                ImportedSessionFamily::Subagent,
            ));
        }
        if name == "multiple_main_matches" {
            files.insert(
                1,
                (root.join("other.jsonl"), ImportedSessionFamily::Session),
            );
        }
    }
    claude_read_set(root, files)
}

#[test]
fn subagent_sessions_are_linked_as_deterministic_siblings() {
    let root = fixture("with_subagent");
    let sessions = parse_imported_agent_sessions(&claude_read_set(
        root.clone(),
        vec![
            (root.join("main.jsonl"), ImportedSessionFamily::Session),
            (
                root.join("main/subagents/agent-aaa.jsonl"),
                ImportedSessionFamily::Subagent,
            ),
        ],
    ))
    .expect("linked sessions");
    assert_eq!(sessions.len(), 2);
    assert_eq!(sessions[0].session_id, "sess-main");
    assert_eq!(sessions[1].session_id, "sess-main#sa#toolu_task_01");
    assert_eq!(
        sessions[1].parent.as_ref().expect("parent").session_id,
        "sess-main"
    );
    assert_eq!(
        sessions[1].parent.as_ref().expect("parent").tool_use_id,
        "toolu_task_01"
    );
}

#[test]
fn subagent_sessions_reject_ambiguous_or_invalid_parent_links() {
    for (name, detail) in [
        ("missing_first_parent", "missing parent tool-use identifier"),
        (
            "inconsistent_parent",
            "inconsistent parent tool-use identifier",
        ),
        ("parent_not_found", "parent tool-use identifier not found"),
        (
            "parent_not_task",
            "parent tool-use does not identify a Task call",
        ),
        ("duplicate_main_task", "duplicate Task tool-use identifier"),
        (
            "duplicate_subagent",
            "multiple subagent files identify one parent Task call",
        ),
        (
            "multiple_main_matches",
            "parent tool-use identifier matches multiple main sessions",
        ),
        (
            "duplicate_final_session",
            "duplicate imported session identifier",
        ),
    ] {
        let error = parse_imported_agent_sessions(&fixture_read_set(name))
            .expect_err(name)
            .to_string();
        assert!(error.contains(detail), "{name}: {error}");
        assert!(
            !error.contains("PRIVATE_"),
            "{name} leaked source data: {error}"
        );
    }
}

#[test]
fn subagent_sessions_never_open_excluded_sidechain_files() {
    let root = tempdir().expect("temporary fixture");
    fs::create_dir_all(root.path().join("main/subagents")).expect("subagent directory");
    fs::write(
        root.path().join("main.jsonl"),
        concat!(
            "{\"type\":\"user\",\"isSidechain\":false,\"sessionId\":\"safe\",\"uuid\":\"u\",\"message\":{\"role\":\"user\",\"content\":\"ask\"}}\n",
            "{\"type\":\"assistant\",\"isSidechain\":false,\"sessionId\":\"safe\",\"uuid\":\"a\",\"message\":{\"role\":\"assistant\",\"id\":\"msg\",\"content\":[]}}\n"
        ),
    )
    .expect("main fixture");
    fs::write(
        root.path().join("main/subagents/agent-aaa.jsonl"),
        "PRIVATE_UNREADABLE_NOT_JSON\n",
    )
    .expect("unreadable sidechain fixture");
    let sessions = parse_imported_agent_sessions(&claude_read_set(
        root.path().to_path_buf(),
        vec![(
            root.path().join("main.jsonl"),
            ImportedSessionFamily::Session,
        )],
    ))
    .expect("excluded subagent is never opened");
    assert_eq!(sessions.len(), 1);
}

#[test]
fn main_linear_history_is_systemless_and_uses_first_metadata() {
    let session = parse_claude_session(&session_file(fixture("linear.jsonl"))).expect("session");
    assert_eq!(session.model.as_deref(), Some("claude-opus-4-6"));
    assert_eq!(session.calls.len(), 2);
    assert!(session.system_prompt.is_none());
    assert_eq!(
        session.calls[0]
            .request_messages
            .iter()
            .map(|message| message.role.as_str())
            .collect::<Vec<_>>(),
        ["user"],
    );
    assert_eq!(
        session.calls[1]
            .request_messages
            .iter()
            .map(|message| message.role.as_str())
            .collect::<Vec<_>>(),
        ["user", "assistant", "user"],
    );
    assert!(session.cwd_present && session.git_branch_present);
    assert!(session.calls.iter().all(|call| !call.tool_schema_available));
    assert!(
        session
            .calls
            .iter()
            .all(|call| call.output_tokens.is_none())
    );
}

#[test]
fn main_parallel_tools_retain_provider_blocks_and_delay() {
    let session =
        parse_claude_session(&session_file(fixture("parallel_tools.jsonl"))).expect("session");
    assert_eq!(session.calls.len(), 2);
    let second = messages(&session.calls[1]);
    assert_eq!(second[1]["content"][0]["type"], "tool_use");
    assert_eq!(second[1]["content"][1]["id"], "toolu_02");
    assert_eq!(second[2]["content"][0]["type"], "tool_result");
    assert_eq!(session.calls[1].delay_after_previous_us, Some(500_000.0));
    assert_eq!(session.observed_tool_count, 2);
}

#[test]
fn main_merges_repeated_assistant_snapshots_without_extra_calls() {
    let root = tempdir().expect("temporary fixtures");
    let file = write(
        root.path(),
        "snapshots.jsonl",
        concat!(
            "{\"type\":\"user\",\"isSidechain\":false,\"sessionId\":\"safe-session\",\"uuid\":\"u-1\",\"message\":{\"role\":\"user\",\"content\":\"ask\"}}\n",
            "{\"type\":\"assistant\",\"isSidechain\":false,\"sessionId\":\"safe-session\",\"uuid\":\"a-1\",\"message\":{\"role\":\"assistant\",\"id\":\"msg-1\",\"model\":\"claude\",\"content\":[{\"type\":\"text\",\"text\":\"hel\"}]}}\n",
            "{\"type\":\"assistant\",\"isSidechain\":false,\"sessionId\":\"safe-session\",\"uuid\":\"a-2\",\"message\":{\"role\":\"assistant\",\"id\":\"msg-1\",\"model\":\"other\",\"content\":[{\"type\":\"text\",\"text\":\"hello\"},{\"type\":\"tool_use\",\"id\":\"tool-1\",\"name\":\"Read\",\"input\":{\"path\":\"x\"}}]}}\n",
            "{\"type\":\"user\",\"isSidechain\":false,\"sessionId\":\"safe-session\",\"uuid\":\"u-2\",\"timestamp\":\"2026-04-02T00:00:01Z\",\"message\":{\"role\":\"user\",\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":\"tool-1\",\"content\":\"done\"}]}}\n",
            "{\"type\":\"assistant\",\"isSidechain\":false,\"sessionId\":\"safe-session\",\"uuid\":\"a-3\",\"message\":{\"role\":\"assistant\",\"id\":\"msg-2\",\"content\":[{\"type\":\"text\",\"text\":\"next\"}]}}\n"
        ),
    );
    let session = parse_claude_session(&file).expect("merged session");
    assert_eq!(session.model.as_deref(), Some("claude"));
    assert_eq!(session.calls.len(), 2);
    assert_eq!(session.calls[0].source_id, "msg-1");
    let history = messages(&session.calls[1]);
    assert_eq!(history[1]["content"][0]["text"], "hello");
    assert_eq!(history[1]["content"][1]["id"], "tool-1");
    assert_eq!(history[2]["content"][0]["tool_use_id"], "tool-1");
}

#[test]
fn main_rejects_conflicts_invalid_correlations_and_never_leaks_private_values() {
    let root = tempdir().expect("temporary fixtures");
    for (name, body, detail) in [
        (
            "conflicting-text.jsonl",
            "{\"type\":\"user\",\"isSidechain\":false,\"sessionId\":\"safe\",\"uuid\":\"u\",\"message\":{\"role\":\"user\",\"content\":\"PRIVATE_PROMPT\"}}\n{\"type\":\"assistant\",\"isSidechain\":false,\"sessionId\":\"safe\",\"uuid\":\"a\",\"message\":{\"role\":\"assistant\",\"id\":\"msg\",\"content\":[{\"type\":\"text\",\"text\":\"one\"}]}}\n{\"type\":\"assistant\",\"isSidechain\":false,\"sessionId\":\"safe\",\"uuid\":\"a2\",\"message\":{\"role\":\"assistant\",\"id\":\"msg\",\"content\":[{\"type\":\"text\",\"text\":\"PRIVATE_REASONING\"}]}}\n",
            "conflicting repeated assistant text block",
        ),
        (
            "bad-result.jsonl",
            "{\"type\":\"user\",\"isSidechain\":false,\"sessionId\":\"safe\",\"uuid\":\"u\",\"message\":{\"role\":\"user\",\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":\"missing\",\"content\":\"PRIVATE_RESULT\"}]}}\n",
            "result does not identify an open tool use",
        ),
        (
            "invalid-time.jsonl",
            "{\"type\":\"user\",\"isSidechain\":false,\"sessionId\":\"safe\",\"uuid\":\"u\",\"message\":{\"role\":\"user\",\"content\":\"PRIVATE_PROMPT\"}}\n{\"type\":\"assistant\",\"isSidechain\":false,\"sessionId\":\"safe\",\"uuid\":\"a\",\"timestamp\":\"PRIVATE_TIMESTAMP\",\"message\":{\"role\":\"assistant\",\"id\":\"msg\",\"content\":[{\"type\":\"tool_use\",\"id\":\"tool\",\"name\":\"Read\",\"input\":{\"path\":\"PRIVATE_ARGUMENT\"}}]}}\n",
            "invalid timestamp",
        ),
        (
            "no-calls.jsonl",
            "{\"type\":\"user\",\"isSidechain\":false,\"sessionId\":\"safe\",\"uuid\":\"u\",\"cwd\":\"PRIVATE_BRANCH\",\"message\":{\"role\":\"user\",\"content\":\"PRIVATE_PROMPT\"}}\n",
            "no inferred model calls",
        ),
    ] {
        let error = parse_claude_session(&write(root.path(), name, body))
            .expect_err(name)
            .to_string();
        assert!(error.contains(detail), "{name}: {error}");
        for private in [
            "PRIVATE_PROMPT",
            "PRIVATE_REASONING",
            "PRIVATE_RESULT",
            "PRIVATE_TIMESTAMP",
            "PRIVATE_CWD",
            "PRIVATE_BRANCH",
        ] {
            assert!(!error.contains(private), "{name} leaked {private}: {error}");
        }
    }
}

#[test]
fn main_filters_sidechains_and_validates_subagent_parent() {
    let root = tempdir().expect("temporary fixtures");
    let file = write(
        root.path(),
        "mixed.jsonl",
        concat!(
            "{\"type\":\"user\",\"isSidechain\":true,\"sessionId\":\"safe\",\"uuid\":\"side-u\",\"message\":{\"role\":\"user\",\"content\":\"ignore\"}}\n",
            "{\"type\":\"user\",\"isSidechain\":false,\"sessionId\":\"safe\",\"uuid\":\"u\",\"message\":{\"role\":\"user\",\"content\":\"keep\"}}\n",
            "{\"type\":\"assistant\",\"isSidechain\":false,\"sessionId\":\"safe\",\"uuid\":\"a\",\"message\":{\"role\":\"assistant\",\"id\":\"msg\",\"content\":[{\"type\":\"text\",\"text\":\"answer\"}]}}\n"
        ),
    );
    let session = parse_claude_session(&file).expect("main session");
    assert_eq!(session.calls.len(), 1);
    assert!(session.parent.is_none());
    assert_eq!(session.ignored_record_count, 1);
    assert_eq!(messages(&session.calls[0])[0]["content"], "keep");

    let subagent = parse_claude_session(&claude_file(
        fixture("with_subagent/main/subagents/agent-aaa.jsonl"),
        ImportedSessionFamily::Subagent,
    ))
    .expect("subagent session");
    let parent = subagent.parent.expect("subagent parent");
    assert_eq!(parent.session_id, "sess-main");
    assert_eq!(parent.tool_use_id, "toolu_task_01");
}

#[test]
fn main_validates_ids_merges_exact_tool_snapshots_and_counts_omissions() {
    let root = tempdir().expect("temporary fixtures");
    let fallback = write(
        root.path(),
        "uuid-fallback.jsonl",
        concat!(
            "{\"type\":\"user\",\"isSidechain\":false,\"sessionId\":\"safe\",\"uuid\":\"u\",\"message\":{\"role\":\"user\",\"content\":\"ask\"}}\n",
            "{\"type\":\"assistant\",\"isSidechain\":false,\"sessionId\":\"safe\",\"uuid\":\"uuid-fallback\",\"message\":{\"role\":\"assistant\",\"id\":\"msg\",\"content\":[{\"type\":\"tool_use\",\"id\":\"tool\",\"name\":\"Read\",\"input\":{}},{\"type\":\"thinking\",\"thinking\":\"PRIVATE_REASONING\"},{\"type\":\"unknown\",\"value\":\"PRIVATE_VALUE\"}]}}\n",
            "{\"type\":\"assistant\",\"isSidechain\":false,\"sessionId\":\"safe\",\"uuid\":\"new-uuid\",\"message\":{\"role\":\"assistant\",\"id\":\"msg\",\"content\":[{\"type\":\"tool_use\",\"id\":\"tool\",\"name\":\"Read\",\"input\":{}}]}}\n"
        ),
    );
    let session = parse_claude_session(&fallback).expect("uuid fallback session");
    assert_eq!(session.calls[0].source_id, "msg");
    assert_eq!(session.observed_tool_count, 1);
    assert_eq!(session.omitted_reasoning_count, 1);
    assert_eq!(session.ignored_record_count, 1);

    let uuid_fallback = write(
        root.path(),
        "uuid-source-id.jsonl",
        "{\"type\":\"user\",\"isSidechain\":false,\"sessionId\":\"safe\",\"uuid\":\"u\",\"message\":{\"role\":\"user\",\"content\":\"ask\"}}\n{\"type\":\"assistant\",\"isSidechain\":false,\"sessionId\":\"safe\",\"uuid\":\"uuid-fallback\",\"message\":{\"role\":\"assistant\",\"content\":[]}}\n",
    );
    assert_eq!(
        parse_claude_session(&uuid_fallback)
            .expect("uuid source-id fallback")
            .calls[0]
            .source_id,
        "uuid-fallback"
    );

    for (name, body, detail) in [
        (
            "no-assistant-id.jsonl",
            "{\"type\":\"user\",\"isSidechain\":false,\"sessionId\":\"safe\",\"uuid\":\"u\",\"message\":{\"role\":\"user\",\"content\":\"ask\"}}\n{\"type\":\"assistant\",\"isSidechain\":false,\"sessionId\":\"safe\",\"message\":{\"role\":\"assistant\",\"content\":[]}}\n",
            "invalid message identifier",
        ),
        (
            "tool-conflict.jsonl",
            "{\"type\":\"user\",\"isSidechain\":false,\"sessionId\":\"safe\",\"uuid\":\"u\",\"message\":{\"role\":\"user\",\"content\":\"ask\"}}\n{\"type\":\"assistant\",\"isSidechain\":false,\"sessionId\":\"safe\",\"uuid\":\"a\",\"message\":{\"role\":\"assistant\",\"id\":\"msg\",\"content\":[{\"type\":\"tool_use\",\"id\":\"tool\",\"name\":\"Read\",\"input\":{}}]}}\n{\"type\":\"assistant\",\"isSidechain\":false,\"sessionId\":\"safe\",\"uuid\":\"b\",\"message\":{\"role\":\"assistant\",\"id\":\"msg\",\"content\":[{\"type\":\"tool_use\",\"id\":\"tool\",\"name\":\"Bash\",\"input\":{}}]}}\n",
            "conflicting tool-use identifier reuse",
        ),
        (
            "inconsistent-session.jsonl",
            "{\"type\":\"user\",\"isSidechain\":false,\"sessionId\":\"safe-one\",\"uuid\":\"u\",\"message\":{\"role\":\"user\",\"content\":\"ask\"}}\n{\"type\":\"assistant\",\"isSidechain\":false,\"sessionId\":\"safe-two\",\"uuid\":\"a\",\"message\":{\"role\":\"assistant\",\"id\":\"msg\",\"content\":[]}}\n",
            "inconsistent session identifier",
        ),
    ] {
        let error = parse_claude_session(&write(root.path(), name, body))
            .expect_err(name)
            .to_string();
        assert!(error.contains(detail), "{name}: {error}");
        assert!(!error.contains("PRIVATE_VALUE"));
    }
}

#[test]
fn main_accepts_reverse_tool_results_and_rejects_duplicate_results() {
    let root = tempdir().expect("temporary fixtures");
    let reverse = write(
        root.path(),
        "reverse-results.jsonl",
        concat!(
            "{\"type\":\"user\",\"isSidechain\":false,\"sessionId\":\"safe\",\"uuid\":\"u\",\"message\":{\"role\":\"user\",\"content\":\"ask\"}}\n",
            "{\"type\":\"assistant\",\"isSidechain\":false,\"sessionId\":\"safe\",\"uuid\":\"a\",\"timestamp\":\"2026-04-02T00:00:00Z\",\"message\":{\"role\":\"assistant\",\"id\":\"msg-1\",\"content\":[{\"type\":\"tool_use\",\"id\":\"one\",\"name\":\"Read\",\"input\":{}},{\"type\":\"tool_use\",\"id\":\"two\",\"name\":\"Bash\",\"input\":{}}]}}\n",
            "{\"type\":\"user\",\"isSidechain\":false,\"sessionId\":\"safe\",\"uuid\":\"r\",\"timestamp\":\"2026-04-02T00:00:01Z\",\"message\":{\"role\":\"user\",\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":\"two\",\"content\":\"two\"},{\"type\":\"tool_result\",\"tool_use_id\":\"one\",\"content\":\"one\"}]}}\n",
            "{\"type\":\"assistant\",\"isSidechain\":false,\"sessionId\":\"safe\",\"uuid\":\"b\",\"message\":{\"role\":\"assistant\",\"id\":\"msg-2\",\"content\":[]}}\n"
        ),
    );
    let session = parse_claude_session(&reverse).expect("reverse results");
    assert!(session.tool_results_complete);
    assert_eq!(session.calls[1].delay_after_previous_us, Some(1_000_000.0));
    let history = messages(&session.calls[1]);
    assert_eq!(history[2]["content"][0]["tool_use_id"], "two");
    assert_eq!(history[2]["content"][1]["tool_use_id"], "one");

    let duplicate = write(
        root.path(),
        "duplicate-result.jsonl",
        concat!(
            "{\"type\":\"user\",\"isSidechain\":false,\"sessionId\":\"safe\",\"uuid\":\"u\",\"message\":{\"role\":\"user\",\"content\":\"ask\"}}\n",
            "{\"type\":\"assistant\",\"isSidechain\":false,\"sessionId\":\"safe\",\"uuid\":\"a\",\"message\":{\"role\":\"assistant\",\"id\":\"msg\",\"content\":[{\"type\":\"tool_use\",\"id\":\"tool\",\"name\":\"Read\",\"input\":{}}]}}\n",
            "{\"type\":\"user\",\"isSidechain\":false,\"sessionId\":\"safe\",\"uuid\":\"r\",\"message\":{\"role\":\"user\",\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":\"tool\",\"content\":\"one\"},{\"type\":\"tool_result\",\"tool_use_id\":\"tool\",\"content\":\"two\"}]}}\n"
        ),
    );
    let error = parse_claude_session(&duplicate)
        .expect_err("duplicate result")
        .to_string();
    assert!(error.contains("duplicate result identifier"));
}

#[test]
fn main_adversarial_fixtures_keep_private_values_out_of_diagnostics() {
    for (name, detail) in [
        (
            "repeated_text_conflict.jsonl",
            "conflicting repeated assistant text block",
        ),
        (
            "dangling_result.jsonl",
            "result does not identify an open tool use",
        ),
        ("invalid_timestamp.jsonl", "invalid timestamp"),
        (
            "six_sentinel_error.jsonl",
            "conflicting tool-use identifier reuse",
        ),
    ] {
        let error = parse_claude_session(&session_file(fixture(&format!(
            "../adversarial/claude_code/{name}"
        ))))
        .expect_err(name)
        .to_string();
        assert!(error.contains(detail), "{name}: {error}");
        for private in [
            "PRIVATE_PROMPT",
            "PRIVATE_REASONING",
            "PRIVATE_RESULT",
            "PRIVATE_TIMESTAMP",
            "PRIVATE_CWD",
            "PRIVATE_BRANCH",
            "PRIVATE_ARGUMENT",
        ] {
            assert!(!error.contains(private), "{name} leaked {private}: {error}");
        }
    }
}

#[test]
fn main_does_not_reopen_finalized_assistant_or_exact_tool_blocks() {
    let root = tempdir().expect("temporary fixtures");
    let file = write(
        root.path(),
        "finalized-repeat.jsonl",
        concat!(
            "{\"type\":\"user\",\"isSidechain\":false,\"sessionId\":\"safe\",\"uuid\":\"u1\",\"message\":{\"role\":\"user\",\"content\":\"first\"}}\n",
            "{\"type\":\"assistant\",\"isSidechain\":false,\"sessionId\":\"safe\",\"uuid\":\"a1\",\"message\":{\"role\":\"assistant\",\"id\":\"msg-1\",\"content\":[{\"type\":\"tool_use\",\"id\":\"tool-1\",\"name\":\"Read\",\"input\":{\"path\":\"PRIVATE_ARGUMENT\"}}]}}\n",
            "{\"type\":\"user\",\"isSidechain\":false,\"sessionId\":\"safe\",\"uuid\":\"r1\",\"message\":{\"role\":\"user\",\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":\"tool-1\",\"content\":\"PRIVATE_RESULT\"}]}}\n",
            "{\"type\":\"assistant\",\"isSidechain\":false,\"sessionId\":\"safe\",\"uuid\":\"a2\",\"message\":{\"role\":\"assistant\",\"id\":\"msg-2\",\"content\":[]}}\n",
            "{\"type\":\"assistant\",\"isSidechain\":false,\"sessionId\":\"safe\",\"uuid\":\"a3\",\"message\":{\"role\":\"assistant\",\"id\":\"msg-1\",\"content\":[{\"type\":\"tool_use\",\"id\":\"tool-1\",\"name\":\"Read\",\"input\":{\"path\":\"PRIVATE_ARGUMENT\"}}]}}\n",
            "{\"type\":\"assistant\",\"isSidechain\":false,\"sessionId\":\"safe\",\"uuid\":\"a4\",\"message\":{\"role\":\"assistant\",\"id\":\"msg-3\",\"content\":[{\"type\":\"tool_use\",\"id\":\"tool-1\",\"name\":\"Read\",\"input\":{\"path\":\"PRIVATE_ARGUMENT\"}}]}}\n",
            "{\"type\":\"assistant\",\"isSidechain\":false,\"sessionId\":\"safe\",\"uuid\":\"a5\",\"message\":{\"role\":\"assistant\",\"id\":\"msg-4\",\"content\":[]}}\n"
        ),
    );
    let session = parse_claude_session(&file).expect("finalized duplicate is ignored");
    assert_eq!(session.calls.len(), 4);
    assert_eq!(session.observed_tool_count, 1);
    assert!(session.tool_results_complete);
    assert_eq!(session.calls[0].source_id, "msg-1");
    assert_eq!(session.calls[1].source_id, "msg-2");
    assert_eq!(session.calls[2].source_id, "msg-3");
    assert_eq!(session.calls[3].source_id, "msg-4");
    assert_eq!(
        messages(&session.calls[3])[4]["content"],
        Value::Array(Vec::new())
    );
}

#[test]
fn main_replays_a_finalized_filtered_global_tool_snapshot_idempotently() {
    let root = tempdir().expect("temporary fixtures");
    let file = write(
        root.path(),
        "filtered-finalized-replay.jsonl",
        concat!(
            "{\"type\":\"user\",\"isSidechain\":false,\"sessionId\":\"safe\",\"uuid\":\"u\",\"message\":{\"role\":\"user\",\"content\":\"ask\"}}\n",
            "{\"type\":\"assistant\",\"isSidechain\":false,\"sessionId\":\"safe\",\"uuid\":\"a\",\"message\":{\"role\":\"assistant\",\"id\":\"msg-a\",\"content\":[{\"type\":\"tool_use\",\"id\":\"tool\",\"name\":\"Read\",\"input\":{}}]}}\n",
            "{\"type\":\"user\",\"isSidechain\":false,\"sessionId\":\"safe\",\"uuid\":\"r\",\"message\":{\"role\":\"user\",\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":\"tool\",\"content\":\"result\"}]}}\n",
            "{\"type\":\"assistant\",\"isSidechain\":false,\"sessionId\":\"safe\",\"uuid\":\"b\",\"message\":{\"role\":\"assistant\",\"id\":\"msg-b\",\"content\":[{\"type\":\"tool_use\",\"id\":\"tool\",\"name\":\"Read\",\"input\":{}}]}}\n",
            "{\"type\":\"assistant\",\"isSidechain\":false,\"sessionId\":\"safe\",\"uuid\":\"flush\",\"message\":{\"role\":\"assistant\",\"id\":\"msg-c\",\"content\":[]}}\n",
            "{\"type\":\"assistant\",\"isSidechain\":false,\"sessionId\":\"safe\",\"uuid\":\"replay\",\"message\":{\"role\":\"assistant\",\"id\":\"msg-b\",\"content\":[{\"type\":\"tool_use\",\"id\":\"tool\",\"name\":\"Read\",\"input\":{}}]}}\n"
        ),
    );
    let session = parse_claude_session(&file).expect("exact replay is idempotent");
    assert_eq!(session.calls.len(), 3);
    assert!(session.tool_results_complete);
    assert_eq!(
        messages(&session.calls[2])[3]["content"],
        Value::Array(Vec::new())
    );
}

#[test]
fn main_limits_timestamp_validation_to_correlated_tools_and_latches_metadata() {
    let root = tempdir().expect("temporary fixtures");
    let ordinary = write(
        root.path(),
        "ordinary-invalid-time.jsonl",
        concat!(
            "{\"type\":\"permission-mode\",\"sessionId\":\"PRIVATE_SESSION\"}\n",
            "{\"type\":\"unknown-metadata\",\"sessionId\":\"PRIVATE_SESSION\"}\n",
            "{\"type\":\"user\",\"isSidechain\":false,\"sessionId\":\"safe\",\"uuid\":\"u1\",\"timestamp\":\"PRIVATE_TIMESTAMP\",\"message\":{\"role\":\"user\",\"content\":\"PRIVATE_PROMPT\"}}\n",
            "{\"type\":\"assistant\",\"isSidechain\":false,\"sessionId\":\"safe\",\"uuid\":\"a1\",\"timestamp\":\"PRIVATE_TIMESTAMP\",\"message\":{\"role\":\"assistant\",\"id\":\"msg-1\",\"model\":\"first-model\",\"content\":[{\"type\":\"text\",\"text\":\"same\"},{\"type\":\"redacted_thinking\",\"data\":\"PRIVATE_REASONING\"}]}}\n",
            "{\"type\":\"assistant\",\"isSidechain\":false,\"sessionId\":\"safe\",\"uuid\":\"a2\",\"cwd\":\"PRIVATE_CWD\",\"gitBranch\":\"PRIVATE_BRANCH\",\"message\":{\"role\":\"assistant\",\"id\":\"msg-1\",\"model\":\"later-model\",\"content\":[{\"type\":\"text\",\"text\":\"same\"},{\"type\":\"text\",\"text\":\"appended\"}]}}\n",
            "{\"type\":\"user\",\"isSidechain\":false,\"sessionId\":\"safe\",\"uuid\":\"u2\",\"message\":{\"role\":\"user\",\"content\":\"next\"}}\n",
            "{\"type\":\"assistant\",\"isSidechain\":false,\"sessionId\":\"safe\",\"uuid\":\"a3\",\"message\":{\"role\":\"assistant\",\"id\":\"msg-2\",\"content\":[]}}\n"
        ),
    );
    let session = parse_claude_session(&ordinary).expect("ordinary malformed timestamps ignored");
    assert_eq!(session.ignored_record_count, 2);
    assert_eq!(session.omitted_reasoning_count, 1);
    assert_eq!(session.model.as_deref(), Some("first-model"));
    assert!(session.cwd_present && session.git_branch_present);
    let history = messages(&session.calls[1]);
    assert_eq!(history[1]["content"][0]["text"], "same");
    assert_eq!(history[1]["content"][1]["text"], "appended");

    let missing_time = write(
        root.path(),
        "missing-tool-time.jsonl",
        concat!(
            "{\"type\":\"user\",\"isSidechain\":false,\"sessionId\":\"safe\",\"uuid\":\"u\",\"message\":{\"role\":\"user\",\"content\":\"ask\"}}\n",
            "{\"type\":\"assistant\",\"isSidechain\":false,\"sessionId\":\"safe\",\"uuid\":\"a\",\"message\":{\"role\":\"assistant\",\"id\":\"msg-1\",\"content\":[{\"type\":\"tool_use\",\"id\":\"tool\",\"name\":\"Read\",\"input\":{}}]}}\n",
            "{\"type\":\"user\",\"isSidechain\":false,\"sessionId\":\"safe\",\"uuid\":\"r\",\"message\":{\"role\":\"user\",\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":\"tool\",\"content\":\"ok\"}]}}\n",
            "{\"type\":\"assistant\",\"isSidechain\":false,\"sessionId\":\"safe\",\"uuid\":\"b\",\"message\":{\"role\":\"assistant\",\"id\":\"msg-2\",\"content\":[]}}\n"
        ),
    );
    assert_eq!(
        parse_claude_session(&missing_time)
            .expect("missing tool timestamps")
            .calls[1]
            .delay_after_previous_us,
        None
    );

    let first_missing_later_present = write(
        root.path(),
        "first-missing-later-present.jsonl",
        concat!(
            "{\"type\":\"user\",\"isSidechain\":false,\"sessionId\":\"safe\",\"uuid\":\"u\",\"message\":{\"role\":\"user\",\"content\":\"ask\"}}\n",
            "{\"type\":\"assistant\",\"isSidechain\":false,\"sessionId\":\"safe\",\"uuid\":\"a\",\"message\":{\"role\":\"assistant\",\"id\":\"msg-1\",\"content\":[{\"type\":\"tool_use\",\"id\":\"one\",\"name\":\"Read\",\"input\":{}}]}}\n",
            "{\"type\":\"assistant\",\"isSidechain\":false,\"sessionId\":\"safe\",\"uuid\":\"a2\",\"timestamp\":\"2026-04-02T00:00:01Z\",\"message\":{\"role\":\"assistant\",\"id\":\"msg-1\",\"content\":[{\"type\":\"tool_use\",\"id\":\"one\",\"name\":\"Read\",\"input\":{}},{\"type\":\"tool_use\",\"id\":\"two\",\"name\":\"Bash\",\"input\":{}}]}}\n",
            "{\"type\":\"user\",\"isSidechain\":false,\"sessionId\":\"safe\",\"uuid\":\"r\",\"timestamp\":\"2026-04-02T00:00:02Z\",\"message\":{\"role\":\"user\",\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":\"one\",\"content\":\"one\"},{\"type\":\"tool_result\",\"tool_use_id\":\"two\",\"content\":\"two\"}]}}\n",
            "{\"type\":\"assistant\",\"isSidechain\":false,\"sessionId\":\"safe\",\"uuid\":\"b\",\"message\":{\"role\":\"assistant\",\"id\":\"msg-2\",\"content\":[]}}\n"
        ),
    );
    assert_eq!(
        parse_claude_session(&first_missing_later_present)
            .expect("first timestamp remains missing")
            .calls[1]
            .delay_after_previous_us,
        None
    );

    let malformed_tool = write(
        root.path(),
        "malformed-tool-time.jsonl",
        "{\"type\":\"user\",\"isSidechain\":false,\"sessionId\":\"safe\",\"uuid\":\"u\",\"message\":{\"role\":\"user\",\"content\":\"ask\"}}\n{\"type\":\"assistant\",\"isSidechain\":false,\"sessionId\":\"safe\",\"uuid\":\"a\",\"timestamp\":\"PRIVATE_TIMESTAMP\",\"message\":{\"role\":\"assistant\",\"id\":\"msg\",\"content\":[{\"type\":\"tool_use\",\"id\":\"tool\",\"name\":\"Read\",\"input\":{\"path\":\"PRIVATE_ARGUMENT\"}}]}}\n",
    );
    let error = parse_claude_session(&malformed_tool)
        .expect_err("correlated malformed timestamp")
        .to_string();
    assert!(error.contains("invalid timestamp"));
    for private in [
        "PRIVATE_PROMPT",
        "PRIVATE_REASONING",
        "PRIVATE_RESULT",
        "PRIVATE_TIMESTAMP",
        "PRIVATE_CWD",
        "PRIVATE_BRANCH",
        "PRIVATE_ARGUMENT",
    ] {
        assert!(!error.contains(private), "leaked {private}: {error}");
    }
}
