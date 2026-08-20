// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Codex JSONL normalization contracts for imported recorded-agent sessions.

use std::fs;
use std::path::{Path, PathBuf};

use aiperf_runtime::graph::recorded::agent_recording::{
    ImportedAgentSession, ImportedAgentSourceFile, ImportedSessionFamily, parse_codex_session,
};
use serde_json::Value;
use tempfile::tempdir;

fn fixture(path: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/recorded_agent_session_import/codex")
        .join(path)
}

fn codex_file(path: PathBuf) -> ImportedAgentSourceFile {
    ImportedAgentSourceFile {
        relative_path: path.file_name().expect("fixture file name").into(),
        path,
        family: ImportedSessionFamily::Session,
    }
}

fn roles(session: &ImportedAgentSession, call_index: usize) -> Vec<&str> {
    session
        .request_messages(call_index)
        .expect("Codex request history")
        .iter()
        .map(|message| message.role.as_str())
        .collect()
}

fn messages(session: &ImportedAgentSession, call_index: usize) -> Vec<Value> {
    session
        .request_messages(call_index)
        .expect("Codex request history")
        .iter()
        .map(|message| serde_json::from_slice(&message.wire).expect("canonical JSON wire"))
        .collect()
}

fn write(root: &Path, name: &str, body: &str) -> ImportedAgentSourceFile {
    let path = root.join(name);
    fs::write(&path, body).expect("write source fixture");
    codex_file(path)
}

#[test]
fn codex_linear_history_is_canonical_and_omits_reasoning() {
    let session =
        parse_codex_session(&codex_file(fixture("linear.jsonl"))).expect("linear session");
    assert_eq!(session.session_id, "019d28a5-b4a1-7b33-ba0b-c1a7637337d9");
    assert!(session.cwd_present && session.git_branch_present);
    assert_eq!(session.calls.len(), 2);
    assert!(
        session
            .calls
            .iter()
            .all(|call| call.request_messages.is_empty())
    );
    assert_eq!(roles(&session, 0), ["system", "user"]);
    assert_eq!(roles(&session, 1), ["system", "user", "assistant", "user"]);
    assert_eq!(session.omitted_reasoning_count, 1);
    assert!((0..session.calls.len()).all(|call_index| {
        session
            .request_messages(call_index)
            .expect("Codex request history")
            .iter()
            .all(|message| !String::from_utf8_lossy(&message.wire).contains("think briefly"))
    }));
    assert_eq!(session.calls[0].source_id, "codex-line-5");
    assert_eq!(session.calls[1].source_id, "codex-line-9");
    assert!(session.calls.iter().all(|call| !call.tool_schema_available));
    assert!(
        session
            .calls
            .iter()
            .all(|call| call.output_tokens.is_none())
    );
}

#[test]
fn codex_tool_bundle_is_causal_and_preserves_result_order() {
    let session =
        parse_codex_session(&codex_file(fixture("with_tools.jsonl"))).expect("tool session");
    assert_eq!(session.calls.len(), 3);
    assert_eq!(session.observed_tool_count, 1);
    assert_eq!(session.calls[1].delay_after_previous_us, Some(250_000.0));
    assert_eq!(roles(&session, 1), ["system", "user", "assistant", "tool"]);
    assert!(session.tool_results_complete);
    let history = messages(&session, 2);
    assert_eq!(history[2]["tool_calls"][0]["function"]["name"], "shell");
    assert_eq!(history[2]["tool_calls"][0]["id"], "call_001");
    assert_eq!(history[3]["tool_call_id"], "call_001");
    assert_eq!(session.calls[2].source_id, "codex-line-7");
    assert_eq!(
        roles(&session, 2),
        ["system", "user", "assistant", "tool", "assistant", "user"]
    );
}

#[test]
fn codex_uses_session_model_then_turn_context_fallback_and_joins_blocks() {
    let root = tempdir().expect("temporary fixtures");
    let file = write(
        root.path(),
        "models.jsonl",
        concat!(
            "{\"type\":\"session_meta\",\"payload\":{\"id\":\"safe-session\",\"model\":\"session-model\",\"base_instructions\":{\"text\":\"system\"}}}\n",
            "{\"type\":\"turn_context\",\"payload\":{\"model\":\"turn-model\"}}\n",
            "{\"type\":\"response_item\",\"payload\":{\"type\":\"message\",\"role\":\"developer\",\"content\":[{\"type\":\"input_text\",\"text\":\"first\"},{\"type\":\"input_text\",\"text\":\"second\"}]}}\n",
            "{\"type\":\"response_item\",\"payload\":{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"answer\"}]}}\n"
        ),
    );
    let session = parse_codex_session(&file).expect("session model wins");
    assert_eq!(session.model.as_deref(), Some("session-model"));
    let history = messages(&session, 0);
    assert_eq!(history[1]["role"], "developer");
    assert_eq!(history[1]["content"], "first\nsecond");

    let fallback = write(
        root.path(),
        "fallback.jsonl",
        concat!(
            "{\"type\":\"session_meta\",\"payload\":{\"id\":\"safe-fallback\"}}\n",
            "{\"type\":\"turn_context\",\"payload\":{\"model\":\"turn-model\"}}\n",
            "{\"type\":\"response_item\",\"payload\":{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"answer\"}]}}\n"
        ),
    );
    assert_eq!(
        parse_codex_session(&fallback)
            .expect("fallback")
            .model
            .as_deref(),
        Some("turn-model")
    );
}

#[test]
fn codex_rejects_invalid_tool_correlation_and_metadata_without_leaking_values() {
    let root = tempdir().expect("temporary fixtures");
    for (name, body, detail) in [
        (
            "invalid-session.jsonl",
            "{\"type\":\"session_meta\",\"payload\":{\"id\":\"PRIVATE SESSION\"}}\n",
            "invalid session identifier",
        ),
        (
            "empty-call.jsonl",
            "{\"type\":\"session_meta\",\"payload\":{\"id\":\"safe\"}}\n{\"type\":\"response_item\",\"payload\":{\"type\":\"function_call\",\"name\":\"f\",\"arguments\":\"PRIVATE_ARGUMENT\",\"call_id\":\"\"}}\n",
            "invalid call identifier",
        ),
        (
            "dangling.jsonl",
            "{\"type\":\"session_meta\",\"payload\":{\"id\":\"safe\"}}\n{\"type\":\"response_item\",\"payload\":{\"type\":\"function_call_output\",\"call_id\":\"unknown\",\"output\":\"PRIVATE_RESULT\"}}\n",
            "result does not identify an open call",
        ),
        (
            "duplicate.jsonl",
            "{\"type\":\"session_meta\",\"payload\":{\"id\":\"safe\"}}\n{\"type\":\"response_item\",\"payload\":{\"type\":\"function_call\",\"name\":\"f\",\"arguments\":\"{}\",\"call_id\":\"call\"}}\n{\"type\":\"response_item\",\"payload\":{\"type\":\"function_call\",\"name\":\"f\",\"arguments\":\"{}\",\"call_id\":\"call\"}}\n",
            "duplicate call identifier",
        ),
        (
            "duplicate-result.jsonl",
            "{\"type\":\"session_meta\",\"payload\":{\"id\":\"safe\"}}\n{\"type\":\"response_item\",\"payload\":{\"type\":\"function_call\",\"name\":\"f\",\"arguments\":\"{}\",\"call_id\":\"call\"}}\n{\"type\":\"response_item\",\"payload\":{\"type\":\"function_call_output\",\"call_id\":\"call\",\"output\":\"PRIVATE_RESULT\"}}\n{\"type\":\"response_item\",\"payload\":{\"type\":\"function_call_output\",\"call_id\":\"call\",\"output\":\"PRIVATE_RESULT\"}}\n",
            "duplicate result identifier",
        ),
        (
            "inconsistent.jsonl",
            "{\"type\":\"session_meta\",\"payload\":{\"id\":\"safe-one\"}}\n{\"type\":\"session_meta\",\"payload\":{\"id\":\"safe-two\"}}\n",
            "inconsistent session identifier",
        ),
    ] {
        let error = parse_codex_session(&write(root.path(), name, body))
            .expect_err(name)
            .to_string();
        assert!(error.contains(detail));
        assert!(!error.contains("PRIVATE_ARGUMENT"));
        assert!(!error.contains("PRIVATE_RESULT"));
    }
}

#[test]
fn codex_hashes_exact_bytes_and_reports_safe_parse_errors() {
    let root = tempdir().expect("temporary fixtures");
    let body = concat!(
        "{\"type\":\"session_meta\",\"payload\":{\"id\":\"safe-hash\",\"cwd\":\"PRIVATE_CWD\",\"git\":{\"branch\":\"PRIVATE_BRANCH\"},\"base_instructions\":{\"text\":\"PRIVATE_PROMPT\"}}}\n\n",
        "{\"type\":\"response_item\",\"payload\":{\"type\":\"reasoning\",\"summary\":[{\"text\":\"PRIVATE_REASONING\"}]}}\n",
        "{\"type\":\"response_item\",\"payload\":{\"type\":\"function_call\",\"name\":\"f\",\"arguments\":\"PRIVATE_ARGUMENT\",\"call_id\":\"call-private\"}}\n",
        "{\"type\":\"response_item\",\"payload\":{\"type\":\"function_call_output\",\"call_id\":\"call-private\",\"output\":\"PRIVATE_RESULT\"}}\n",
        "{\"type\":\"response_item\",\"payload\":{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"answer\"}]}}\n"
    );
    let file = write(root.path(), "hash.jsonl", body);
    let session = parse_codex_session(&file).expect("hashed session");
    assert_eq!(
        session.source_digest,
        blake3::hash(body.as_bytes()).to_hex().to_string()
    );
    assert_eq!(session.ignored_record_count, 0);
    let malformed = write(
        root.path(),
        "private.jsonl",
        "not-json-PRIVATE_PROMPT_PRIVATE_REASONING_PRIVATE_CWD_PRIVATE_BRANCH_PRIVATE_ARGUMENT_PRIVATE_RESULT\n",
    );
    let error = parse_codex_session(&malformed)
        .expect_err("malformed")
        .to_string();
    for secret in [
        "PRIVATE_PROMPT",
        "PRIVATE_REASONING",
        "PRIVATE_CWD",
        "PRIVATE_BRANCH",
        "PRIVATE_ARGUMENT",
        "PRIVATE_RESULT",
    ] {
        assert!(!error.contains(secret));
    }
}

#[test]
fn codex_retains_eof_bundle_without_inventing_result_or_delay() {
    let root = tempdir().expect("temporary fixtures");
    let file = write(
        root.path(),
        "eof.jsonl",
        concat!(
            "{\"type\":\"session_meta\",\"payload\":{\"id\":\"safe-eof\"}}\n",
            "{\"timestamp\":\"2026-03-26T05:38:18.000Z\",\"type\":\"response_item\",\"payload\":{\"type\":\"function_call\",\"name\":\"f\",\"arguments\":\"{}\",\"call_id\":\"call-eof\"}}\n"
        ),
    );
    let session = parse_codex_session(&file).expect("unfinished bundle");
    assert_eq!(session.calls.len(), 1);
    assert_eq!(roles(&session, 0), Vec::<&str>::new());
    assert_eq!(session.calls[0].delay_after_previous_us, None);
    assert!(!session.tool_results_complete);
}

#[test]
fn codex_retains_new_assistant_text_after_results_for_the_next_bundle() {
    let session = parse_codex_session(&codex_file(
        fixture("linear.jsonl")
            .parent()
            .expect("fixture parent")
            .parent()
            .expect("session fixture root")
            .join("adversarial/codex/assistant_after_results.jsonl"),
    ))
    .expect("two tool bundles");
    let second_bundle_index = session
        .calls
        .iter()
        .position(|call| call.source_id == "codex-line-7")
        .expect("second bundle call");
    let history = messages(&session, second_bundle_index);
    assert!(history.iter().any(|message| {
        message["role"] == "assistant" && message["content"] == "first summary"
    }));
}

#[test]
fn codex_preserves_reverse_order_results_and_handles_missing_tool_timestamps() {
    let root = tempdir().expect("temporary fixtures");
    let file = write(
        root.path(),
        "parallel.jsonl",
        concat!(
            "{\"type\":\"session_meta\",\"payload\":{\"id\":\"safe-parallel\"}}\n",
            "{\"type\":\"response_item\",\"payload\":{\"type\":\"function_call\",\"name\":\"one\",\"arguments\":\"{}\",\"call_id\":\"call-one\"}}\n",
            "{\"type\":\"response_item\",\"payload\":{\"type\":\"function_call\",\"name\":\"two\",\"arguments\":\"{}\",\"call_id\":\"call-two\"}}\n",
            "{\"type\":\"response_item\",\"payload\":{\"type\":\"function_call_output\",\"call_id\":\"call-two\",\"output\":\"two\"}}\n",
            "{\"type\":\"response_item\",\"payload\":{\"type\":\"function_call_output\",\"call_id\":\"call-one\",\"output\":\"one\"}}\n",
            "{\"type\":\"response_item\",\"payload\":{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"done\"}]}}\n"
        ),
    );
    let session = parse_codex_session(&file).expect("parallel tools");
    assert_eq!(session.calls[1].delay_after_previous_us, None);
    let history = messages(&session, 1);
    assert_eq!(history[1]["tool_call_id"], "call-two");
    assert_eq!(history[2]["tool_call_id"], "call-one");
}

#[test]
fn codex_rejects_invalid_timestamps_and_safe_malformed_records() {
    let root = tempdir().expect("temporary fixtures");
    let invalid_timestamp = write(
        root.path(),
        "invalid-timestamp.jsonl",
        concat!(
            "{\"type\":\"session_meta\",\"payload\":{\"id\":\"safe-time\"}}\n",
            "{\"timestamp\":\"not-a-time\",\"type\":\"response_item\",\"payload\":{\"type\":\"function_call\",\"name\":\"f\",\"arguments\":\"PRIVATE_ARGUMENT\",\"call_id\":\"call-time\"}}\n"
        ),
    );
    assert!(
        parse_codex_session(&invalid_timestamp)
            .expect_err("invalid timestamp")
            .to_string()
            .contains("invalid timestamp")
    );
    let object = write(root.path(), "non-object.jsonl", "[]\n");
    assert!(
        parse_codex_session(&object)
            .expect_err("non-object source")
            .to_string()
            .contains("record must be a JSON object")
    );
    let invalid_utf8 = root.path().join("invalid-utf8.jsonl");
    fs::write(&invalid_utf8, b"\xff\n").expect("write invalid UTF-8");
    let error = parse_codex_session(&codex_file(invalid_utf8))
        .expect_err("invalid UTF-8")
        .to_string();
    assert!(error.contains("invalid JSON"));
    assert!(!error.contains("PRIVATE_ARGUMENT"));
}

#[test]
fn codex_ignores_additive_records_without_retaining_private_content() {
    let root = tempdir().expect("temporary fixtures");
    let file = write(
        root.path(),
        "additive.jsonl",
        concat!(
            "{\"type\":\"session_meta\",\"payload\":{\"id\":\"safe-additive\"}}\n",
            "{\"type\":\"future_record\",\"payload\":{\"private\":\"PRIVATE_PROMPT\"}}\n",
            "{\"type\":\"response_item\",\"payload\":{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"answer\"}]}}\n"
        ),
    );
    let session = parse_codex_session(&file).expect("additive source");
    assert_eq!(session.calls.len(), 1);
    assert_eq!(session.ignored_record_count, 1);
}

#[test]
fn codex_rejects_a_session_without_an_inferred_model_call() {
    let root = tempdir().expect("temporary fixtures");
    let file = write(
        root.path(),
        "no-call.jsonl",
        "{\"type\":\"session_meta\",\"payload\":{\"id\":\"safe-no-call\"}}\n",
    );
    assert!(
        parse_codex_session(&file)
            .expect_err("no model output")
            .to_string()
            .contains("no inferred model calls")
    );
}

#[test]
fn codex_keeps_tool_delay_across_history_only_messages_and_consumes_it_on_tool_output() {
    let root = tempdir().expect("temporary fixtures");
    let file = write(
        root.path(),
        "delay-through-history.jsonl",
        concat!(
            "{\"type\":\"session_meta\",\"payload\":{\"id\":\"safe-delay\"}}\n",
            "{\"timestamp\":\"2026-03-26T05:38:18.000Z\",\"type\":\"response_item\",\"payload\":{\"type\":\"function_call\",\"name\":\"one\",\"arguments\":\"{}\",\"call_id\":\"call-one\"}}\n",
            "{\"timestamp\":\"2026-03-26T05:38:18.250Z\",\"type\":\"response_item\",\"payload\":{\"type\":\"function_call_output\",\"call_id\":\"call-one\",\"output\":\"done\"}}\n",
            "{\"type\":\"response_item\",\"payload\":{\"type\":\"message\",\"role\":\"system\",\"content\":[{\"type\":\"input_text\",\"text\":\"also keep\"}]}}\n",
            "{\"type\":\"response_item\",\"payload\":{\"type\":\"message\",\"role\":\"developer\",\"content\":[{\"type\":\"input_text\",\"text\":\"keep\"}]}}\n",
            "{\"type\":\"response_item\",\"payload\":{\"type\":\"message\",\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"going\"}]}}\n",
            "{\"type\":\"response_item\",\"payload\":{\"type\":\"function_call\",\"name\":\"two\",\"arguments\":\"{}\",\"call_id\":\"call-two\"}}\n"
        ),
    );
    let session = parse_codex_session(&file).expect("delayed second tool output");
    assert_eq!(session.calls.len(), 2);
    assert_eq!(session.calls[1].source_id, "codex-line-7");
    assert_eq!(session.calls[1].delay_after_previous_us, Some(250_000.0));
}

#[test]
fn codex_accepts_empty_tool_results_and_preserves_multi_block_text_order() {
    let root = tempdir().expect("temporary fixtures");
    let file = write(
        root.path(),
        "blocks.jsonl",
        concat!(
            "{\"type\":\"session_meta\",\"payload\":{\"id\":\"safe-blocks\"}}\n",
            "{\"type\":\"response_item\",\"payload\":{\"type\":\"message\",\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"one\"},{\"type\":\"text\",\"text\":\"two\"}]}}\n",
            "{\"type\":\"response_item\",\"payload\":{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"three\"},{\"type\":\"text\",\"text\":\"four\"}]}}\n",
            "{\"type\":\"response_item\",\"payload\":{\"type\":\"function_call\",\"name\":\"f\",\"arguments\":\"{}\",\"call_id\":\"call-empty\"}}\n",
            "{\"type\":\"response_item\",\"payload\":{\"type\":\"function_call_output\",\"call_id\":\"call-empty\",\"output\":\"\"}}\n",
            "{\"type\":\"response_item\",\"payload\":{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"five\"}]}}\n"
        ),
    );
    let session = parse_codex_session(&file).expect("empty output is valid");
    let history = messages(&session, 2);
    assert_eq!(history[0]["content"], "one\ntwo");
    assert_eq!(history[1]["content"], "three\nfour");
    assert_eq!(history[3]["content"], "");
}
