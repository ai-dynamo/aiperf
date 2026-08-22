// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Product proof for imported Codex and Claude Code recorded-agent sessions.

mod common;

use std::fs;
use std::path::{Path, PathBuf};

use aiperf_runtime::dataset::{TextTokenizer, TiktokenTokenizer};
use common::*;
use serde_json::{Value, json};

const FALLBACK_MAX_TOKENS: u64 = 32_768;
const OUTPUT_TOKENS: u64 = 4;
const TTFT_MS: f64 = 10.0;
const ITL_MS: f64 = 2.0;

fn fixture(path: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../runtime/tests/fixtures/recorded_agent_session_import")
        .join(path)
}

fn imported_session_mock() -> MockServerConfig {
    let mut config = tuned_mock_config(TTFT_MS, ITL_MS);
    config.fixed_output_tokens = Some(OUTPUT_TOKENS as usize);
    config.no_tokenizer = true;
    config
}

fn write_config(
    directory: &Path,
    url: &str,
    source: &Path,
    source_format: &str,
    endpoint_type: &str,
    graph_extra: &str,
    scenario: Option<&str>,
) -> PathBuf {
    let scenario = scenario.map_or_else(String::new, |value| format!("\x20 scenario: {value}\n"));
    let config = format!(
        "schemaVersion: \"2.0\"\n\
         benchmark:\n\
        {scenario}\
        \x20 model: fixture-model\n\
        \x20 tokenizer:\n\
        \x20   name: openai/gpt-oss-120b\n\
        \x20 endpoint:\n\
        \x20   url: {url}\n\
        \x20   type: {endpoint_type}\n\
        \x20   streaming: true\n\
        \x20 dataset:\n\
        \x20   type: file\n\
        \x20   path: {source}\n\
        \x20   format: agent_recording\n\
        \x20   sampling: sequential\n\
        \x20   graph:\n\
        \x20     source_format: {source_format}\n\
        {graph_extra}\
        \x20 profiling:\n\
        \x20   type: concurrency\n\
        \x20   sessions: 1\n\
        \x20   concurrency: 1\n\
        \x20 artifacts:\n\
        \x20   records:\n\
        \x20     - jsonl\n\
        \x20   raw: true\n\
         runtime:\n\
        \x20 workers: 1\n\
        \x20 ui: none\n",
        source = source.display(),
    );
    let path = directory.join("recorded-agent-session-import.yaml");
    fs::write(&path, config).expect("write imported-session config");
    path
}

fn run_import(
    harness: &AIPerfHarness,
    source: &Path,
    source_format: &str,
    endpoint_type: &str,
    graph_extra: &str,
    scenario: Option<&str>,
) -> RunResult {
    let directory = tempfile::tempdir().expect("temporary imported-session config directory");
    let config = write_config(
        directory.path(),
        &harness.mock.url,
        source,
        source_format,
        endpoint_type,
        graph_extra,
        scenario,
    );
    harness.run(&format!(
        "--config {} --export-level raw --ui none",
        config.display()
    ))
}

fn expected_codex_call_0() -> Value {
    json!([
        {"role": "system", "content": "You are Codex…"},
        {"role": "user", "content": "Write fizzbuzz."},
    ])
}

fn expected_codex_call_1() -> Value {
    json!([
        {"role": "system", "content": "You are Codex…"},
        {"role": "user", "content": "Write fizzbuzz."},
        {"role": "assistant", "content": "Sure, here it is."},
        {"role": "user", "content": "Now in Go."},
    ])
}

fn expected_claude_call_0() -> Value {
    json!([{"role": "user", "content": "Read foo.txt and run ls."}])
}

fn expected_claude_call_1() -> Value {
    json!([
        {"role": "user", "content": "Read foo.txt and run ls."},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "toolu_01", "name": "Read", "input": {"file_path": "foo.txt"}},
            {"type": "tool_use", "id": "toolu_02", "name": "Bash", "input": {"command": "ls -la"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_01", "content": "file body here"},
            {"type": "tool_result", "tool_use_id": "toolu_02", "content": "total 0"},
        ]},
    ])
}

fn response_text(record: &Value) -> String {
    record["responses"]
        .as_array()
        .expect("raw record responses must be an array")
        .iter()
        .flat_map(|response| {
            response["packets"]
                .as_array()
                .expect("raw response packets must be an array")
        })
        .filter(|packet| packet["name"] == "data")
        .filter_map(|packet| packet["value"].as_str())
        .filter_map(|chunk| serde_json::from_str::<Value>(chunk).ok())
        .filter_map(|chunk| {
            chunk
                .pointer("/choices/0/delta/content")
                .and_then(Value::as_str)
                .or_else(|| chunk.pointer("/delta/text").and_then(Value::as_str))
                .map(str::to_owned)
        })
        .collect()
}

fn input_token_count(tokenizer: &TiktokenTokenizer, messages: &Value) -> u64 {
    messages
        .as_array()
        .expect("raw request messages must be an array")
        .iter()
        .map(|message| serde_json::to_string(message).expect("serialize canonical message"))
        .map(|message| {
            tokenizer
                .count(&message)
                .expect("tokenize canonical message")
        })
        .map(|count| u64::try_from(count).expect("input token count fits u64"))
        .sum()
}

fn assert_successful_import(
    result: &RunResult,
    expected_trace_id: &str,
    source: &str,
    expected_responses: &[&str],
) {
    assert!(
        result.success(),
        "imported-session run failed (exit {}):\nstdout:\n{}\nstderr:\n{}",
        result.exit_code,
        result.stdout,
        result.stderr
    );
    let raw_records = result.artifacts.raw_records();
    assert_eq!(raw_records.len(), 2, "raw records={raw_records:#?}");
    let expected_execution_trace_id = format!("{expected_trace_id}::instance-0");
    assert!(
        raw_records
            .iter()
            .all(|record| record["status"] == 200 && record["error"].is_null()),
        "raw records={raw_records:#?}"
    );
    for (index, record) in raw_records.iter().enumerate() {
        let expected_node_correlation = format!("{expected_execution_trace_id}:llm_{index}");
        assert_eq!(
            record["metadata"]["x_correlation_id"], expected_node_correlation,
            "raw record {index} lost its graph node identity: {record:#?}"
        );
        assert_eq!(
            record["metadata"]["conversation_id"], expected_execution_trace_id,
            "raw record {index} lost its root execution identity: {record:#?}"
        );
        assert_eq!(
            record["request_headers"]["X-Correlation-ID"], expected_execution_trace_id,
            "raw record {index} sent a non-root scheduling identity: {record:#?}"
        );
    }
    assert_raw_records_timing_and_data(
        &raw_records,
        &TunedExpectations::new(TTFT_MS, ITL_MS, OUTPUT_TOKENS as usize).tol_ms(40.0, 2.0),
    );
    assert!(
        raw_records.iter().all(|record| {
            extract_timing(record)
                .ttft_ms
                .is_some_and(|ttft| ttft >= TTFT_MS / 2.0)
        }),
        "a raw record lost the configured first-token delay: {raw_records:#?}"
    );

    let records = result.artifacts.jsonl();
    assert_eq!(records.len(), 2, "record artifacts={records:#?}");
    let tokenizer = TiktokenTokenizer::builtin();
    for (index, record) in records.iter().enumerate() {
        let expected_isl =
            input_token_count(&tokenizer, &raw_records[index]["payload"]["messages"]);
        assert!(
            record["metrics"]["input_sequence_length"]["value"]
                .as_f64()
                .is_some_and(|value| value == expected_isl as f64),
            "record {index} ISL does not match its exact canonical request history: {record}"
        );
        assert_eq!(
            record["metrics"]["output_sequence_length"]["value"].as_f64(),
            Some(OUTPUT_TOKENS as f64),
            "record {index}: {record}"
        );
        assert_eq!(
            response_text(&raw_records[index]),
            expected_responses[index],
            "record {index} has unexpected deterministic response content: {}",
            raw_records[index]
        );
    }

    let provenance = result.artifacts.read_json_file("**/replay-provenance.json");
    let provenance_trace_id = format!("{source}:{expected_trace_id}");
    assert!(
        provenance["request_profile_digests"][&provenance_trace_id]
            .as_str()
            .is_some_and(|digest| digest.len() == 64),
        "provenance={provenance}"
    );
    let trace_summary = result
        .artifacts
        .read_json_file("**/profile_export_graph_trace_summary.json");
    assert_eq!(trace_summary["trace_count"], 1, "summary={trace_summary}");
    assert_eq!(
        trace_summary["aggregate"]["model_calls"], 2,
        "summary={trace_summary}"
    );
    assert_eq!(
        trace_summary["traces"][0]["trace_id"], expected_execution_trace_id,
        "summary={trace_summary}"
    );
    let tool_time = result
        .artifacts
        .read_json_file("**/profile_export_graph_tool_time.json");
    assert!(
        tool_time.is_null()
            || (tool_time["command_count"] == 0
                && tool_time["durations_s"]
                    .as_array()
                    .is_none_or(Vec::is_empty)),
        "imported sessions must not execute tools: {tool_time}"
    );
}

fn assert_refused_before_traffic(result: &RunResult, required_error: &str) {
    assert!(
        !result.success(),
        "invalid imported session unexpectedly ran"
    );
    let output = format!("{}\n{}", result.stdout, result.stderr);
    assert!(
        output.contains(required_error),
        "expected {required_error:?} in subprocess failure:\n{output}"
    );
    assert!(
        result.artifacts.raw_records().is_empty(),
        "preflight refusal must occur before benchmark traffic"
    );
}

#[tokio::test]
async fn codex_import_replays_canonical_chat_histories_without_tools() {
    let harness = AIPerfHarness::new_with(imported_session_mock()).await;
    let result = run_import(
        &harness,
        &fixture("codex/linear.jsonl"),
        "codex",
        "chat",
        "",
        None,
    );
    assert_successful_import(
        &result,
        "019d28a5-b4a1-7b33-ba0b-c1a7637337d9",
        "codex",
        &["Write fizzbuzz.", "Write fizzbuzz.\nNow "],
    );

    let raw_records = result.artifacts.raw_records();
    assert_eq!(
        raw_records[0]["payload"]["messages"],
        expected_codex_call_0()
    );
    assert_eq!(
        raw_records[1]["payload"]["messages"],
        expected_codex_call_1()
    );
    assert_eq!(
        raw_records[0]["payload"]["max_completion_tokens"],
        FALLBACK_MAX_TOKENS
    );
    assert!(
        raw_records.iter().all(|record| {
            record["payload"]["model"] == "fixture-model"
                && record["payload"]["stream"] == true
                && record["payload"].get("tools").is_none()
        }),
        "raw records={raw_records:#?}"
    );
}

#[tokio::test]
async fn claude_import_replays_canonical_messages_histories_without_tools() {
    let harness = AIPerfHarness::new_with(imported_session_mock()).await;
    let result = run_import(
        &harness,
        &fixture("claude_code/parallel_tools.jsonl"),
        "claude_code",
        "messages",
        "",
        None,
    );
    assert_successful_import(
        &result,
        "sess-cc-tools",
        "claude_code",
        &["Read foo.txt and", "Read foo.txt and"],
    );

    let raw_records = result.artifacts.raw_records();
    assert_eq!(
        raw_records[0]["payload"]["messages"],
        expected_claude_call_0()
    );
    assert_eq!(
        raw_records[1]["payload"]["messages"],
        expected_claude_call_1()
    );
    assert_eq!(
        raw_records[1]["payload"]["messages"][1]["content"][0]["type"],
        "tool_use"
    );
    assert_eq!(
        raw_records[1]["payload"]["messages"][2]["content"][0]["type"],
        "tool_result"
    );
    assert_eq!(raw_records[0]["payload"]["max_tokens"], FALLBACK_MAX_TOKENS);
    assert!(
        raw_records.iter().all(|record| {
            record["payload"]["model"] == "fixture-model"
                && record["payload"]["stream"] == true
                && record["payload"].get("tools").is_none()
        }),
        "raw records={raw_records:#?}"
    );
}

#[tokio::test]
async fn claude_import_rejects_chat_endpoint_before_benchmark_traffic() {
    let harness = AIPerfHarness::new_with(imported_session_mock()).await;

    let claude_chat = run_import(
        &harness,
        &fixture("claude_code/parallel_tools.jsonl"),
        "claude_code",
        "chat",
        "",
        None,
    );
    assert_refused_before_traffic(
        &claude_chat,
        "Claude Code imported session replay requires endpoint type messages",
    );
}

#[tokio::test]
async fn codex_import_rejects_tool_execution_before_benchmark_traffic() {
    let harness = AIPerfHarness::new_with(imported_session_mock()).await;

    let codex_tools = run_import(
        &harness,
        &fixture("codex/with_tools.jsonl"),
        "codex",
        "chat",
        "\x20     execute_tools: true\n",
        None,
    );
    assert_refused_before_traffic(
        &codex_tools,
        "imported sessions reject executable tools, recorded sampling, and standard scenario",
    );
}

#[tokio::test]
async fn imported_session_rejects_recorded_agent_default_before_benchmark_traffic() {
    let harness = AIPerfHarness::new_with(imported_session_mock()).await;

    let standard_scenario = run_import(
        &harness,
        &fixture("codex/linear.jsonl"),
        "codex",
        "chat",
        "",
        Some("recorded-agent-default"),
    );
    assert_refused_before_traffic(&standard_scenario, "recorded-agent-default");
}

#[tokio::test]
async fn imported_session_parse_errors_do_not_echo_private_source_content() {
    let harness = AIPerfHarness::new_with(imported_session_mock()).await;
    let source = tempfile::tempdir().expect("temporary source-qualified privacy fixture");
    let fixture_body =
        fs::read_to_string(fixture("adversarial/claude_code/six_sentinel_error.jsonl"))
            .expect("read privacy fixture");
    // `parentUuid` selects the Claude importer without changing parser semantics.
    let source_body =
        fixture_body.replacen("\"uuid\":\"u\"", "\"parentUuid\":null,\"uuid\":\"u\"", 1);
    let source_path = source.path().join("six-sentinel-error.jsonl");
    fs::write(&source_path, source_body).expect("write source-qualified privacy fixture");
    let result = run_import(&harness, &source_path, "claude_code", "messages", "", None);
    assert_refused_before_traffic(&result, "conflicting tool-use identifier reuse");
    let output = format!("{}\n{}", result.stdout, result.stderr);
    for private in [
        "PRIVATE_PROMPT",
        "PRIVATE_REASONING",
        "PRIVATE_CWD",
        "PRIVATE_BRANCH",
        "PRIVATE_ARGUMENT",
        "PRIVATE_RESULT",
    ] {
        assert!(
            !output.contains(private),
            "subprocess leaked {private}: {output}"
        );
    }
}
