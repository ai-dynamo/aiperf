// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reference-vs-native local replay request parity for recorded-agent playback.

mod common;
use common::*;

use std::fs;
use std::path::{Path, PathBuf};

use regex::Regex;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const PINNED_OPEN_LAB_REVISION: &str = "b8897f5de1664ad6de9cd669a96c3ba5d379e81e";
const REFERENCE_CAPTURE_SCHEMA_VERSION: u64 = 1;

fn fixture_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/recorded_agent_replay/pinch_task_pack")
}

fn reference_capture() -> Value {
    let path = fixture_root().join("reference_capture.json");
    serde_json::from_slice(&fs::read(&path).expect("read committed Open LAB request capture"))
        .expect("committed Open LAB request capture is JSON")
}

#[test]
fn committed_reference_capture_is_pinned_and_complete() {
    let capture = reference_capture();
    assert_eq!(
        capture["schema_version"], REFERENCE_CAPTURE_SCHEMA_VERSION,
        "capture={capture}"
    );
    assert_eq!(
        capture["reference"]["repository"], "open-lab-benchmark",
        "capture={capture}"
    );
    assert_eq!(
        capture["reference"]["revision"], PINNED_OPEN_LAB_REVISION,
        "capture={capture}"
    );
    assert_eq!(
        capture["capture"]["transport"], "raw-openai-http",
        "capture={capture}"
    );
    assert_eq!(
        capture["capture"]["warmup_requests"], 1,
        "capture={capture}"
    );
    assert_eq!(
        capture["capture"]["profile_requests"], 5,
        "capture={capture}"
    );
    assert_eq!(
        capture["capture"]["profile_request_fixture"], "expected_requests.json",
        "capture={capture}"
    );
    let profile_payloads = read_reference_profile_payloads();
    assert_eq!(
        profile_payloads.len(),
        capture["capture"]["profile_requests"]
            .as_u64()
            .unwrap_or_default() as usize,
        "capture={capture}"
    );
    let digest = format!(
        "{:x}",
        Sha256::digest(
            fs::read(fixture_root().join("expected_requests.json"))
                .expect("read committed Open LAB profile request capture"),
        )
    );
    assert_eq!(
        capture["capture"]["profile_request_sha256"], digest,
        "capture={capture}"
    );
}

fn read_reference_profile_payloads() -> Vec<Value> {
    serde_json::from_slice(
        &fs::read(fixture_root().join("expected_requests.json"))
            .expect("read committed Open LAB profile request capture"),
    )
    .expect("committed Open LAB profile request capture is JSON")
}

#[test]
fn native_normalization_preserves_litellm_only_drop_params() {
    let request = normalize_native_request(json!({"drop_params": true}));
    assert_eq!(request["drop_params"], true);
}

#[test]
fn normalization_strips_only_a_32_digit_cache_namespace() {
    let short = "1 2 3\nPerformance replay cache namespace. Ignore the digits above.\n\nbody";
    let short_request = normalize_native_request(json!({
        "messages": [{"role": "system", "content": short}]
    }));
    assert_eq!(short_request["messages"][0]["content"], short);

    let namespace = (0..32)
        .map(|digit| (digit % 10).to_string())
        .collect::<Vec<_>>()
        .join(" ");
    let full = format!(
        "{namespace}\nPerformance replay cache namespace. Ignore the digits above.\n\nbody"
    );
    let full_request = normalize_native_request(json!({
        "messages": [{"role": "system", "content": full}]
    }));
    assert_eq!(full_request["messages"][0]["content"], "body");
}

#[test]
#[should_panic(expected = "both max_tokens and max_completion_tokens")]
fn normalization_rejects_both_completion_cap_spellings() {
    normalize_native_request(json!({"max_tokens": 4, "max_completion_tokens": 8}));
}

fn write_native_config(dir: &Path, url: &str) -> PathBuf {
    let fixture = fixture_root();
    let yaml = format!(
        "schemaVersion: \"2.0\"\n\
         benchmark:\n\
        \x20 model: qwen3.6:35b-a3b\n\
        \x20 endpoint:\n\
        \x20   url: {url}\n\
        \x20   type: chat\n\
        \x20   streaming: true\n\
        \x20   use_server_token_count: true\n\
        \x20 dataset:\n\
        \x20   type: file\n\
        \x20   path: {recording}\n\
        \x20   format: agent_recording\n\
        \x20   graph:\n\
        \x20     replay_root: {root}\n\
        \x20     execute_tools: true\n\
        \x20     emit_warmup: true\n\
        \x20 metadata:\n\
        \x20   hardware: unknown\n\
        \x20   endpoint_placement: remote\n\
        \x20 profiling:\n\
        \x20   concurrency: 1\n\
        \x20   sessions: 1\n\
         runtime:\n\
        \x20 workers: 1\n\
        \x20 ui: none\n",
        recording = fixture.join("recording.json").display(),
        root = fixture.display(),
    );
    let path = dir.join("recorded-agent-replay.yaml");
    fs::write(&path, yaml).expect("write recorded-agent config");
    path
}

fn normalize_native_request(request: Value) -> Value {
    normalize_request(request, false)
}

fn normalize_reference_request(request: Value) -> Value {
    normalize_request(request, true)
}

fn normalize_request(mut request: Value, is_litellm_capture: bool) -> Value {
    let object = request
        .as_object_mut()
        .expect("request body is a JSON object");
    if is_litellm_capture {
        object.remove("drop_params");
    }
    if let Some(max_completion_tokens) = object.remove("max_completion_tokens") {
        assert!(
            !object.contains_key("max_tokens"),
            "request contains both max_tokens and max_completion_tokens"
        );
        object.insert("max_tokens".to_string(), max_completion_tokens);
    }
    object
        .entry("stream_options")
        .or_insert_with(|| json!({"include_usage": true}));
    normalize_cache_namespace_prefix(&mut request);
    request
}

fn normalize_cache_namespace_prefix(request: &mut Value) {
    static CACHE_NAMESPACE_PREFIX: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    let regex = CACHE_NAMESPACE_PREFIX.get_or_init(|| {
        Regex::new(
            r"^\d(?: \d){31}\nPerformance replay cache namespace\. Ignore the digits above\.\n\n",
        )
        .expect("cache namespace regex compiles")
    });
    let Some(messages) = request.get_mut("messages").and_then(Value::as_array_mut) else {
        return;
    };
    for message in messages.iter_mut() {
        let Some(object) = message.as_object_mut() else {
            continue;
        };
        if object.get("role").and_then(Value::as_str) != Some("system") {
            continue;
        }
        let Some(Value::String(content)) = object.get_mut("content") else {
            continue;
        };
        let normalized = regex.replace(content, "");
        *content = normalized.into_owned();
        return;
    }
}

fn native_payloads(result: &RunResult) -> Vec<Value> {
    let records = result.artifacts.raw_records();
    assert_eq!(records.len(), 5, "native profiling record count");
    assert!(
        records.iter().all(|record| record["status"] == 200),
        "native profiling requests must all complete with HTTP 200: {records:#?}"
    );
    records
        .into_iter()
        .map(|record| {
            normalize_native_request(
                record
                    .get("payload")
                    .cloned()
                    .unwrap_or_else(|| panic!("raw record missing payload: {record}")),
            )
        })
        .collect()
}

fn reference_payloads() -> Vec<Value> {
    let capture = reference_capture();
    assert_eq!(capture["schema_version"], REFERENCE_CAPTURE_SCHEMA_VERSION);
    assert_eq!(capture["reference"]["revision"], PINNED_OPEN_LAB_REVISION);
    assert_eq!(capture["capture"]["transport"], "raw-openai-http");
    assert_eq!(capture["capture"]["warmup_requests"], 1);
    assert_eq!(capture["capture"]["profile_requests"], 5);
    read_reference_profile_payloads()
        .into_iter()
        .map(normalize_reference_request)
        .collect()
}

#[tokio::test]
async fn reference_and_native_only_normalize_documented_wire_differences() {
    let harness = AIPerfHarness::new().await;
    let temp = tempfile::tempdir().expect("temporary parity directory");
    let config = write_native_config(temp.path(), &harness.mock.url);

    let native = harness.run(&format!(
        "--config {} --export-level raw --ui none",
        config.display()
    ));
    assert!(
        native.success(),
        "native recorded-agent replay failed (exit {}):\nstdout:\n{}\nstderr:\n{}",
        native.exit_code,
        native.stdout,
        native.stderr
    );
    let tool_time = native
        .artifacts
        .read_json_file("**/profile_export_graph_tool_time.json");
    assert_eq!(tool_time["command_count"], 5, "tool_time={tool_time}");
    assert_eq!(tool_time["trace_count"], 1, "tool_time={tool_time}");
    assert_eq!(tool_time["backend"], "local", "tool_time={tool_time}");
    let trace_summary = native
        .artifacts
        .read_json_file("**/profile_export_graph_trace_summary.json");
    assert_eq!(trace_summary["trace_count"], 1, "summary={trace_summary}");
    assert_eq!(
        trace_summary["aggregate"]["model_calls"], 5,
        "summary={trace_summary}"
    );
    assert_eq!(
        trace_summary["aggregate"]["tool_calls"], 5,
        "summary={trace_summary}"
    );
    let failures_path = native
        .artifacts
        .find_file("**/failures.tsv")
        .expect("native failures.tsv exists");
    assert_eq!(
        fs::read_to_string(failures_path).expect("read native failures.tsv"),
        "adapter\ttask_id\tclassification\n"
    );
    let native = native_payloads(&native);
    let reference = reference_payloads();

    assert_eq!(native.len(), reference.len());
    if let Some((index, (native, reference))) = native
        .iter()
        .zip(&reference)
        .enumerate()
        .find(|(_, (native, reference))| native != reference)
    {
        let native = native.as_object().expect("native request is an object");
        let reference = reference
            .as_object()
            .expect("reference request is an object");
        let differing_fields = native
            .keys()
            .chain(reference.keys())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .filter(|key| native.get(*key) != reference.get(*key))
            .map(|key| {
                format!(
                    "{key}: native={} reference={}",
                    native.get(key).unwrap_or(&Value::Null),
                    reference.get(key).unwrap_or(&Value::Null)
                )
            })
            .collect::<Vec<_>>();
        panic!(
            "reference and native replay request {index} diverged beyond the documented normalization set:\n{}",
            differing_fields.join("\n")
        );
    }
}
