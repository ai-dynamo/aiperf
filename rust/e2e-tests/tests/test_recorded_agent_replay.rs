// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Product-level local replay proof for the minimal recorded-agent PinchBench path.

mod common;
use common::*;

use std::fs;
use std::path::{Path, PathBuf};

use regex::Regex;
use serde_json::{Value, json};

fn fixture_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/recorded_agent_replay/pinch_task_pack")
}

fn swe_fixture_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/recorded_agent_replay/swe_recording")
}

fn write_native_config(dir: &Path, url: &str) -> PathBuf {
    let fixture = fixture_root();
    let yaml = format!(
        "schemaVersion: \"2.0\"\n\
         benchmark:\n\
        \x20 model: openai/qwen3.6:35b-a3b\n\
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

fn write_swe_local_config(dir: &Path, url: &str) -> PathBuf {
    let fixture = swe_fixture_root();
    let yaml = format!(
        "schemaVersion: \"2.0\"\n\
         benchmark:\n\
        \x20 model: openai/qwen3-coder:30b\n\
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
    let path = dir.join("recorded-agent-swe-local.yaml");
    fs::write(&path, yaml).expect("write local SWE recorded-agent config");
    path
}

fn read_fixture_expected_requests() -> Vec<Value> {
    serde_json::from_slice(
        &fs::read(fixture_root().join("expected_requests.json"))
            .expect("expected request fixture is readable"),
    )
    .expect("expected request fixture is JSON")
}

fn read_swe_expected_requests() -> Vec<Value> {
    let recording: Value = serde_json::from_slice(
        &fs::read(swe_fixture_root().join("recording.json"))
            .expect("SWE recording fixture is readable"),
    )
    .expect("SWE recording fixture is JSON");
    recording["events"]
        .as_array()
        .expect("SWE recording has events")
        .iter()
        .filter(|event| event["type"] == "model_call")
        .map(|event| {
            event["provider_request"]
                .as_object()
                .unwrap_or_else(|| panic!("model-call event has provider_request: {event}"));
            let mut request = event["provider_request"].clone();
            let completion_tokens = event
                .pointer("/response_message/extra/response/usage/completion_tokens")
                .and_then(Value::as_u64)
                .unwrap_or_else(|| panic!("model-call event has completion usage: {event}"));
            assert!(completion_tokens > 0, "recorded completion cap is positive");
            request["max_tokens"] = Value::from(completion_tokens);
            request
        })
        .collect()
}

fn normalize_expected_request(request: &Value) -> Value {
    let mut request = request.clone();
    let object = request
        .as_object_mut()
        .expect("recorded provider request is a JSON object");
    object.remove("api_base");
    object.remove("api_key");
    object.remove("timeout");
    object.remove("max_retries");
    object.remove("drop_params");
    if let Some(trimmed) = object
        .get("model")
        .and_then(Value::as_str)
        .and_then(|model| model.strip_prefix("openai/"))
    {
        object.insert("model".to_string(), Value::String(trimmed.to_string()));
    }
    object.entry("stream").or_insert_with(|| Value::Bool(true));
    object
        .entry("stream_options")
        .or_insert_with(|| json!({"include_usage": true}));
    normalize_cache_namespace_prefix(&mut request);
    request
}

fn normalize_native_payload(request: &Value) -> Value {
    let mut request = request.clone();
    let object = request
        .as_object_mut()
        .expect("native raw payload is a JSON object");
    if let Some(trimmed) = object
        .get("model")
        .and_then(Value::as_str)
        .and_then(|model| model.strip_prefix("openai/"))
    {
        object.insert("model".to_string(), Value::String(trimmed.to_string()));
    }
    if let Some(max_completion_tokens) = object.remove("max_completion_tokens") {
        object.entry("max_tokens").or_insert(max_completion_tokens);
    }
    object.remove("drop_params");
    object.entry("stream").or_insert_with(|| Value::Bool(true));
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

fn raw_payloads(result: &RunResult) -> Vec<Value> {
    result
        .artifacts
        .raw_records()
        .into_iter()
        .map(|record| {
            record
                .get("payload")
                .cloned()
                .unwrap_or_else(|| panic!("raw record missing payload: {record}"))
        })
        .collect()
}

fn fixture_tool_commands(path: &Path) -> Vec<String> {
    let recording: Value = serde_json::from_slice(&fs::read(path).expect("recording is readable"))
        .expect("recording is JSON");
    recording["events"]
        .as_array()
        .expect("recording events")
        .iter()
        .filter(|event| event["type"] == "tool_call")
        .map(|event| {
            event["action"]["command"]
                .as_str()
                .unwrap_or_else(|| panic!("tool event has command: {event}"))
                .to_owned()
        })
        .collect()
}

fn assert_pinch_fixture_lifecycle() {
    let commands = fixture_tool_commands(&fixture_root().join("recording.json"));
    assert_eq!(commands.len(), 5, "commands={commands:#?}");
    assert_eq!(commands[0], "cat /workspace/deployment.yml");
    assert!(
        commands[1].starts_with("cat > /workspace/deployment.yml << 'EOF'\n"),
        "commands={commands:#?}"
    );
    assert!(
        commands[1].contains("memory: \"256Mi\"")
            && commands[1].contains("name: web-api-secrets")
            && commands[1].contains("targetPort: 8080"),
        "commands={commands:#?}"
    );
    assert_eq!(commands[2], "cat /workspace/deployment.yml");
    assert!(
        commands[3].starts_with("python3 -c \"import yaml;"),
        "commands={commands:#?}"
    );
    assert_eq!(commands[4], "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT");
}

fn assert_swe_fixture_testbed_lifecycle() {
    let commands = fixture_tool_commands(&swe_fixture_root().join("recording.json"));
    assert_eq!(commands.len(), 28, "commands={commands:#?}");
    assert!(
        commands.iter().any(|command| {
            command
                == "cd /testbed && sed -i 's/args += \\[dbname\\]/args.extend(\\[dbname\\])/' django/db/backends/postgresql/client.py"
        }),
        "commands={commands:#?}"
    );
    assert!(
        commands.iter().any(|command| {
            command == "cd /testbed && python3 -m pytest tests/dbshell/test_postgresql.py::PostgreSqlDbshellCommandTestCase::test_parameters -v"
        }),
        "commands={commands:#?}"
    );
    assert!(
        commands
            .iter()
            .any(|command| command == "cd /testbed && git diff > patch.txt"),
        "commands={commands:#?}"
    );
    assert_eq!(
        commands.last().map(String::as_str),
        Some("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt")
    );
}

fn assert_successful_replay_artifacts(
    result: &RunResult,
    expected_calls: u64,
    expected_backend: &str,
) {
    let metrics = result
        .artifacts
        .read_json_file("**/profile_export_aiperf.json");
    assert_eq!(
        metrics["request_count"]["sum"], expected_calls as f64,
        "metrics={metrics}"
    );
    assert_eq!(
        metrics["completed_request_count"]["avg"], expected_calls as f64,
        "metrics={metrics}"
    );
    assert_eq!(
        metrics["request_error_rate"]["avg"], 0.0,
        "metrics={metrics}"
    );
    assert_eq!(metrics["was_cancelled"], false, "metrics={metrics}");
    assert_eq!(metrics["error_summary"], json!([]), "metrics={metrics}");

    let tool_time = result
        .artifacts
        .read_json_file("**/profile_export_graph_tool_time.json");
    assert_eq!(
        tool_time["command_count"], expected_calls,
        "tool_time={tool_time}"
    );
    assert_eq!(tool_time["trace_count"], 1, "tool_time={tool_time}");
    assert_eq!(
        tool_time["backend"], expected_backend,
        "tool_time={tool_time}"
    );
    assert_eq!(
        tool_time["durations_s"].as_array().map(Vec::len),
        Some(expected_calls as usize),
        "tool_time={tool_time}"
    );
    assert!(
        tool_time["durations_s"]
            .as_array()
            .expect("tool time durations are an array")
            .iter()
            .all(|duration| duration.as_f64().is_some_and(f64::is_finite)),
        "tool_time={tool_time}"
    );

    let trace_summary = result
        .artifacts
        .read_json_file("**/profile_export_graph_trace_summary.json");
    assert_eq!(trace_summary["trace_count"], 1, "summary={trace_summary}");
    assert_eq!(
        trace_summary["aggregate"]["model_calls"], expected_calls,
        "summary={trace_summary}"
    );
    assert_eq!(
        trace_summary["aggregate"]["tool_calls"], expected_calls,
        "summary={trace_summary}"
    );
    assert_eq!(
        trace_summary["traces"].as_array().map(Vec::len),
        Some(1),
        "summary={trace_summary}"
    );

    let provenance = result.artifacts.read_json_file("**/replay-provenance.json");
    assert_eq!(provenance["comparable"], true, "provenance={provenance}");
    assert_eq!(
        provenance["cache_isolation_mode"], "first_message_prefix",
        "provenance={provenance}"
    );
    for field in [
        "manifest_digest",
        "cache_namespace_digest",
        "recording_digests",
        "request_profile_digests",
        "environment_digests",
    ] {
        assert!(
            !provenance[field].is_null(),
            "provenance missing {field}: {provenance}"
        );
    }
    assert_eq!(
        provenance["debug_overrides"],
        json!([]),
        "provenance={provenance}"
    );

    let checkpoint = result.artifacts.read_json_file("**/replay-checkpoint.json");
    assert_eq!(checkpoint["version"], 2, "checkpoint={checkpoint}");
    assert_eq!(
        checkpoint["completed"].as_array().map(Vec::len),
        Some(1),
        "checkpoint={checkpoint}"
    );
    let completed = &checkpoint["completed"][0]["completed"];
    assert_eq!(
        completed["successful_call_count"], expected_calls,
        "checkpoint={checkpoint}"
    );
    assert_eq!(
        completed["classification"], "successful",
        "checkpoint={checkpoint}"
    );
    assert_eq!(
        completed["artifact_offset_start"], 0,
        "checkpoint={checkpoint}"
    );
    assert_eq!(
        completed["artifact_offset_end"], expected_calls,
        "checkpoint={checkpoint}"
    );

    let backend = result.artifacts.read_json_file("**/backend-metadata.json");
    assert_eq!(backend["backend"], expected_backend, "backend={backend}");
    assert_eq!(
        backend["backends"],
        json!([expected_backend]),
        "backend={backend}"
    );
    assert_eq!(backend["trace_count"], 1, "backend={backend}");
    assert_eq!(
        backend["command_count"], expected_calls,
        "backend={backend}"
    );

    let failures = result
        .artifacts
        .find_file("**/failures.tsv")
        .expect("failures.tsv exists");
    assert_eq!(
        fs::read_to_string(failures).expect("failures.tsv is readable"),
        "adapter\ttask_id\tclassification\n"
    );
}

/// Config for the `recorded-agent-default` scenario pointed at a bundle that is
/// *not* the canonical corpus.
///
/// Every scenario lock is satisfied so `resolve` admits the run and it reaches
/// the runner; only the canonical-bundle identity check can reject it.
fn write_non_canonical_scenario_config(dir: &Path, url: &str) -> PathBuf {
    let fixture = fixture_root();
    let yaml = format!(
        "schemaVersion: \"2.0\"\n\
         benchmark:\n\
        \x20 model: openai/qwen3.6:35b-a3b\n\
        \x20 scenario: recorded-agent-default\n\
        \x20 endpoint:\n\
        \x20   url: {url}\n\
        \x20   type: chat\n\
        \x20   streaming: true\n\
        \x20   use_server_token_count: true\n\
        \x20 dataset:\n\
        \x20   type: file\n\
        \x20   path: {recording}\n\
        \x20   format: agent_recording\n\
        \x20   sampling: sequential\n\
        \x20   graph:\n\
        \x20     replay_root: {root}\n\
        \x20     execute_tools: true\n\
        \x20     emit_warmup: true\n\
        \x20     pinch_image: aiperf-recorded-agent-pinchbench:v1\n\
        \x20 metadata:\n\
        \x20   hardware: unknown\n\
        \x20   endpoint_placement: remote\n\
        \x20 profiling:\n\
        \x20   concurrency: 1\n\
        \x20   sessions: 1\n\
         runtime:\n\
        \x20 workers: 1\n\
        \x20 cells: 1\n\
        \x20 ui: none\n",
        recording = fixture.join("recording.json").display(),
        root = fixture.display(),
    );
    let path = dir.join("recorded-agent-non-canonical.yaml");
    fs::write(&path, yaml).expect("write non-canonical scenario config");
    path
}

/// `scenario: recorded-agent-default` must reject a bundle whose programs are
/// not the canonical manifest's tasks and recording digests.
///
/// This guards a check with no other coverage. `validate_canonical_recorded_agent_bundle`
/// (`online_execution.rs:1284`) runs only behind `workload.recorded_agent_default`,
/// which the runner derives from `cfg.scenario` — so if that flag is ever dropped
/// while threading the workload config, validation silently stops running and the
/// run *succeeds* instead of failing to decode. The resolve-side scenario locks do
/// not cover it: `RecordedAgentScenarioInputs::canonical` seeds `manifest_digest`,
/// `recording_digests`, and `task_order` from the fixture itself and `resolve`
/// never overrides them, so those three lock comparisons are tautological, and
/// `CanonicalReplayFixture::validate_replay_root` has no production caller.
#[tokio::test]
async fn recorded_agent_default_scenario_rejects_non_canonical_bundle() {
    let harness = AIPerfHarness::new().await;
    let temp = tempfile::tempdir().expect("temporary config directory");
    let config = write_non_canonical_scenario_config(temp.path(), &harness.mock.url);
    let result = harness.run(&format!("--config {}", config.display()));

    assert!(
        !result.success(),
        "a non-canonical bundle was accepted under scenario=recorded-agent-default \
         (exit {}); the canonical-bundle check did not run\nstdout:\n{}\nstderr:\n{}",
        result.exit_code,
        result.stdout,
        result.stderr
    );
    let output = format!("{}{}", result.stdout, result.stderr);
    assert!(
        output.contains("canonical manifest task order and recording digests")
            || output.contains("recorded-agent-default requires recorded replay metadata"),
        "run failed for some other reason than the canonical-bundle check\nstdout:\n{}\nstderr:\n{}",
        result.stdout,
        result.stderr
    );
}

#[tokio::test]
async fn recorded_agent_default_replays_exact_warmup_and_profile_wires() {
    assert_pinch_fixture_lifecycle();
    let harness = AIPerfHarness::new().await;
    let temp = tempfile::tempdir().expect("temporary config directory");
    let config = write_native_config(temp.path(), &harness.mock.url);
    let result = harness.run(&format!(
        "--config {} --export-level raw --ui none",
        config.display()
    ));
    assert!(
        result.success(),
        "recorded-agent replay run failed (exit {}):\nstdout:\n{}\nstderr:\n{}",
        result.exit_code,
        result.stdout,
        result.stderr
    );

    let expected = read_fixture_expected_requests()
        .into_iter()
        .map(|value| normalize_expected_request(&value))
        .collect::<Vec<_>>();
    let actual = raw_payloads(&result)
        .into_iter()
        .map(|value| normalize_native_payload(&value))
        .collect::<Vec<_>>();
    assert_eq!(
        actual, expected,
        "native replay wire body diverged from the recorded source requests"
    );

    let tool_time = result
        .artifacts
        .read_json_file("**/profile_export_graph_tool_time.json");
    assert_eq!(tool_time["command_count"], 5, "tool_time={tool_time}");
    assert_eq!(tool_time["trace_count"], 1, "tool_time={tool_time}");
    assert_eq!(tool_time["backend"], "local", "tool_time={tool_time}");

    let trace_summary = result
        .artifacts
        .read_json_file("**/profile_export_graph_trace_summary.json");
    assert_ne!(
        trace_summary,
        Value::Null,
        "profile_export_graph_trace_summary.json missing"
    );
    let metrics = result.artifacts.read_json_file("**/metrics.json");
    assert_ne!(metrics, Value::Null, "metrics.json missing");
    let provenance = result.artifacts.read_json_file("**/replay-provenance.json");
    assert_ne!(provenance, Value::Null, "replay-provenance.json missing");
    let backend = result.artifacts.read_json_file("**/backend-metadata.json");
    assert_ne!(backend, Value::Null, "backend-metadata.json missing");
    assert_eq!(backend["backend"], "local", "backend={backend}");
    assert_eq!(backend["backends"], json!(["local"]), "backend={backend}");
    assert_eq!(backend["trace_count"], 1, "backend={backend}");
    assert_eq!(backend["command_count"], 5, "backend={backend}");
    assert_successful_replay_artifacts(&result, 5, "local");
}

#[tokio::test]
async fn recorded_agent_swe_local_preserves_testbed_and_exact_replay() {
    assert_swe_fixture_testbed_lifecycle();
    let harness = AIPerfHarness::new().await;
    let temp = tempfile::tempdir().expect("temporary SWE config directory");
    let config = write_swe_local_config(temp.path(), &harness.mock.url);
    let result = harness.run(&format!(
        "--config {} --export-level raw --ui none",
        config.display()
    ));
    assert!(
        result.success(),
        "local SWE recorded-agent replay failed (exit {}):\nstdout:\n{}\nstderr:\n{}",
        result.exit_code,
        result.stdout,
        result.stderr
    );

    let expected = read_swe_expected_requests()
        .iter()
        .map(normalize_expected_request)
        .collect::<Vec<_>>();
    assert_eq!(expected.len(), 28);
    let records = result.artifacts.raw_records();
    assert_eq!(records.len(), 28, "raw records={records:#?}");
    assert!(
        records.iter().all(|record| record["status"] == 200),
        "every SWE model request must complete successfully: {records:#?}"
    );
    let actual = records
        .iter()
        .map(|record| normalize_native_payload(&record["payload"]))
        .collect::<Vec<_>>();
    if let Some((index, (actual, expected))) = actual
        .iter()
        .zip(&expected)
        .enumerate()
        .find(|(_, (actual, expected))| actual != expected)
    {
        let actual = actual.as_object().expect("native request is an object");
        let expected = expected.as_object().expect("recorded request is an object");
        let differing_fields = actual
            .keys()
            .chain(expected.keys())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .filter(|key| actual.get(*key) != expected.get(*key))
            .map(|key| {
                format!(
                    "{key}: native={} recorded={}",
                    actual.get(key).unwrap_or(&Value::Null),
                    expected.get(key).unwrap_or(&Value::Null)
                )
            })
            .collect::<Vec<_>>();
        panic!(
            "native SWE replay request {index} diverged from the existing recording:\n{}",
            differing_fields.join("\n")
        );
    }

    let tool_time = result
        .artifacts
        .read_json_file("**/profile_export_graph_tool_time.json");
    assert_eq!(tool_time["command_count"], 28, "tool_time={tool_time}");
    assert_eq!(tool_time["trace_count"], 1, "tool_time={tool_time}");
    assert_eq!(tool_time["backend"], "local", "tool_time={tool_time}");

    let trace_summary = result
        .artifacts
        .read_json_file("**/profile_export_graph_trace_summary.json");
    assert_eq!(trace_summary["trace_count"], 1, "summary={trace_summary}");
    assert_eq!(
        trace_summary["aggregate"]["model_calls"], 28,
        "summary={trace_summary}"
    );
    assert_eq!(
        trace_summary["aggregate"]["tool_calls"], 28,
        "summary={trace_summary}"
    );

    let backend = result.artifacts.read_json_file("**/backend-metadata.json");
    assert_eq!(backend["backend"], "local", "backend={backend}");
    assert_eq!(backend["backends"], json!(["local"]), "backend={backend}");
    assert_eq!(backend["trace_count"], 1, "backend={backend}");
    assert_eq!(backend["command_count"], 28, "backend={backend}");

    let failures_path = result
        .artifacts
        .find_file("**/failures.tsv")
        .expect("failures.tsv exists");
    let failures = fs::read_to_string(failures_path).expect("failures.tsv is readable");
    assert_eq!(failures, "adapter\ttask_id\tclassification\n");
    assert_successful_replay_artifacts(&result, 28, "local");
}
