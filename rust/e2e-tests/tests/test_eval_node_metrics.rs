// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Product-level NativeGraph node-record coverage against the deterministic mock server.

mod common;

use std::{fs, path::Path};

use common::{AIPerfHarness, tuned_mock_config};
use serde_json::{Value, json};

const MODEL: &str = "gpt-4";
const TTFT_MS: f64 = 100.0;
const ITL_MS: f64 = 10.0;
const OUTPUT_TOKENS: u64 = 4;

#[tokio::test]
async fn eval_node_metrics_preserve_two_streamed_model_records_and_reward_json() {
    let harness = AIPerfHarness::new_with(tuned_mock_config(TTFT_MS, ITL_MS)).await;
    let task = harness.artifact_path().join("native-graph-task");
    write_native_graph_task(&task, &harness.mock.url);
    let model_runtime = harness.artifact_path().join("model-runtime.toml");
    fs::write(&model_runtime, "version = 1\n").expect("write model runtime");
    let lifecycle = harness.artifact_path().join("lifecycle.json");
    write_lifecycle(&lifecycle);
    let records_output = harness.artifact_path().join("eval-node-records.jsonl");

    let result = harness.run_no_server(&format!(
        "eval --task {} --model-runtime {} --lifecycle-request {} --records-output {}",
        task.display(),
        model_runtime.display(),
        lifecycle.display(),
        records_output.display(),
    ));
    assert!(
        result.success(),
        "eval failed with {}\nstdout:\n{}\nstderr:\n{}",
        result.exit_code,
        result.stdout,
        result.stderr
    );

    let rows = read_jsonl(&records_output);
    assert_eq!(rows.len(), 2, "one canonical row per model node: {rows:#?}");
    let correlations = rows
        .iter()
        .map(|row| row["metadata"]["x_correlation_id"].as_str().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(
        correlations,
        [
            "eval-node-metrics:first-model",
            "eval-node-metrics:second-model"
        ]
    );

    for (index, row) in rows.iter().enumerate() {
        assert_eq!(metric(row, "input_sequence_length"), 4.0, "row {index}");
        assert_eq!(
            metric(row, "output_token_count"),
            OUTPUT_TOKENS as f64,
            "row {index}: terminal usage or [DONE] must not count as output"
        );
        assert_eq!(
            metric(row, "output_sequence_length"),
            OUTPUT_TOKENS as f64,
            "row {index}"
        );
        assert_timing_near(
            metric(row, "time_to_first_token"),
            TTFT_MS,
            40.0,
            "TTFT",
            index,
        );
        assert_timing_near(
            metric(row, "inter_token_latency"),
            ITL_MS,
            2.0,
            "ITL",
            index,
        );
        assert_eq!(row["error"], Value::Null, "row {index}: {row}");
    }

    let server_metrics = harness
        .mock
        .state
        .recorder
        .labeled("/v1/chat/completions", MODEL);
    assert_eq!(
        server_metrics.requests_total_200.get(),
        2,
        "both node requests must complete with HTTP 200"
    );
    assert_eq!(
        server_metrics.streaming_requests.get(),
        2,
        "both model bindings must reach the mock as streaming requests"
    );
    assert_eq!(
        server_metrics.requests_by_model.get(),
        2,
        "both requests must carry the fixed model identity"
    );

    let reward: Value = serde_json::from_str(result.stdout.trim()).expect("reward output is JSON");
    assert_eq!(
        reward,
        json!({
            "task": "example/eval-node-metrics",
            "artifacts": [],
            "reward": {"reward": 1.0},
            "episodes": 1
        }),
        "the records sidecar must not alter scored stdout"
    );
}

fn metric(row: &Value, name: &str) -> f64 {
    row["metrics"][name]["value"]
        .as_f64()
        .unwrap_or_else(|| panic!("missing {name} metric in {row}"))
}

fn assert_timing_near(actual: f64, expected: f64, tolerance: f64, name: &str, index: usize) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "row {index}: {name} {actual:.3}ms is not within {tolerance}ms of {expected:.3}ms"
    );
}

fn read_jsonl(path: &Path) -> Vec<Value> {
    fs::read_to_string(path)
        .expect("read eval node records")
        .lines()
        .map(|line| serde_json::from_str(line).expect("canonical record row is JSON"))
        .collect()
}

fn write_native_graph_task(task: &Path, endpoint: &str) {
    fs::create_dir_all(task.join("environment")).expect("create environment directory");
    fs::create_dir_all(task.join("tests")).expect("create verifier directory");
    fs::write(
        task.join("task.toml"),
        r#"schema_version = "1.1"

[task]
name = "example/eval-node-metrics"

[native_graph]
profile = "native_graph"
program = "agent_graph.json"
model_bindings = "models.toml"
adapter_manifest = "adapters.toml"
"#,
    )
    .expect("write task manifest");
    fs::write(task.join("instruction.md"), "Complete both model nodes.\n")
        .expect("write instruction");
    fs::write(
        task.join("environment/Dockerfile"),
        "FROM alpine:3.20\nRUN mkdir -p /work /logs/verifier && chmod 0777 /work /logs/verifier\n",
    )
    .expect("write Dockerfile");
    fs::write(
        task.join("tests/test.sh"),
        "mkdir -p /logs/verifier\nprintf '{\"reward\":1.0}' > /logs/verifier/reward.json\n",
    )
    .expect("write verifier");
    fs::write(
        task.join("agent_graph.json"),
        r#"{
  "schema_version": "1.0",
  "trace_id": "eval-node-metrics",
  "stage_bound": 1,
  "channels": {
    "prompt": {"type": "messages", "reducer": "add_messages"},
    "first": {"type": "messages", "reducer": "add_messages"},
    "second": {"type": "messages", "reducer": "add_messages"}
  },
  "nodes": [
    {"id": "first-model", "kind": "model", "binding": "primary", "inputs": ["prompt"], "output": "first", "streaming": true, "max_tokens": 4},
    {"id": "second-model", "kind": "model", "binding": "primary", "inputs": ["prompt"], "output": "second", "streaming": true, "max_tokens": 4}
  ],
  "edges": [
    {"source": "START", "target": "first-model"},
    {"source": "first-model", "target": "second-model"},
    {"source": "second-model", "target": "END"}
  ],
  "terminal_outputs": [],
  "initial_state": {
    "prompt": [{"role": "user", "content": "one two three four"}]
  }
}"#,
    )
    .expect("write graph");
    fs::write(
        task.join("models.toml"),
        format!(
            r#"[[model_bindings]]
id = "primary"
endpoint_profile_id = "provider-default"
endpoint_factory_id = "chat"
transport_factory_id = "http"
model = "{MODEL}"
urls = ["{endpoint}/v1"]
streaming = true
request_timeout_ms = 30000
capture = "metadata"

[model_bindings.tokenizer]
type = "local"
name = "builtin"
revision = "main"
apply_chat_template = false

[model_bindings.generation]
max_tokens = 4
min_tokens = 4
"#
        ),
    )
    .expect("write model bindings");
    fs::write(task.join("adapters.toml"), "").expect("write empty adapter manifest");
}

fn write_lifecycle(path: &Path) {
    let policy = format!("blake3:{}", "a".repeat(64));
    fs::write(
        path,
        serde_json::to_vec(&json!({
            "version": 1,
            "agent_variant": "native-graph",
            "model": {"provider": "provider-default", "model": MODEL},
            "seed": 11,
            "policy": policy,
            "runtime": "native:e2e",
            "attempt": "eval-node-metrics-attempt",
            "budget": {"execution_seconds": 30.0, "verifier_seconds": 30.0},
            "agent_contract": "native_graph",
            "command": ["aiperf-native-graph"],
            "initial_score": {"metric": "reward", "rationale": format!("blake3:{}", "b".repeat(64))},
            "regrade": {"metric": "reward", "rationale": format!("blake3:{}", "c".repeat(64))}
        }))
        .expect("serialize lifecycle"),
    )
    .expect("write lifecycle");
}
