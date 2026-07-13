// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process-level proofs for strict protocol-v2 bootstrap and validation.

use std::io::Write;
use std::process::{Command, Output, Stdio};

use serde_json::{Value, json};

fn runner_capabilities() -> Value {
    let output = Command::new(env!("CARGO_BIN_EXE_aiperf-runner"))
        .arg("--capabilities")
        .output()
        .unwrap();
    assert!(output.status.success(), "{:?}", output);
    one_json_line(&output.stdout)
}

fn request(operation: &str, distribution_id: &str) -> Value {
    json!({
        "protocol_version": 2,
        "operation": operation,
        "expected_distribution_id": distribution_id,
        "run": {
            "identity": {"benchmark_id": "v2-process-proof"},
            "artifact_target": "/tmp/aiperf-v2-never-created",
            "transport": {"type": "http", "config": {}},
            "workload": {"type": "scheduled", "config": {
                "worker_count": 1,
                "dataset": {"type": "synthetic", "entries": 1},
                "tokenizer": {"name": "builtin"},
                "phases": [{
                    "name": "profiling",
                    "type": "concurrency",
                    "exclude_from_results": false,
                    "concurrency": 1
                }]
            }},
            "resources": {
                "models": {"items": [{"name": "mock-model"}]},
                "endpoints": {"profiles": [{
                    "id": "default",
                    "type": "chat",
                    "urls": ["http://127.0.0.1:9"],
                    "streaming": true,
                    "use_legacy_max_tokens": false,
                    "use_server_token_count": true,
                    "timeout_seconds": 10.0,
                    "connection_reuse": "pooled",
                    "download_video_content": false,
                    "extra": {},
                    "headers": {},
                    "http2": false,
                    "wait_for_model_timeout": 0.0,
                    "wait_for_model_interval": 5.0,
                    "wait_for_model_mode": "inference"
                }]},
                "metrics": {},
                "artifacts": {},
                "sidecars": {}
            }
        }
    })
}

fn unregistered_pair_request(operation: &str, distribution_id: &str) -> Value {
    let mut request = request(operation, distribution_id);
    request["run"]["transport"]["type"] = json!("grpc");
    request["run"]["workload"]["type"] = json!("graph");
    request["run"]["workload"]["config"]["dataset"] = json!({
        "type": "file",
        "format": "dag_jsonl",
        "sampling": "sequential",
        "records": []
    });
    request
}

fn run(request: &Value) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_aiperf-runner"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(serde_json::to_string(request).unwrap().as_bytes())
        .unwrap();
    child.wait_with_output().unwrap()
}

fn one_json_line(stdout: &[u8]) -> Value {
    let lines = stdout
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>();
    assert_eq!(lines.len(), 1, "stdout={}", String::from_utf8_lossy(stdout));
    serde_json::from_slice(lines[0]).unwrap()
}

#[test]
fn capabilities_distinguish_static_compatibility_from_executable_pairs() {
    let capabilities = runner_capabilities();

    assert_eq!(capabilities["capabilities_schema_version"], 2);
    assert_eq!(capabilities["protocol_versions"], json!([2]));
    assert!(
        capabilities["transports"]
            .as_array()
            .unwrap()
            .iter()
            .any(|entry| entry["id"] == "http")
    );
    for workload in ["scheduled", "graph", "static_accuracy", "agentic"] {
        assert!(
            capabilities["workloads"]
                .as_array()
                .unwrap()
                .iter()
                .any(|entry| entry["id"] == workload),
            "missing {workload}"
        );
    }
    assert!(
        capabilities["statically_compatible_pairs"]
            .as_array()
            .unwrap()
            .contains(&json!(["http", "scheduled"]))
    );
    assert!(
        capabilities["supported_pairs"]
            .as_array()
            .unwrap()
            .contains(&json!(["http", "scheduled"])),
        "the protocol-v2 scheduled adapter must be product-reachable"
    );
    assert!(
        capabilities["supported_pairs"]
            .as_array()
            .unwrap()
            .contains(&json!(["http", "agentic"])),
        "the registered agentic pair must be product-reachable"
    );
}

#[test]
fn validate_emits_one_typed_failure_for_a_recognized_but_unregistered_pair() {
    let capabilities = runner_capabilities();
    let output = run(&unregistered_pair_request(
        "validate",
        capabilities["distribution_id"].as_str().unwrap(),
    ));
    let response = one_json_line(&output.stdout);

    assert_eq!(output.status.code(), Some(1));
    assert_eq!(response["event"], "run_validation");
    assert_eq!(response["protocol_version"], 2);
    assert_eq!(response["benchmark_id"], "v2-process-proof");
    assert_eq!(response["success"], false);
    assert_eq!(response["completeness"], "static");
    assert_eq!(
        response["errors"][0]["code"],
        "invalid_transport_workload_selection"
    );
    assert!(
        response["errors"][0]["message"]
            .as_str()
            .unwrap()
            .contains("does not contain transport/workload pair")
    );
}

#[test]
fn distribution_mismatch_precedes_strict_run_validation_and_exits_two() {
    let mut malformed = request("validate", &format!("blake3:{}", "0".repeat(64)));
    malformed["run"] = json!({"identity": {"benchmark_id": "digest-first"}});
    let output = run(&malformed);
    let response = one_json_line(&output.stdout);

    assert_eq!(output.status.code(), Some(2));
    assert_eq!(response["event"], "run_validation");
    assert_eq!(response["benchmark_id"], "digest-first");
    assert_eq!(response["errors"][0]["code"], "distribution_mismatch");
}

#[test]
fn unknown_v2_fields_fail_as_protocol_errors_without_side_effects() {
    let capabilities = runner_capabilities();
    let mut authored = request(
        "validate",
        capabilities["distribution_id"].as_str().unwrap(),
    );
    authored["run"]["unknown_outer_field"] = json!(true);
    let output = run(&authored);
    let response = one_json_line(&output.stdout);

    assert_eq!(output.status.code(), Some(2));
    assert_eq!(response["event"], "run_validation");
    assert_eq!(response["errors"][0]["code"], "invalid_request");
    assert!(
        response["errors"][0]["message"]
            .as_str()
            .unwrap()
            .contains("unknown field `unknown_outer_field`")
    );
}

#[test]
fn execute_uses_typed_terminal_and_validation_exit_one() {
    let capabilities = runner_capabilities();
    let output = run(&unregistered_pair_request(
        "execute",
        capabilities["distribution_id"].as_str().unwrap(),
    ));
    let response = one_json_line(&output.stdout);

    assert_eq!(output.status.code(), Some(1));
    assert_eq!(response["event"], "run_terminal");
    assert_eq!(response["protocol_version"], 2);
    assert_eq!(response["stage"], "validation");
    assert_eq!(response["success"], false);
}
