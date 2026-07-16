// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process-level proofs for strict protocol-v2 bootstrap and validation.

use std::io::Write;
use std::process::{Command, Output, Stdio};

use serde_json::{Value, json};

fn runner_capabilities() -> Value {
    // Capabilities is an in-process call now — one binary, no subprocess.
    serde_json::to_value(
        aiperf_cli::execute_mode::capabilities_catalog().expect("capabilities catalog"),
    )
    .expect("catalog to Value")
}

fn request(operation: &str) -> Value {
    json!({
        "protocol_version": 2,
        "operation": operation,
        "run": {
            "benchmark_id": "v2-process-proof",
            "artifact_dir": "/tmp/aiperf-v2-never-created",
            "cfg": {
                "models": {"items": [{"name": "mock-model"}]},
                "endpoint": {
                    "type": "chat",
                    "urls": ["http://127.0.0.1:9"],
                    "streaming": true
                },
                "datasets": [{"type": "synthetic", "entries": 1}],
                "phases": [{
                    "name": "profiling",
                    "type": "concurrency",
                    "exclude_from_results": false,
                    "concurrency": 1
                }],
                "transport": {"type": "http"},
                "runtime": {"workers": 1}
            }
        }
    })
}

fn graph_grpc_request(operation: &str) -> Value {
    let mut request = request(operation);
    request["run"]["cfg"]["transport"]["type"] = json!("grpc");
    request["run"]["cfg"]["datasets"] = json!([{
        "type": "file",
        "format": "dag_jsonl",
        "sampling": "sequential",
        "records": []
    }]);
    request
}

fn run(request: &Value) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_aiperf"))
        .arg("--execute")
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
fn capabilities_emit_plugins_shaped_catalog() {
    let capabilities = runner_capabilities();

    assert_eq!(capabilities["schema_version"], "1.0");
    assert!(
        capabilities["transport"].get("http").is_some(),
        "{capabilities}"
    );
    assert!(capabilities["endpoint"].get("chat").is_some());
    assert!(capabilities.get("supported_pairs").is_none());
    assert!(capabilities.get("distribution_id").is_none());
}

#[test]
fn graph_dataset_selects_graph_path_before_execution() {
    // `grpc + graph` is now admitted at selection (any workload runs over any
    // transport — no pair object, no compatibility predicate). The graph
    // workload resolves the gRPC transport and reaches run-level validation,
    // which fails here because the authored `chat` endpoint over `http://` has no
    // gRPC binding — proving the graph path was selected and validated before any
    // execution, not rejected at selection.
    let output = run(&graph_grpc_request("validate"));
    let response = one_json_line(&output.stdout);

    assert_eq!(output.status.code(), Some(1));
    assert_eq!(response["event"], "run_validation");
    assert_eq!(response["protocol_version"], 2);
    assert_eq!(response["benchmark_id"], "v2-process-proof");
    assert_eq!(response["success"], false);
    assert_eq!(response["completeness"], "static");
    assert_eq!(
        response["errors"][0]["code"],
        "invalid_transport_workload_run"
    );
    assert!(
        response["errors"][0]["message"]
            .as_str()
            .unwrap()
            .contains("gRPC binding")
    );
}

#[test]
fn authored_workload_field_is_ignored_not_rejected() {
    // `workload` was a removed selector. The runner ignores unknown Config keys
    // by design (Python dumps the whole BenchmarkConfig), so a stray `workload`
    // is decoded as if absent rather than rejected by a dedicated guard.
    let mut authored = request("validate");
    authored["run"]["cfg"]["tokenizer"] = json!({"name": "builtin"});
    authored["run"]["cfg"]["workload"] = json!({"type": "scheduled"});
    let output = run(&authored);
    let response = one_json_line(&output.stdout);

    assert_eq!(output.status.code(), Some(0));
    assert_eq!(response["event"], "run_validation");
    assert_eq!(response["benchmark_id"], "v2-process-proof");
    assert_eq!(response["success"], true);
}

#[test]
fn unknown_v2_fields_fail_as_protocol_errors_without_side_effects() {
    let mut authored = request("validate");
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
    let output = run(&graph_grpc_request("execute"));
    let response = one_json_line(&output.stdout);

    assert_eq!(output.status.code(), Some(1));
    assert_eq!(response["event"], "run_terminal");
    assert_eq!(response["protocol_version"], 2);
    assert_eq!(response["stage"], "validation");
    assert_eq!(response["success"], false);
}
