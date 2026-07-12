// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process-level reachability proofs for the feature-bearing offline runner.

#![cfg(feature = "dynamo-offline")]

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

use serde_json::{Value, json};

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_aiperf-runner")
}

fn capabilities() -> Value {
    let output = Command::new(binary())
        .arg("--capabilities")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    one_line(&output)
}

fn request(operation: &str, distribution_id: &str, artifact_target: &Path) -> Value {
    json!({
        "protocol_version": 2,
        "operation": operation,
        "expected_distribution_id": distribution_id,
        "run": {
            "identity": {
                "benchmark_id": "offline-v2-process",
                "random_seed": 17
            },
            "artifact_target": artifact_target,
            "models": {"items": [{"name": "mock-model"}]},
            "endpoints": {"profiles": [{
                "id": "default",
                "type": "chat",
                "urls": ["http://127.0.0.1:9"]
            }]},
            "backend": {
                "type": "dynamo_offline",
                "config": {
                    "artifacts": {
                        "report_json": "dynamo/report.json",
                        "per_request_jsonl": "dynamo/requests.jsonl"
                    }
                }
            },
            "workload": {
                "type": "graph",
                "config": {
                    "worker_count": 1,
                    "dataset": {
                        "name": "default",
                        "type": "file",
                        "format": "dag_jsonl",
                        "sampling": "sequential",
                        "records": [
                            {
                                "session_id": "root",
                                "turns": [
                                    {
                                        "messages": [{"role": "user", "content": "root"}],
                                        "forks": [{"child": "child", "background": true}],
                                        "max_tokens": 2
                                    },
                                    {
                                        "messages": [{"role": "user", "content": "joined"}],
                                        "max_tokens": 2
                                    }
                                ]
                            },
                            {
                                "session_id": "child",
                                "turns": [{
                                    "messages": [{"role": "user", "content": "child"}],
                                    "max_tokens": 2
                                }]
                            }
                        ]
                    },
                    "tokenizer": {
                        "name": "builtin",
                        "revision": "main",
                        "trust_remote_code": false,
                        "apply_chat_template": false
                    },
                    "phases": [{
                        "name": "profiling",
                        "type": "concurrency",
                        "exclude_from_results": false,
                        "requests": 3,
                        "concurrency": 2,
                        "seamless": false
                    }]
                }
            },
            "metrics": {},
            "artifacts": {},
            "sidecars": {}
        }
    })
}

fn run(request: &Value) -> Output {
    let mut child = Command::new(binary())
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

fn one_line(output: &Output) -> Value {
    let lines = output
        .stdout
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>();
    assert_eq!(
        lines.len(),
        1,
        "stdout={}",
        String::from_utf8_lossy(&output.stdout)
    );
    serde_json::from_slice(lines[0]).unwrap()
}

fn target(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "aiperf-runner-offline-stdio-{}-{name}",
        std::process::id()
    ))
}

#[test]
fn capabilities_advertise_both_executable_offline_pairs() {
    let capabilities = capabilities();
    assert!(
        capabilities["backends"]
            .as_array()
            .unwrap()
            .iter()
            .any(|backend| backend["id"] == "dynamo_offline")
    );
    assert!(
        capabilities["supported_pairs"]
            .as_array()
            .unwrap()
            .contains(&json!(["dynamo_offline", "graph"]))
    );
    assert!(
        capabilities["supported_pairs"]
            .as_array()
            .unwrap()
            .contains(&json!(["dynamo_offline", "scheduled"]))
    );
}

#[test]
fn validate_is_side_effect_free_and_execute_commits_native_and_dynamo_reports() {
    let capabilities = capabilities();
    let distribution_id = capabilities["distribution_id"].as_str().unwrap();
    let target = target("execute");
    let _ = std::fs::remove_dir_all(&target);

    let validation = run(&request("validate", distribution_id, &target));
    let validation_response = one_line(&validation);
    assert!(validation.status.success(), "{validation_response}");
    assert_eq!(validation_response["event"], "run_validation");
    assert_eq!(validation_response["success"], true);
    assert!(!target.exists(), "validate created the artifact target");

    let execution = run(&request("execute", distribution_id, &target));
    let terminal = one_line(&execution);
    assert!(
        execution.status.success(),
        "terminal={terminal}, stderr={}",
        String::from_utf8_lossy(&execution.stderr)
    );
    assert_eq!(terminal["event"], "run_terminal");
    assert_eq!(terminal["success"], true);
    assert_eq!(terminal["provenance"]["backend"], "dynamo_offline");
    assert_eq!(terminal["provenance"]["workload"], "graph");
    assert_eq!(terminal["provenance"]["parity_shared_fields"], "74");

    let report_path = PathBuf::from(terminal["report_path"].as_str().unwrap());
    assert_eq!(report_path, target.join("native-v2.json"));
    assert!(report_path.is_file());
    let native: Value = serde_json::from_slice(&std::fs::read(&report_path).unwrap()).unwrap();
    assert_eq!(native["schema_version"], "2.0");
    assert_eq!(native["run"]["mode"], "offline:graph");
    assert_eq!(native["run"]["graph"]["input_format"], "dag_jsonl");
    assert_eq!(native["run"]["graph"]["root_count"], 1);
    assert_eq!(native["run"]["graph"]["node_count"], 3);
    assert_eq!(native["run"]["graph"]["worker_count"], 1);
    assert_eq!(native["run"]["graph"]["phase_count"], 1);
    assert_eq!(native["run"]["graph"]["outcome"]["admitted"], 1);
    assert_eq!(native["run"]["graph"]["outcome"]["completed"], 1);
    assert_eq!(native["run"]["graph"]["outcome"]["failed"], 0);
    assert_eq!(native["run"]["dynamo"]["clock"], "sim");
    assert_eq!(native["run"]["dynamo"]["topology"], "single");
    assert_eq!(native["run"]["dynamo"]["router"], "round_robin");
    assert_eq!(native["run"]["dynamo"]["workers"], 1);
    assert_eq!(native["run"]["dynamo"]["prefill_workers"], 1);
    assert_eq!(native["run"]["dynamo"]["decode_workers"], 1);
    assert_eq!(native["run"]["dynamo"]["parity"]["shared_fields"], 74);
    assert_eq!(
        native["run"]["dynamo"]["parity"]["independently_accumulated_fields"],
        69
    );
    assert_eq!(native["run"]["dynamo"]["parity"]["backend_owned_fields"], 5);
    assert!(native["run"]["dynamo"]["capacity"].is_object());

    let dynamo: Value =
        serde_json::from_slice(&std::fs::read(target.join("dynamo/report.json")).unwrap()).unwrap();
    assert_eq!(dynamo["completed_requests"], 3);
    assert_eq!(
        std::fs::read_to_string(target.join("dynamo/requests.jsonl"))
            .unwrap()
            .lines()
            .count(),
        3
    );
    std::fs::remove_dir_all(target).unwrap();
}

#[test]
fn uncompiled_optional_feature_fails_before_artifacts_or_engine_execution() {
    if cfg!(feature = "dynamo-kvbm-offload") {
        return;
    }
    let capabilities = capabilities();
    let target = target("missing-feature");
    let _ = std::fs::remove_dir_all(&target);
    let mut authored = request(
        "execute",
        capabilities["distribution_id"].as_str().unwrap(),
        &target,
    );
    authored["run"]["backend"]["config"]["engine"] = json!({"num_g2_blocks": 8});
    let output = run(&authored);
    let terminal = one_line(&output);

    assert_eq!(output.status.code(), Some(1));
    assert_eq!(terminal["success"], false);
    assert_eq!(terminal["stage"], "validation");
    assert!(
        terminal["errors"][0]["message"]
            .as_str()
            .unwrap()
            .contains("dynamo-kvbm-offload")
    );
    assert!(!target.exists());
}
