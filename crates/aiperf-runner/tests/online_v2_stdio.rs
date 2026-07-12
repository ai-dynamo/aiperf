// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process proofs for executable protocol-v2 online pairs.

use std::io::Write;
use std::process::{Command, Output, Stdio};

use axum::{
    Json, Router,
    http::header,
    response::IntoResponse,
    routing::{get, post},
};
use serde_json::{Value, json};

const COMMIT: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const TOKENIZER_JSON: &str = r#"{
  "version":"1.0",
  "truncation":null,
  "padding":null,
  "added_tokens":[
    {"id":0,"content":"[UNK]","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}
  ],
  "normalizer":null,
  "pre_tokenizer":{"type":"Whitespace"},
  "post_processor":null,
  "decoder":null,
  "model":{"type":"WordLevel","vocab":{"[UNK]":0,"root":1,"answer":2},"unk_token":"[UNK]"}
}"#;

fn capabilities() -> Value {
    let output = Command::new(env!("CARGO_BIN_EXE_aiperf-runner"))
        .arg("--capabilities")
        .output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");
    serde_json::from_slice(&output.stdout).unwrap()
}

fn run_child(request: Value, environment: &[(&str, &str)]) -> Output {
    let bytes = serde_json::to_vec(&request).unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_aiperf-runner"))
        .envs(environment.iter().copied())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(&bytes).unwrap();
    child.wait_with_output().unwrap()
}

#[test]
fn graph_adapter_and_profile_references_fail_before_artifact_creation() {
    let capabilities = capabilities();
    let artifacts = tempfile::tempdir().unwrap();
    let artifact_target = artifacts.path().join("must-not-exist");
    let request = json!({
        "protocol_version": 2,
        "operation": "execute",
        "expected_distribution_id": capabilities["distribution_id"],
        "run": {
            "identity": {"benchmark_id": "online-v2-invalid-graph"},
            "artifact_target": artifact_target,
            "resources": {
                "models": {"strategy": "round_robin", "items": [{"name": "fixture-model"}]},
                "endpoints": {"profiles": [{
                    "id": "default",
                    "type": "chat",
                    "urls": ["http://127.0.0.1:1"],
                    "streaming": true,
                    "wait_for_model_timeout": 0.0,
                    "wait_for_model_interval": 5.0,
                    "wait_for_model_mode": "inference"
                }]}
            },
            "backend": {"type": "online_http", "config": {}},
            "workload": {"type": "graph", "config": {
                "worker_count": 1,
                "dataset": {
                    "type": "file",
                    "format": "dag_jsonl",
                    "sampling": "sequential",
                    "records": [{
                        "session_id": "root",
                        "turns": [{
                            "messages": [{"role": "user", "content": "root"}],
                            "endpoint": "missing-profile",
                            "max_tokens": 1
                        }]
                    }]
                },
                "tokenizer": {
                    "name": "cl100k_base",
                    "revision": "main",
                    "trust_remote_code": false,
                    "apply_chat_template": false
                },
                "phases": [{
                    "type": "concurrency",
                    "name": "profiling",
                    "exclude_from_results": false,
                    "sessions": 1,
                    "concurrency": 1
                }]
            }}
        }
    });
    let output = run_child(request, &[]);
    assert!(!output.status.success(), "{output:?}");
    let terminal: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(terminal["stage"], "preparation");
    assert!(
        terminal["errors"][0]["message"]
            .as_str()
            .unwrap()
            .contains("missing-profile")
    );
    assert!(!artifact_target.exists());
}

async fn model_metadata() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "application/json")],
        format!(r#"{{"sha":"{COMMIT}"}}"#),
    )
}

async fn tokenizer_json() -> impl IntoResponse {
    ([(header::CONTENT_TYPE, "application/json")], TOKENIZER_JSON)
}

async fn tokenizer_config() -> impl IntoResponse {
    ([(header::CONTENT_TYPE, "application/json")], "{}")
}

async fn chat() -> impl IntoResponse {
    let body = concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"answer\"}}]}\n\n",
        "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1}}\n\n",
        "data: [DONE]\n\n",
    );
    ([(header::CONTENT_TYPE, "text/event-stream")], body)
}

async fn kserve_v1_predict(Json(payload): Json<Value>) -> impl IntoResponse {
    let text = payload["instances"][0]["text"].as_str().unwrap();
    assert!(!text.is_empty());
    Json(json!({"predictions": [{"output": "answer"}]}))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn scheduled_pair_executes_kserve_v1_endpoint_only_through_v2() {
    let app = Router::new().route("/v1/models/fixture-model:predict", post(kserve_v1_predict));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let capabilities = capabilities();
    assert!(
        capabilities["supported_pairs"]
            .as_array()
            .unwrap()
            .contains(&json!(["online_http", "scheduled"]))
    );
    let artifacts = tempfile::tempdir().unwrap();
    let artifact_target = artifacts.path().join("kserve-v1-run");
    let request = json!({
        "protocol_version": 2,
        "operation": "execute",
        "expected_distribution_id": capabilities["distribution_id"],
        "run": {
            "identity": {"benchmark_id": "online-v2-kserve-v1", "random_seed": 9},
            "artifact_target": artifact_target,
            "resources": {
                "models": {"items": [{"name": "fixture-model"}]},
                "endpoints": {"profiles": [{
                    "id": "default",
                    "type": "kserve_v1_predict",
                    "urls": [format!("http://{address}")],
                    "streaming": false,
                    "wait_for_model_timeout": 0.0
                }]}
            },
            "backend": {"type": "online_http", "config": {}},
            "workload": {"type": "scheduled", "config": {
                "worker_count": 1,
                "dataset": {
                    "type": "synthetic",
                    "entries": 1,
                    "sampling": "sequential",
                    "prompts": {
                        "isl": {"value": 4.0},
                        "osl": {"value": 1.0}
                    }
                },
                "tokenizer": {
                    "name": "cl100k_base",
                    "revision": "main",
                    "trust_remote_code": false,
                    "apply_chat_template": false
                },
                "phases": [{
                    "type": "concurrency",
                    "name": "profiling",
                    "exclude_from_results": false,
                    "requests": 1,
                    "concurrency": 1
                }]
            }}
        }
    });
    let output = tokio::task::spawn_blocking(move || run_child(request, &[]))
        .await
        .unwrap();
    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let terminal: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(terminal["protocol_version"], 2);
    assert_eq!(terminal["success"], true);
    assert_eq!(terminal["provenance"]["backend"], "online_http");
    assert_eq!(terminal["provenance"]["workload"], "scheduled");
    let report: Value =
        serde_json::from_slice(&std::fs::read(artifact_target.join("native-v2.json")).unwrap())
            .unwrap();
    assert_eq!(
        report["run"]["endpoint_profiles"],
        json!([{"profile_id": "default", "endpoint_id": "kserve_v1_predict"}])
    );
    assert_eq!(
        report["metrics"]["request_count"]["series"][0]["stats"]["total"], 1.0,
        "report={report}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn graph_pair_executes_direct_dag_with_remote_tokenizer_over_stdio() {
    let app = Router::new()
        .route(
            "/api/models/fixture/model/revision/locked",
            get(model_metadata),
        )
        .route(
            &format!("/fixture/model/resolve/{COMMIT}/tokenizer.json"),
            get(tokenizer_json),
        )
        .route(
            &format!("/fixture/model/resolve/{COMMIT}/tokenizer_config.json"),
            get(tokenizer_config),
        )
        .route("/v1/chat/completions", post(chat));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let capabilities = capabilities();
    assert!(
        capabilities["supported_pairs"]
            .as_array()
            .unwrap()
            .contains(&json!(["online_http", "graph"]))
    );
    let artifacts = tempfile::tempdir().unwrap();
    let artifact_target = artifacts.path().join("graph-run");
    let cache = artifacts.path().join("cache");
    let endpoint = format!("http://{address}");
    let request = json!({
        "protocol_version": 2,
        "operation": "execute",
        "expected_distribution_id": capabilities["distribution_id"],
        "run": {
            "identity": {"benchmark_id": "online-v2-graph", "random_seed": 7},
            "artifact_target": artifact_target,
            "resources": {
                "models": {"strategy": "round_robin", "items": [{"name": "fixture-model"}]},
                "endpoints": {"profiles": [
                    {
                        "id": "judge",
                        "type": "chat_completions",
                        "urls": [endpoint],
                        "streaming": true,
                        "use_server_token_count": true,
                        "wait_for_model_timeout": 0.0,
                        "wait_for_model_interval": 5.0,
                        "wait_for_model_mode": "inference"
                    },
                    {
                        "id": "default",
                        "type": "chat",
                        "urls": [endpoint],
                        "streaming": true,
                        "use_server_token_count": true,
                        "wait_for_model_timeout": 0.0,
                        "wait_for_model_interval": 5.0,
                        "wait_for_model_mode": "inference"
                    }
                ]}
            },
            "backend": {"type": "online_http", "config": {}},
            "workload": {"type": "graph", "config": {
                "worker_count": 1,
                "dataset": {
                    "type": "file",
                    "format": "dag_jsonl",
                    "sampling": "sequential",
                    "records": [{
                        "session_id": "root",
                        "turns": [{
                            "messages": [{"role": "user", "content": "root"}],
                            "max_tokens": 1
                        }]
                    }]
                },
                "tokenizer": {
                    "name": "fixture/model",
                    "revision": "locked",
                    "trust_remote_code": false,
                    "apply_chat_template": false
                },
                "phases": [{
                    "type": "concurrency",
                    "name": "profiling",
                    "exclude_from_results": false,
                    "sessions": 1,
                    "concurrency": 1
                }]
            }}
        }
    });
    let output = tokio::task::spawn_blocking(move || {
        run_child(
            request,
            &[
                ("HF_ENDPOINT", endpoint.as_str()),
                ("AIPERF_CACHE_DIR", cache.to_str().unwrap()),
            ],
        )
    })
    .await
    .unwrap();
    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let terminal: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(terminal["event"], "run_terminal");
    assert_eq!(terminal["success"], true);
    assert_eq!(terminal["provenance"]["backend"], "online_http");
    assert_eq!(terminal["provenance"]["workload"], "graph");
    let report: Value =
        serde_json::from_slice(&std::fs::read(artifact_target.join("native-v2.json")).unwrap())
            .unwrap();
    assert_eq!(
        report["run"]["distribution_id"],
        capabilities["distribution_id"]
    );
    assert_eq!(report["run"]["backend"], "online_http");
    assert_eq!(report["run"]["workload"], "graph");
    assert_eq!(report["run"]["extensions"], json!([]));
    assert_eq!(
        report["run"]["endpoint_profiles"],
        json!([
            {"profile_id": "judge", "endpoint_id": "chat"},
            {"profile_id": "default", "endpoint_id": "chat"}
        ])
    );
    assert_eq!(report["run"]["mode"], "graph");
    assert_eq!(
        report["metrics"]["request_count"]["series"][0]["stats"]["total"],
        1.0
    );
}
