// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process coverage for executable protocol-v2 online configurations.

use std::io::Write;
use std::process::{Command, Output, Stdio};

use axum::{
    Json, Router,
    http::header,
    response::IntoResponse,
    routing::{get, post},
};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use url::Url;

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
    serde_json::to_value(
        aiperf_cli::execute_mode::capabilities_catalog().expect("capabilities catalog"),
    )
    .expect("catalog to Value")
}

fn run_child(request: Value, environment: &[(&str, &str)]) -> Output {
    let bytes = serde_json::to_vec(&request["run"]).unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_aiperf"))
        .arg("--execute")
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
    let artifacts = tempfile::tempdir().unwrap();
    let artifact_dir = artifacts.path().join("must-not-exist");
    let request = json!({
        "protocol_version": 2,
        "operation": "execute",
        "run": {
            "benchmark_id": "online-v2-invalid-graph",
            "artifact_dir": artifact_dir,
            "cfg": {
                "models": {"strategy": "round_robin", "items": [{"name": "fixture-model"}]},
                "endpoint": {
                    "type": "chat",
                    "urls": ["http://127.0.0.1:1"],
                    "streaming": true,
                    "wait_for_model_timeout": 0.0,
                    "wait_for_model_interval": 5.0,
                    "wait_for_model_mode": "inference"
                },
                "datasets": [{
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
                }],
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
                }],
                "transport": {"type": "http"},
                "runtime": {"workers": 1}
            }
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
    assert!(!artifact_dir.exists());
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

async fn content_server_chat(Json(payload): Json<Value>) -> impl IntoResponse {
    let image_url = payload["messages"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|message| message["content"].as_array().into_iter().flatten())
        .find(|content| content["type"] == "image_url")
        .and_then(|content| content["image_url"]["url"].as_str())
        .expect("chat request contains one image URL");
    assert!(
        image_url.starts_with("http://127.0.0.1:"),
        "synthetic image was not externalized: {image_url}"
    );
    let parsed = Url::parse(image_url).unwrap();
    let address = format!(
        "{}:{}",
        parsed.host_str().unwrap(),
        parsed.port_or_known_default().unwrap()
    );
    let mut stream = tokio::net::TcpStream::connect(address).await.unwrap();
    stream
        .write_all(
            format!(
                "GET {} HTTP/1.1\r\nHost: {}\r\nConnection: close\r\n\r\n",
                parsed.path(),
                parsed.host_str().unwrap()
            )
            .as_bytes(),
        )
        .await
        .unwrap();
    let mut response = Vec::new();
    stream.read_to_end(&mut response).await.unwrap();
    assert!(response.starts_with(b"HTTP/1.1 200 OK"));
    let body = response
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|index| &response[index + 4..])
        .unwrap();
    assert!(body.starts_with(b"\x89PNG\r\n\x1a\n"));
    chat().await
}

async fn kserve_v1_predict(Json(payload): Json<Value>) -> impl IntoResponse {
    let text = payload["instances"][0]["text"].as_str().unwrap();
    assert!(!text.is_empty());
    Json(json!({"predictions": [{"output": "answer"}]}))
}

async fn vllm_generate(Json(payload): Json<Value>) -> impl IntoResponse {
    assert_eq!(payload["model"], "fixture-model");
    assert_eq!(payload["stream"], false);
    assert_eq!(payload["sampling_params"]["max_tokens"], 3);
    let token_ids = payload["token_ids"].as_array().unwrap();
    assert_eq!(token_ids.len(), 4);
    assert!(token_ids.iter().all(Value::is_u64));
    assert!(payload.get("prompt").is_none());
    assert!(payload.get("messages").is_none());
    Json(json!({
        "request_id": "server-request",
        "choices": [{
            "index": 0,
            "token_ids": [90, 91, 92],
            "finish_reason": "stop"
        }]
    }))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn scheduled_pair_executes_kserve_v1_endpoint_only_through_v2() {
    let app = Router::new().route("/v1/models/fixture-model:predict", post(kserve_v1_predict));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    assert!(
        capabilities()["transport"].get("http").is_some(),
        "catalog must expose the HTTP transport"
    );
    let artifacts = tempfile::tempdir().unwrap();
    let artifact_dir = artifacts.path().join("kserve-v1-run");
    let request = json!({
        "protocol_version": 2,
        "operation": "execute",
        "run": {
            "benchmark_id": "online-v2-kserve-v1",
            "artifact_dir": artifact_dir,
            "random_seed": 9,
            "cfg": {
                "models": {"items": [{"name": "fixture-model"}]},
                "endpoint": {
                    "type": "kserve_v1_predict",
                    "urls": [format!("http://{address}")],
                    "streaming": false,
                    "wait_for_model_timeout": 0.0
                },
                "datasets": [{
                    "type": "synthetic",
                    "entries": 1,
                    "sampling": "sequential",
                    "prompts": {
                        "isl": {"value": 4.0},
                        "osl": {"value": 1.0}
                    }
                }],
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
                }],
                "transport": {"type": "http"},
                "runtime": {"workers": 1}
            }
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
    assert_eq!(terminal["run_metadata"]["transport"], "http");
    assert_eq!(terminal["run_metadata"]["workload"], "scheduled");
    let report: Value =
        serde_json::from_slice(&std::fs::read(artifact_dir.join("native-v2.json")).unwrap())
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
async fn scheduled_pair_executes_vllm_token_arrays_without_text_round_trip() {
    let app = Router::new().route("/inference/v1/generate", post(vllm_generate));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let descriptor = &capabilities()["endpoint"]["vllm_generate"]["metadata"];
    assert_eq!(descriptor["requires_raw_token_ids"], true);
    assert_eq!(descriptor["tokenizes_input"], false);
    assert_eq!(descriptor["produces_tokens"], true);

    let artifacts = tempfile::tempdir().unwrap();
    let artifact_dir = artifacts.path().join("vllm-generate-run");
    let request = json!({
        "protocol_version": 2,
        "operation": "execute",
        "run": {
            "benchmark_id": "online-v2-vllm-generate",
            "artifact_dir": artifact_dir,
            "random_seed": 19,
            "cfg": {
                "models": {"items": [{"name": "fixture-model"}]},
                "endpoint": {
                    "type": "vllm_generate",
                    "urls": [format!("http://{address}")],
                    "streaming": false,
                    "wait_for_model_timeout": 0.0
                },
                "datasets": [{
                    "type": "synthetic",
                    "entries": 1,
                    "sampling": "sequential",
                    "prompts": {
                        "isl": {"value": 4.0},
                        "osl": {"value": 3.0}
                    }
                }],
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
                }],
                "transport": {"type": "http"},
                "runtime": {"workers": 1}
            }
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
    assert_eq!(terminal["success"], true);
    let report: Value =
        serde_json::from_slice(&std::fs::read(artifact_dir.join("native-v2.json")).unwrap())
            .unwrap();
    assert_eq!(
        report["run"]["endpoint_profiles"],
        json!([{"profile_id": "default", "endpoint_id": "vllm_generate"}])
    );
    assert_eq!(
        report["metrics"]["total_usage_prompt_tokens"]["series"][0]["stats"]["value"],
        4.0
    );
    assert_eq!(
        report["metrics"]["total_usage_completion_tokens"]["series"][0]["stats"]["value"],
        3.0
    );
    assert_eq!(
        report["metrics"]["total_isl"]["series"][0]["stats"]["value"], 4.0,
        "the authored token array must be the measured input length"
    );
    assert_eq!(
        report["metrics"]["total_output_tokens"]["series"][0]["stats"]["value"], 3.0,
        "non-text response IDs must emit one output-token observation per ID"
    );
    assert!(
        report["metrics"]["time_to_first_token"]
            .get("series")
            .is_some(),
        "a non-text token array must still establish TTFT"
    );
}

#[test]
fn vllm_token_requirement_fails_during_dataset_preparation() {
    let artifacts = tempfile::tempdir().unwrap();
    let artifact_dir = artifacts.path().join("must-not-exist");
    let request = json!({
        "protocol_version": 2,
        "operation": "execute",
        "run": {
            "benchmark_id": "vllm-generate-invalid-dataset",
            "artifact_dir": artifact_dir,
            "cfg": {
                "models": {"items": [{"name": "fixture-model"}]},
                "endpoint": {
                    "type": "vllm_generate",
                    "urls": ["http://127.0.0.1:1"],
                    "streaming": false,
                    "wait_for_model_timeout": 0.0
                },
                "datasets": [{
                    "type": "file",
                    "format": "single_turn",
                    "sampling": "sequential",
                    "records": [{"text": "this must not be tokenized on dispatch"}]
                }],
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
                }],
                "transport": {"type": "http"},
                "runtime": {"workers": 1}
            }
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
            .contains("raw_token_ids")
    );
    assert!(!artifact_dir.exists());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn scheduled_pair_serves_generated_image_urls_for_the_full_run_lifecycle() {
    let app = Router::new().route("/v1/chat/completions", post(content_server_chat));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let port_reservation = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let content_port = port_reservation.local_addr().unwrap().port();
    drop(port_reservation);
    let artifacts = tempfile::tempdir().unwrap();
    let artifact_dir = artifacts.path().join("content-server-run");
    let content_dir = artifacts.path().join("content");
    std::fs::create_dir(&content_dir).unwrap();
    let request = json!({
        "protocol_version": 2,
        "operation": "execute",
        "run": {
            "benchmark_id": "online-v2-content-server",
            "artifact_dir": artifact_dir,
            "random_seed": 9,
            "cfg": {
                "models": {"items": [{"name": "fixture-model"}]},
                "endpoint": {
                    "type": "chat",
                    "urls": [format!("http://{address}")],
                    "streaming": true,
                    "wait_for_model_timeout": 0.0
                },
                "sidecars": {
                    "content_server": {
                        "host": "127.0.0.1",
                        "port": content_port,
                        "content_dir": content_dir,
                        "max_tracked_records": 100
                    }
                },
                "datasets": [{
                    "type": "synthetic",
                    "entries": 1,
                    "sampling": "sequential",
                    "prompts": {
                        "isl": {"value": 4.0},
                        "osl": {"value": 1.0}
                    },
                    "images": {
                        "batch_size": 1,
                        "width": {"value": 4.0},
                        "height": {"value": 3.0},
                        "format": "png",
                        "source": "noise",
                        "source_sampling": "random-with-replacement"
                    }
                }],
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
                }],
                "transport": {"type": "http"},
                "runtime": {"workers": 1}
            }
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
    assert_eq!(terminal["success"], true);
    let image_path = content_dir.join("images/img_000001.png");
    assert!(
        std::fs::read(image_path)
            .unwrap()
            .starts_with(b"\x89PNG\r\n\x1a\n")
    );

    let released = tokio::net::TcpListener::bind(("127.0.0.1", content_port))
        .await
        .expect("content-server listener is released when the run exits");
    drop(released);
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

    assert!(
        capabilities()["transport"].get("http").is_some(),
        "catalog must expose the HTTP transport"
    );
    let artifacts = tempfile::tempdir().unwrap();
    let artifact_dir = artifacts.path().join("graph-run");
    let cache = artifacts.path().join("cache");
    let endpoint = format!("http://{address}");
    let request = json!({
        "protocol_version": 2,
        "operation": "execute",
        "run": {
            "benchmark_id": "online-v2-graph",
            "artifact_dir": artifact_dir,
            "random_seed": 7,
            "cfg": {
                "models": {"strategy": "round_robin", "items": [{"name": "fixture-model"}]},
                "endpoint": {
                    "type": "chat",
                    "urls": [endpoint],
                    "streaming": true,
                    "use_server_token_count": true,
                    "wait_for_model_timeout": 0.0,
                    "wait_for_model_interval": 5.0,
                    "wait_for_model_mode": "inference"
                },
                "endpoint_profiles": {
                    "judge": {
                        "type": "chat_completions",
                        "urls": [endpoint],
                        "streaming": true,
                        "use_server_token_count": true,
                        "wait_for_model_timeout": 0.0,
                        "wait_for_model_interval": 5.0,
                        "wait_for_model_mode": "inference"
                    }
                },
                "datasets": [{
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
                }],
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
                }],
                "transport": {"type": "http"},
                "runtime": {"workers": 1}
            }
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
    assert_eq!(terminal["run_metadata"]["transport"], "http");
    assert_eq!(terminal["run_metadata"]["workload"], "graph");
    let report: Value =
        serde_json::from_slice(&std::fs::read(artifact_dir.join("native-v2.json")).unwrap())
            .unwrap();
    assert_eq!(report["run"]["transport"], "http");
    assert_eq!(report["run"]["workload"], "graph");
    assert_eq!(report["run"]["extensions"], json!([]));
    assert_eq!(
        report["run"]["endpoint_profiles"],
        json!([
            {"profile_id": "default", "endpoint_id": "chat"},
            {"profile_id": "judge", "endpoint_id": "chat"}
        ])
    );
    assert_eq!(report["run"]["mode"], "graph");
    assert_eq!(
        report["metrics"]["request_count"]["series"][0]["stats"]["total"],
        1.0
    );
}
