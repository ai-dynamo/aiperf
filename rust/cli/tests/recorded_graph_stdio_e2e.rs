// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process-level byte parity for native WEKA and Dynamo Graph-IR inputs.

use std::io::Write;
use std::process::{Command, Output, Stdio};
use std::sync::{Arc, Mutex};

use axum::{
    Router,
    body::Bytes,
    extract::State,
    http::{HeaderMap, header},
    response::IntoResponse,
    routing::post,
};
use serde_json::{Value, json};

#[derive(Debug)]
struct CapturedRequest {
    body: Bytes,
    session_id: Option<String>,
    parent_session_id: Option<String>,
    session_final: Option<String>,
}

#[derive(Default)]
struct WireCapture {
    requests: Mutex<Vec<CapturedRequest>>,
}

async fn chat(
    State(capture): State<Arc<WireCapture>>,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse {
    let header = |name: &'static str| {
        headers
            .get(name)
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned)
    };
    capture.requests.lock().unwrap().push(CapturedRequest {
        body,
        session_id: header("x-dynamo-session-id"),
        parent_session_id: header("x-dynamo-parent-session-id"),
        session_final: header("x-dynamo-session-final"),
    });
    (
        [(header::CONTENT_TYPE, "text/event-stream")],
        "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n\
         data: {\"choices\":[],\"usage\":{\"prompt_tokens\":16,\"completion_tokens\":1}}\n\n\
         data: [DONE]\n\n",
    )
}

fn capabilities() -> Value {
    // Capabilities is an in-process call now — one binary, no subprocess.
    serde_json::to_value(
        aiperf_cli::execute_mode::capabilities_catalog().expect("capabilities catalog"),
    )
    .expect("catalog to Value")
}

fn benchmark_run(legacy: Value) -> Value {
    let mut endpoint = legacy["resources"]["endpoints"]["profiles"][0].clone();
    endpoint.as_object_mut().unwrap().remove("id");
    let mut cfg = json!({
        "models": legacy["resources"]["models"],
        "endpoint": endpoint,
        "datasets": [legacy["workload"]["config"]["dataset"]],
        "phases": legacy["workload"]["config"]["phases"],
        "tokenizer": legacy["workload"]["config"]["tokenizer"],
        "transport": {"type": legacy["transport"]["type"]},
        "runtime": {"workers": legacy["workload"]["config"]["worker_count"]}
    });
    // Side-channel telemetry producers are authored under `cfg.sidecars`; the
    // graph path accepts and runs them exactly as the scheduled path does.
    let sidecars = legacy["resources"]["sidecars"].clone();
    if !sidecars.is_null() {
        cfg.as_object_mut()
            .unwrap()
            .insert("sidecars".to_owned(), sidecars);
    }
    json!({
        "benchmark_id": legacy["identity"]["benchmark_id"],
        "artifact_dir": legacy["artifact_target"],
        "random_seed": legacy["identity"]["random_seed"],
        "cfg": cfg
    })
}

fn run_child(mut request: Value) -> Output {
    request["run"] = benchmark_run(request["run"].take());
    let bytes = serde_json::to_vec(&request["run"]).unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_aiperf"))
        .arg("--execute")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(&bytes).unwrap();
    child.wait_with_output().unwrap()
}

fn synthesis() -> Value {
    json!({
        "speedup_ratio": 1.0,
        "prefix_len_multiplier": 1.0,
        "prefix_root_multiplier": 1,
        "prompt_len_multiplier": 1.0,
        "output_len_multiplier": 1.0,
        "idle_gap_cap_seconds": 60.0,
        "corpus": "sonnet"
    })
}

fn request(
    endpoint: &str,
    artifact_target: &std::path::Path,
    benchmark_id: &str,
    dataset: Value,
) -> Value {
    json!({
        "protocol_version": 2,
        "operation": "execute",
        "run": {
            "identity": {"benchmark_id": benchmark_id, "random_seed": 2_026_070_7_u64},
            "artifact_target": artifact_target,
            "resources": {
                "models": {
                    "strategy": "round_robin",
                    "items": [{"name": "recorded-model"}]
                },
                "endpoints": {"profiles": [{
                    "id": "default",
                    "type": "chat",
                    "urls": [endpoint],
                    "streaming": true,
                    "use_server_token_count": true,
                    "wait_for_model_timeout": 0.0,
                    "wait_for_model_interval": 5.0,
                    "wait_for_model_mode": "inference"
                }]}
            },
            "transport": {"type": "http", "config": {}},
            "workload": {"type": "graph", "config": {
                "worker_count": 1,
                "dataset": dataset,
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
    })
}

fn assert_success(output: &Output) {
    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let terminal: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(terminal["event"], "run_terminal");
    assert_eq!(terminal["success"], true);
}

/// A Graph-IR run must accept and run the same GPU/network/server telemetry
/// side-channels the scheduled path does. Telemetry is a barrier-synchronized
/// side-channel, not part of the workload, so an enabled-but-unreachable source
/// is tolerated (it logs and is skipped) and the run still completes.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn graph_run_accepts_enabled_but_unreachable_telemetry_sidecars() {
    let app = Router::new()
        .route("/v1/chat/completions", post(chat))
        .with_state(Arc::new(WireCapture::default()));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let endpoint = format!("http://{address}");
    let temporary = tempfile::tempdir().unwrap();
    let dynamo_path = temporary.path().join("trace.jsonl");
    let dynamo_record = json!({
        "schema": "dynamo.request.trace.v1",
        "event_type": "request_end",
        "event_time_unix_ms": 1_000_200,
        "event_source": "dynamo",
        "agent_context": {"session_id": "telemetry"},
        "request": {
            "request_id": "request-0",
            "model": "recorded-model",
            "input_tokens": 16,
            "output_tokens": 1,
            "cached_tokens": 0,
            "request_received_ms": 1_000_000,
            "total_time_ms": 200,
            "ttft_ms": 100,
            "replay": {
                "trace_block_size": 16,
                "input_length": 16,
                "input_sequence_hashes": [123456789]
            }
        }
    });
    std::fs::write(&dynamo_path, serde_json::to_vec(&dynamo_record).unwrap()).unwrap();

    let dataset = json!({
        "type": "file",
        "format": "dynamo_trace",
        "sampling": "sequential",
        "synthesis": synthesis(),
        "path": dynamo_path
    });
    let mut request = request(
        &endpoint,
        &temporary.path().join("telemetry-artifacts"),
        "graph-telemetry",
        dataset,
    );
    // An unreachable DCGM endpoint plus a fixed-mean network calibration and a
    // server-metrics scrape target: none reachable, all authored under the same
    // sidecar block the scheduled path consumes.
    request["run"]["resources"]["sidecars"] = json!({
        "gpu_telemetry": {
            "collection_interval_ns": 100_000_000,
            "request_timeout_ns": 50_000_000,
            "records_path": "gpu.jsonl",
            "sources": [{"type": "dcgm", "url": "http://127.0.0.1:1/metrics"}]
        },
        "network_latency": {"mean_rtt_ns": 250000.0},
        "server_metrics": {
            "collection_interval_ns": 100_000_000,
            "reachability_timeout_ns": 50_000_000,
            "urls": ["http://127.0.0.1:1/metrics"],
            "formats": ["json"]
        }
    });
    let output = tokio::task::spawn_blocking(move || run_child(request))
        .await
        .unwrap();
    assert_success(&output);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn config_v2_weka_and_dynamo_dispatch_byte_identical_http_bodies() {
    let capture = Arc::new(WireCapture::default());
    let app = Router::new()
        .route("/v1/chat/completions", post(chat))
        .with_state(capture.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let capabilities = capabilities();
    assert!(
        capabilities["transport"].get("http").is_some(),
        "{capabilities}"
    );
    let endpoint = format!("http://{address}");
    let temporary = tempfile::tempdir().unwrap();
    let dynamo_path = temporary.path().join("trace.jsonl");
    let dynamo_record = json!({
        "schema": "dynamo.request.trace.v1",
        "event_type": "request_end",
        "event_time_unix_ms": 1_000_200,
        "event_source": "dynamo",
        "agent_context": {"session_id": "parity"},
        "request": {
            "request_id": "request-0",
            "model": "recorded-model",
            "input_tokens": 16,
            "output_tokens": 1,
            "cached_tokens": 0,
            "request_received_ms": 1_000_000,
            "total_time_ms": 200,
            "ttft_ms": 100,
            "replay": {
                "trace_block_size": 16,
                "input_length": 16,
                "input_sequence_hashes": [123456789]
            }
        }
    });
    std::fs::write(&dynamo_path, serde_json::to_vec(&dynamo_record).unwrap()).unwrap();

    let weka_dataset = json!({
        "type": "file",
        "format": "weka_trace",
        "sampling": "sequential",
        "synthesis": synthesis(),
        "records": [{
            "id": "parity",
            "models": ["recorded-model"],
            "block_size": 16,
            "hash_id_scope": "global",
            "requests": [{
                "t": 0.0,
                "type": "s",
                "model": "recorded-model",
                "in": 16,
                "out": 1,
                "hash_ids": [123456789],
                "api_time": 0.2,
                "ttft": 0.1
            }]
        }]
    });
    let dynamo_dataset = json!({
        "type": "file",
        "format": "dynamo_trace",
        "sampling": "sequential",
        "synthesis": synthesis(),
        "path": dynamo_path
    });
    let weka_artifacts = temporary.path().join("weka-artifacts");
    let dynamo_artifacts = temporary.path().join("dynamo-artifacts");

    let weka = request(
        &endpoint,
        &weka_artifacts,
        "weka-recorded-parity",
        weka_dataset,
    );
    assert_success(
        &tokio::task::spawn_blocking(move || run_child(weka))
            .await
            .unwrap(),
    );
    let dynamo = request(
        &endpoint,
        &dynamo_artifacts,
        "dynamo-recorded-parity",
        dynamo_dataset,
    );
    assert_success(
        &tokio::task::spawn_blocking(move || run_child(dynamo))
            .await
            .unwrap(),
    );

    let captured = capture.requests.lock().unwrap();
    assert_eq!(captured.len(), 2);
    let weka = captured
        .iter()
        .find(|request| request.session_id.is_none())
        .expect("WEKA request without Dynamo identity headers");
    let dynamo = captured
        .iter()
        .find(|request| request.session_id.is_some())
        .expect("Dynamo request with identity headers");
    assert_eq!(weka.body, dynamo.body, "exact HTTP request body bytes");

    let body: Value = serde_json::from_slice(&weka.body).unwrap();
    assert_eq!(body["model"], "recorded-model");
    assert_eq!(body["max_completion_tokens"], 1);
    assert_eq!(body["stream"], true);
    assert_eq!(body["messages"].as_array().unwrap().len(), 1);
    assert_eq!(body["messages"][0]["role"], "user");
    assert!(
        body["messages"][0]["content"]
            .as_str()
            .is_some_and(|content| !content.is_empty())
    );
    assert_eq!(
        dynamo.session_id.as_deref(),
        Some("parity::profiling-instance-0")
    );
    assert_eq!(dynamo.parent_session_id, None);
    assert_eq!(dynamo.session_final.as_deref(), Some("true"));
}
