// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process-level cache-pressure warmup and profiling handoff coverage.

use std::io::Write;
use std::process::{Command, Output, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use axum::{Router, body::Bytes, extract::State, response::IntoResponse, routing::post};
use serde_json::{Value, json};

#[derive(Default)]
struct MockState {
    requests: AtomicU64,
}

async fn chat(State(state): State<Arc<MockState>>, _body: Bytes) -> impl IntoResponse {
    state.requests.fetch_add(1, Ordering::SeqCst);
    (
        [(axum::http::header::CONTENT_TYPE, "text/event-stream")],
        "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n\
         data: {\"choices\":[],\"usage\":{\"prompt_tokens\":16,\"completion_tokens\":1}}\n\n\
         data: [DONE]\n\n",
    )
        .into_response()
}

fn benchmark_run(source: Value) -> Value {
    let mut endpoint = source["resources"]["endpoints"]["profiles"][0].clone();
    endpoint.as_object_mut().unwrap().remove("id");
    let cfg = json!({
        "models": source["resources"]["models"],
        "endpoint": endpoint,
        "datasets": [source["workload"]["config"]["dataset"]],
        "phases": source["workload"]["config"]["phases"],
        "tokenizer": source["workload"]["config"]["tokenizer"],
        "transport": {"type": source["transport"]["type"]},
        "runtime": {"workers": source["workload"]["config"]["worker_count"]}
    });
    json!({
        "benchmark_id": source["identity"]["benchmark_id"],
        "artifact_dir": source["artifact_target"],
        "random_seed": source["identity"]["random_seed"],
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
        "corpus": "sonnet",
        "trajectory_start_min_ratio": 0.5,
        "trajectory_start_max_ratio": 0.5,
        "t_star_random_seed": 0
    })
}

fn weka_dataset() -> Value {
    let request = |t: f64| {
        json!({
            "t": t,
            "type": "s",
            "model": "recorded-model",
            "in": 16,
            "out": 1,
            "hash_ids": [123456789],
            "api_time": 0.2,
            "ttft": 0.1
        })
    };
    json!({
        "type": "file",
        "format": "weka_trace",
        "sampling": "sequential",
        "synthesis": synthesis(),
        "records": [{
            "id": "pressure_trace",
            "models": ["recorded-model"],
            "block_size": 16,
            "hash_id_scope": "global",
            "requests": [request(0.0), request(0.02), request(0.04)]
        }]
    })
}

fn request(
    endpoint: &str,
    artifact_target: &std::path::Path,
    benchmark_id: &str,
    warmup_duration_s: f64,
) -> Value {
    json!({
        "protocol_version": 2,
        "operation": "execute",
        "run": {
            "identity": {"benchmark_id": benchmark_id, "random_seed": 20_260_715_u64},
            "artifact_target": artifact_target,
            "resources": {
                "models": {"strategy": "round_robin", "items": [{"name": "recorded-model"}]},
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
                "dataset": weka_dataset(),
                "tokenizer": {
                    "name": "cl100k_base",
                    "revision": "main",
                    "trust_remote_code": false,
                    "apply_chat_template": false
                },
                "phases": [
                    {
                        // The pressure duration is the deadline; another stop
                        // condition could end recycling after the first trace.
                        "type": "concurrency",
                        "name": "warmup",
                        "exclude_from_results": true,
                        "concurrency": 1,
                        "agentic_cache_warmup_duration": warmup_duration_s
                    },
                    {
                        "type": "concurrency",
                        "name": "profiling",
                        "exclude_from_results": false,
                        "sessions": 1,
                        "concurrency": 1
                    }
                ]
            }}
        }
    })
}

async fn spawn_mock() -> (String, Arc<MockState>) {
    let state = Arc::new(MockState::default());
    let app = Router::new()
        .route("/v1/chat/completions", post(chat))
        .with_state(state.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (format!("http://{address}"), state)
}

fn request_count_total(metrics: &Value) -> f64 {
    metrics["request_count"]["series"][0]["stats"]["total"]
        .as_f64()
        .unwrap_or_else(|| panic!("request_count.total missing: {metrics}"))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cache_pressure_warmup_recycles_then_profiling_resumes() {
    let (endpoint, state) = spawn_mock().await;
    let temporary = tempfile::tempdir().unwrap();
    let artifact_target = temporary.path().join("cache-pressure-warmup");
    let request = request(&endpoint, &artifact_target, "cache-pressure-warmup", 0.3);

    // Wall-clock guard: the run is duration-bounded and the mock always answers,
    // so this only ever trips on a genuine hang (never on the happy path).
    let output: Output = tokio::time::timeout(
        std::time::Duration::from_secs(30),
        tokio::task::spawn_blocking(move || run_child(request)),
    )
    .await
    .expect("cache-pressure warmup run hung past the timeout budget")
    .unwrap();

    let terminal: Value = serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "non-JSON terminal line: {error}\nstdout={}\nstderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
    });
    assert_eq!(terminal["event"], "run_terminal", "{terminal}");
    assert_eq!(terminal["success"], true, "{terminal}");

    let report_path = terminal["report_path"].as_str().unwrap();
    let report: Value = serde_json::from_slice(&std::fs::read(report_path).unwrap()).unwrap();
    assert_eq!(report["schema_version"], "2.0", "{report}");

    // This single-template corpus dispatches one node per pass.
    let warmup = &report["warmup_metrics"];
    assert!(warmup.is_object(), "warmup_metrics missing: {report}");
    let warmup_requests = request_count_total(warmup);
    assert!(
        warmup_requests > 1.0,
        "expected warmup cache-pressure recycle (>1 corpus pass), saw request_count={warmup_requests}"
    );

    let profiling_requests = request_count_total(&report["metrics"]);
    assert!(
        profiling_requests >= 1.0,
        "expected profiling records for the resumed frontier, saw request_count={profiling_requests}"
    );

    let total_dispatches = state.requests.load(Ordering::SeqCst);
    assert!(
        total_dispatches >= 3,
        "expected recycle + profiling dispatches at the backend, saw {total_dispatches}"
    );
}
