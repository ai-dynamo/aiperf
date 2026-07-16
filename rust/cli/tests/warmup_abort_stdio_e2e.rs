// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process-level gating: a terminal (non-cancelled) failure during the WARMUP
//! graph phase aborts the run BEFORE profiling with a structured
//! `trajectory_warmup_failed` protocol-v2 failure envelope.
//!
//! Mirrors the Python AgentX warmup gate: `report_warmup_failures`
//! (`src/aiperf/timing/strategies/graph_ir_replay.py:908-931`) raised from
//! `PhaseRunner._run_strategy` (`src/aiperf/timing/phase/runner.py:578`) so
//! `TrajectoryWarmupFailedError` propagates and PROFILING never starts.

use std::io::Write;
use std::process::{Command, Output, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use axum::{
    Router, body::Bytes, extract::State, http::StatusCode, response::IntoResponse, routing::post,
};
use serde_json::{Value, json};

/// Shared mock state: whether the endpoint returns a terminal error, and how
/// many node dispatches it observed (used to prove profiling never fires when
/// warmup aborts).
#[derive(Default)]
struct MockState {
    fail: bool,
    requests: AtomicU64,
}

async fn chat(State(state): State<Arc<MockState>>, _body: Bytes) -> impl IntoResponse {
    state.requests.fetch_add(1, Ordering::SeqCst);
    if state.fail {
        return (StatusCode::INTERNAL_SERVER_ERROR, "backend down").into_response();
    }
    (
        [(axum::http::header::CONTENT_TYPE, "text/event-stream")],
        "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n\
         data: {\"choices\":[],\"usage\":{\"prompt_tokens\":16,\"completion_tokens\":1}}\n\n\
         data: [DONE]\n\n",
    )
        .into_response()
}

fn benchmark_run(legacy: Value) -> Value {
    let mut endpoint = legacy["resources"]["endpoints"]["profiles"][0].clone();
    endpoint.as_object_mut().unwrap().remove("id");
    let cfg = json!({
        "models": legacy["resources"]["models"],
        "endpoint": endpoint,
        "datasets": [legacy["workload"]["config"]["dataset"]],
        "phases": legacy["workload"]["config"]["phases"],
        "tokenizer": legacy["workload"]["config"]["tokenizer"],
        "transport": {"type": legacy["transport"]["type"]},
        "runtime": {"workers": legacy["workload"]["config"]["worker_count"]}
    });
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

/// A `t*` window pinned at the midpoint so WARMUP primes the pre-`t*` boundary
/// turn and PROFILING replays the post-`t*` frontier — both phases dispatch a
/// real node.
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

/// One WEKA session with three time-spaced requests so the lowered chain
/// straddles the midpoint `t*`: at least one node lands in WARMUP and one in
/// PROFILING.
///
/// Every turn's lowered node (`warmup_trace:<turn>`) carries the same
/// `metadata["conversation_id"]` (the weka scope, stamped by the recorded trie
/// lowerer at `graph/recorded/trie/mod.rs:170`), which
/// `aiperf_runtime::graph::snapshot::warmup_boundary_nodes` groups per-session chains
/// by; a chain live across `t*` then yields a non-empty warmup boundary that
/// actually dispatches.
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
            "id": "warmup_trace",
            "models": ["recorded-model"],
            "block_size": 16,
            "hash_id_scope": "global",
            "requests": [request(0.0), request(2.0), request(4.0)]
        }]
    })
}

fn request(endpoint: &str, artifact_target: &std::path::Path, benchmark_id: &str) -> Value {
    json!({
        "protocol_version": 2,
        "operation": "execute",
        "run": {
            "identity": {"benchmark_id": benchmark_id, "random_seed": 20_260_707_u64},
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
                        "type": "concurrency",
                        "name": "warmup",
                        "exclude_from_results": true,
                        "sessions": 1,
                        "concurrency": 1
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

async fn spawn_mock(fail: bool) -> (String, Arc<MockState>) {
    let state = Arc::new(MockState {
        fail,
        requests: AtomicU64::new(0),
    });
    let app = Router::new()
        .route("/v1/chat/completions", post(chat))
        .with_state(state.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (format!("http://{address}"), state)
}

/// Terminal warmup failure aborts BEFORE profiling with the structured
/// `trajectory_warmup_failed` envelope naming the failed trace.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn terminal_warmup_failure_aborts_before_profiling() {
    let (endpoint, state) = spawn_mock(true).await;
    let temporary = tempfile::tempdir().unwrap();
    let request = request(
        &endpoint,
        &temporary.path().join("warmup-abort"),
        "warmup-abort",
    );
    let output = tokio::task::spawn_blocking(move || run_child(request))
        .await
        .unwrap();

    let terminal: Value = serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "non-JSON terminal line: {error}\nstdout={}\nstderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
    });
    assert_eq!(terminal["event"], "run_terminal", "{terminal}");
    assert_eq!(terminal["success"], false, "{terminal}");
    assert_eq!(terminal["stage"], "execution", "{terminal}");
    assert_eq!(
        terminal["errors"][0]["code"], "trajectory_warmup_failed",
        "{terminal}"
    );
    let message = terminal["errors"][0]["message"].as_str().unwrap();
    assert!(message.contains("warmup_trace"), "{message}");
    assert!(terminal["report_path"].is_null(), "{terminal}");
    // PROFILING must never start: only the single WARMUP boundary node was
    // dispatched to the backend before the run aborted.
    assert_eq!(
        state.requests.load(Ordering::SeqCst),
        1,
        "profiling must not dispatch after a terminal warmup failure"
    );
}

/// A clean warmup does not trip the gate: profiling runs and the run succeeds.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn clean_warmup_lets_profiling_proceed() {
    let (endpoint, state) = spawn_mock(false).await;
    let temporary = tempfile::tempdir().unwrap();
    let request = request(
        &endpoint,
        &temporary.path().join("warmup-clean"),
        "warmup-clean",
    );
    let output = tokio::task::spawn_blocking(move || run_child(request))
        .await
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
    // Warmup AND profiling both dispatched at least one node.
    assert!(
        state.requests.load(Ordering::SeqCst) >= 2,
        "expected warmup + profiling dispatches, saw {}",
        state.requests.load(Ordering::SeqCst)
    );
}
