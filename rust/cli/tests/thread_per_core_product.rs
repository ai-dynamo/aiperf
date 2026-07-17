// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Product coverage for native thread-per-core HTTP placement.
//!
//! Each worker owns its scheduler and persistent transport connection. Phase
//! accounting is merged at the cell join.

use std::io::Write;
use std::net::SocketAddr;
use std::process::{Command, Output, Stdio};
use std::sync::{Arc, Mutex};

use axum::{
    Router,
    extract::{ConnectInfo, State},
    http::header,
    response::IntoResponse,
    routing::post,
};
use serde_json::{Value, json};

#[derive(Clone, Default)]
struct ConnectionLog {
    peers: Arc<Mutex<Vec<SocketAddr>>>,
}

async fn chat(
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    State(log): State<ConnectionLog>,
) -> impl IntoResponse {
    log.peers.lock().unwrap().push(peer);
    let body = concat!(
        "data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"fixture-model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"answer\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"fixture-model\",\"choices\":[],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":1}}\n\n",
        "data: [DONE]\n\n",
    );
    ([(header::CONTENT_TYPE, "text/event-stream")], body)
}

fn benchmark_run(source: Value) -> Value {
    let mut endpoint = source["resources"]["endpoints"]["profiles"][0].clone();
    endpoint.as_object_mut().unwrap().remove("id");
    json!({
        "benchmark_id": source["identity"]["benchmark_id"],
        "artifact_dir": source["artifact_target"],
        "random_seed": source["identity"]["random_seed"],
        "cfg": {
            "models": source["resources"]["models"],
            "endpoint": endpoint,
            "datasets": [source["workload"]["config"]["dataset"]],
            "phases": source["workload"]["config"]["phases"],
            "tokenizer": source["workload"]["config"]["tokenizer"],
            "transport": {"type": source["transport"]["type"]},
            "runtime": {"workers": source["workload"]["config"]["worker_count"]}
        }
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn v2_sharded_workers_own_persistent_connections_with_balanced_slices() {
    let connection_log = ConnectionLog::default();
    let app = Router::new()
        .route("/v1/chat/completions", post(chat))
        .with_state(connection_log.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(
            listener,
            app.into_make_service_with_connect_info::<SocketAddr>(),
        )
        .await
        .unwrap()
    });

    let artifacts = tempfile::tempdir().unwrap();
    let artifact_target = artifacts.path().join("thread-per-core");
    let request = json!({
        "protocol_version": 2,
        "operation": "execute",
        "run": {
            "identity": {
                "benchmark_id": "thread-per-core-product-proof",
                "random_seed": 11
            },
            "artifact_target": artifact_target,
            "resources": {
                "models": {"items": [{"name": "fixture-model"}]},
                "endpoints": {"profiles": [{
                    "id": "default",
                    "type": "chat",
                    "urls": [format!("http://{address}")],
                    "streaming": true,
                    "use_server_token_count": true,
                    "wait_for_model_timeout": 0.0
                }]}
            },
            "transport": {"type": "http", "config": {}},
            "workload": {"type": "scheduled", "config": {
                "worker_count": 3,
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
                    "name": "warmup",
                    "exclude_from_results": true,
                    "requests": 3,
                    "concurrency": 1
                }, {
                    "type": "concurrency",
                    "name": "profiling",
                    "exclude_from_results": false,
                    "requests": 6,
                    "concurrency": 1
                }]
            }}
        }
    });
    let output = tokio::task::spawn_blocking(move || run_child(request))
        .await
        .unwrap();

    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let terminal: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(terminal["success"], true, "{terminal}");
    assert_eq!(terminal["provenance"]["transport"], "http");
    assert_eq!(terminal["provenance"]["workload"], "scheduled");

    let peers = connection_log.peers.lock().unwrap().clone();
    assert_eq!(peers.len(), 9, "one server request per authored turn");

    // Each of the three sharded sub-cells owns one persistent worker-local
    // connection (concurrency 1 per thread + HTTP keep-alive), so the run uses
    // exactly three distinct peers.
    let distinct: std::collections::BTreeSet<_> = peers.iter().collect();
    assert_eq!(
        distinct.len(),
        3,
        "worker_count=3 must open exactly three worker-local connections: {peers:?}"
    );

    // Deterministic-per-topology slice: each sub-cell owns `owned_positions` of
    // warmup(3) and profiling(6) over three threads — 1 warmup + 2 profiling = 3
    // turns per connection. The global interleaving across threads is a benign
    // scheduling race, but the per-connection turn counts are fixed.
    for peer in &distinct {
        let count = peers.iter().filter(|p| p == peer).count();
        assert_eq!(
            count, 3,
            "each worker-local connection must dispatch its balanced 3-turn slice: {peers:?}"
        );
    }

    let report: Value =
        serde_json::from_slice(&std::fs::read(artifact_target.join("native-v2.json")).unwrap())
            .unwrap();
    assert_eq!(
        report["warmup_metrics"]["request_count"]["series"][0]["stats"]["total"], 3.0,
        "warmup accounting must sum to the phase budget after the sharded merge: {report}"
    );
    assert_eq!(
        report["metrics"]["request_count"]["series"][0]["stats"]["total"], 6.0,
        "profiling accounting must sum to the phase budget after the sharded merge: {report}"
    );

    server.abort();
}
