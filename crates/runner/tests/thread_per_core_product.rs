// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Product proof for native thread-per-core HTTP placement.
//!
//! Config v2 resolves one worker count, the runner retains one phase
//! coordinator and one logical dispatcher, and only prepared HTTP turns cross
//! the placement seam. This process test proves that three authored workers
//! become three persistent worker-local transport pools selected in stable
//! round-robin order while warmup and profiling accounting remains phase-owned.

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

fn capabilities() -> Value {
    let output = Command::new(env!("CARGO_BIN_EXE_aiperf-runner"))
        .arg("--capabilities")
        .output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");
    serde_json::from_slice(&output.stdout).unwrap()
}

fn run_child(request: Value) -> Output {
    let bytes = serde_json::to_vec(&request).unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_aiperf-runner"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(&bytes).unwrap();
    child.wait_with_output().unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn v2_workers_own_transport_pools_below_one_phase_coordinator() {
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

    let capabilities = capabilities();
    let artifacts = tempfile::tempdir().unwrap();
    let artifact_target = artifacts.path().join("thread-per-core");
    let request = json!({
        "protocol_version": 2,
        "operation": "execute",
        "expected_distribution_id": capabilities["distribution_id"],
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
    let worker_peers = &peers[..3];
    assert!(
        worker_peers
            .iter()
            .enumerate()
            .all(|(index, peer)| !worker_peers[..index].contains(peer)),
        "each placement worker must own a distinct persistent connection: {peers:?}"
    );
    for (turn, peer) in peers.iter().enumerate() {
        assert_eq!(
            peer,
            &worker_peers[turn % worker_peers.len()],
            "turn {turn} did not follow deterministic round-robin placement: {peers:?}"
        );
    }

    let report: Value =
        serde_json::from_slice(&std::fs::read(artifact_target.join("native-v2.json")).unwrap())
            .unwrap();
    assert_eq!(
        report["warmup_metrics"]["request_count"]["series"][0]["stats"]["total"], 3.0,
        "warmup accounting must remain coordinator-owned: {report}"
    );
    assert_eq!(
        report["metrics"]["request_count"]["series"][0]["stats"]["total"], 6.0,
        "profiling accounting must remain coordinator-owned: {report}"
    );

    server.abort();
}
