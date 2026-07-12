// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process-level proof that `dag_jsonl` selects the direct Graph-IR dispatcher.

use std::io::Write;
use std::process::{Command, Output, Stdio};
use std::sync::{Arc, Mutex};

use axum::{
    Router, body::Bytes, extract::State, http::header, response::IntoResponse, routing::post,
};
use serde_json::{Value, json};

#[derive(Default)]
struct WireCapture {
    bodies: Mutex<Vec<Value>>,
}

async fn graph_chat_handler(
    State(capture): State<Arc<WireCapture>>,
    body: Bytes,
) -> impl IntoResponse {
    let body: Value = serde_json::from_slice(&body).unwrap();
    let last_user = body["messages"]
        .as_array()
        .unwrap()
        .iter()
        .rev()
        .find(|message| message["role"] == "user")
        .and_then(|message| message["content"].as_str())
        .unwrap()
        .to_string();
    capture.bodies.lock().unwrap().push(body);
    let answer = format!("answer-{last_user}");
    let stream = format!(
        "data: {{\"choices\":[{{\"delta\":{{\"content\":{}}}}}]}}\n\n\
         data: {{\"choices\":[],\"usage\":{{\"prompt_tokens\":8,\"completion_tokens\":1}}}}\n\n\
         data: [DONE]\n\n",
        serde_json::to_string(&answer).unwrap(),
    );
    ([(header::CONTENT_TYPE, "text/event-stream")], stream)
}

fn runner_request(base_url: &str, artifact_dir: &std::path::Path, records: Value) -> Value {
    json!({
        "protocol_version": 1,
        "run": {
            "benchmark_id": "direct-dag-stdio",
            "random_seed": 19,
            "workers": 2,
            "artifact_dir": artifact_dir,
            "models": {
                "strategy": "round_robin",
                "items": [{"name": "fixture-model"}]
            },
            "endpoint": {
                "urls": [base_url],
                "type": "chat",
                "streaming": true,
                "use_server_token_count": true
            },
            "dataset": {
                "type": "file",
                "records": records,
                "format": "dag_jsonl"
            },
            "phases": [{
                "type": "concurrency",
                "name": "profiling",
                "exclude_from_results": false,
                "requests": 4,
                "concurrency": 1
            }],
            "metrics": {},
            "artifacts": {}
        }
    })
}

async fn run_child(request: Value) -> Output {
    let bytes = serde_json::to_vec(&request).unwrap();
    tokio::task::spawn_blocking(move || {
        let mut child = Command::new(env!("CARGO_BIN_EXE_aiperf-runner"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let mut stdin = child.stdin.take().unwrap();
        stdin.write_all(&bytes).unwrap();
        drop(stdin);
        child.wait_with_output().unwrap()
    })
    .await
    .unwrap()
}

fn contents(body: &Value) -> Vec<String> {
    body["messages"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|message| message["content"].as_str().map(str::to_string))
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn direct_dag_runner_dispatches_fork_spawn_join_with_canonical_histories() {
    let capture = Arc::new(WireCapture::default());
    let app = Router::new()
        .route("/v1/chat/completions", post(graph_chat_handler))
        .with_state(capture.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let artifacts = tempfile::tempdir().unwrap();
    let records = json!([
        {"session_id":"root","turns":[
            {
                "messages":[{"role":"user","content":"root-0"}],
                "forks":[{"child":"fork","background":true}],
                "spawns":[{"children":["spawn"],"join_at":1}]
            },
            {"messages":[{"role":"user","content":"root-1"}]}
        ]},
        {"session_id":"fork","turns":[
            {"messages":[{"role":"user","content":"fork-0"}]}
        ]},
        {"session_id":"spawn","turns":[
            {"messages":[{"role":"user","content":"spawn-0"}]}
        ]}
    ]);
    let output = run_child(runner_request(
        &format!("http://{address}"),
        artifacts.path(),
        records,
    ))
    .await;
    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let terminal: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(terminal["success"], true);
    let report: Value =
        serde_json::from_slice(&std::fs::read(artifacts.path().join("native-v2.json")).unwrap())
            .unwrap();
    assert_eq!(report["run"]["mode"], "graph");
    assert_eq!(
        report["metrics"]["request_count"]["series"][0]["stats"]["total"],
        4.0
    );

    let bodies = capture.bodies.lock().unwrap();
    assert_eq!(bodies.len(), 4);
    let fork = bodies
        .iter()
        .find(|body| contents(body).contains(&"fork-0".into()))
        .unwrap();
    assert_eq!(contents(fork), ["root-0", "answer-root-0", "fork-0"]);
    let spawn = bodies
        .iter()
        .find(|body| contents(body).contains(&"spawn-0".into()))
        .unwrap();
    assert_eq!(contents(spawn), ["spawn-0"]);
    let joined = bodies
        .iter()
        .find(|body| contents(body).contains(&"root-1".into()))
        .unwrap();
    assert_eq!(contents(joined), ["root-0", "answer-root-0", "root-1"]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn malformed_direct_dag_fails_before_any_http_request() {
    let capture = Arc::new(WireCapture::default());
    let app = Router::new()
        .route("/v1/chat/completions", post(graph_chat_handler))
        .with_state(capture.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let artifacts = tempfile::tempdir().unwrap();
    let output = run_child(runner_request(
        &format!("http://{address}"),
        artifacts.path(),
        json!([{
            "session_id":"root",
            "turns":[{"messages":[{"role":"user"}],"spawns":["missing"]}]
        }]),
    ))
    .await;
    assert!(!output.status.success());
    let terminal: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(terminal["success"], false);
    assert!(capture.bodies.lock().unwrap().is_empty());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cycled_dag_roots_publish_unique_trace_instance_conversation_ids() {
    let capture = Arc::new(WireCapture::default());
    let app = Router::new()
        .route("/v1/chat/completions", post(graph_chat_handler))
        .with_state(capture.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let artifacts = tempfile::tempdir().unwrap();
    let mut request = runner_request(
        &format!("http://{address}"),
        artifacts.path(),
        json!([{
            "session_id":"root",
            "turns":[{"messages":[{"role":"user","content":"root"}]}]
        }]),
    );
    request["run"]["phases"][0]["requests"] = json!(2);
    request["run"]["artifacts"]["records_path"] = json!("records.jsonl");

    let output = run_child(request).await;
    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(capture.bodies.lock().unwrap().len(), 2);

    let records = std::fs::read_to_string(artifacts.path().join("records.jsonl")).unwrap();
    let conversation_ids = records
        .lines()
        .map(|line| {
            serde_json::from_str::<Value>(line).unwrap()["metadata"]["conversation_id"]
                .as_str()
                .unwrap()
                .to_string()
        })
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        conversation_ids,
        ["root#instance-0".to_string(), "root#instance-1".to_string()]
            .into_iter()
            .collect()
    );
}
