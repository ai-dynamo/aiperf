// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process proof for the prepare -> artifact commit -> activate sidecar barrier.

use std::io::Write;
use std::process::{Command, Stdio};

use axum::{Router, http::header, response::IntoResponse, routing::post};

const WORKER_SOURCE: &str = r#"
import json
import os
import sys

artifact_dir = os.environ["FIXTURE_ARTIFACT_DIR"]


def read_event():
    line = sys.stdin.readline()
    if not line:
        raise RuntimeError("unexpected EOF")
    return json.loads(line)


def reply(value):
    print(json.dumps(value), flush=True)


initialize = read_event()
assert initialize["protocol_version"] == 1
assert initialize["event"] == "initialize"
assert initialize["config"]["artifact_dir"] == artifact_dir
if os.path.exists(artifact_dir):
    raise RuntimeError("artifact target existed during side-effect-free prepare")
reply({
    "protocol_version": 1,
    "event": "prepared",
    "active": True,
    "disabled_reason": None,
})

activate = read_event()
assert activate == {"protocol_version": 1, "event": "activate"}
if not os.path.isdir(artifact_dir):
    raise RuntimeError("artifact target was absent during activation")
with open(os.path.join(artifact_dir, "live-activation-proof.json"), "w", encoding="utf-8") as output:
    json.dump({"activated_after_artifact_commit": True}, output)
reply({
    "protocol_version": 1,
    "event": "ready",
    "active": False,
    "disabled_reason": "fixture does not export",
})

shutdown = read_event()
assert shutdown["protocol_version"] == 1
assert shutdown["event"] == "shutdown"
reply({
    "protocol_version": 1,
    "event": "terminal",
    "success": True,
    "metric_records": 0,
    "phase_events": 0,
    "processing_errors": 0,
    "dropped_events": shutdown["dropped_events"],
})
"#;

async fn chat_handler() -> impl IntoResponse {
    let body = concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n",
        "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":1}}\n\n",
        "data: [DONE]\n\n",
    );
    ([(header::CONTENT_TYPE, "text/event-stream")], body)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "product wire no longer projects this mode; modules remain linked for later deletion"]
async fn live_worker_activates_only_after_rust_owns_artifact_target() {
    let app = Router::new().route("/v1/chat/completions", post(chat_handler));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let root = tempfile::tempdir().unwrap();
    let artifact_dir = root.path().join("artifacts");
    assert!(!artifact_dir.exists());
    let worker_root = root.path().join("worker");
    let package = worker_root.join("fixture_live_worker");
    std::fs::create_dir_all(&package).unwrap();
    std::fs::write(package.join("__init__.py"), "").unwrap();
    std::fs::write(package.join("worker.py"), WORKER_SOURCE).unwrap();

    let python = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.venv/bin/python");
    assert!(python.is_file());
    let request = serde_json::json!({
        "protocol_version": 1,
        "run": {
            "benchmark_id": "live-streaming-lifecycle",
            "artifact_dir": artifact_dir,
            "models": {"items": [{"name": "mock-model"}]},
            "endpoint": {
                "urls": [format!("http://{address}/v1/chat/completions")],
                "type": "chat",
                "streaming": true,
                "use_server_token_count": true
            },
            "dataset": {
                "type": "synthetic",
                "entries": 1,
                "prompts": {
                    "isl": {"value": 2.0},
                    "osl": {"value": 1.0}
                }
            },
            "phases": [{
                "type": "concurrency",
                "name": "profiling",
                "exclude_from_results": false,
                "requests": 1,
                "concurrency": 1
            }],
            "live_streaming": {
                "python_executable": python,
                "worker_module": "fixture_live_worker.worker",
                "buffer_capacity": 8,
                "otel": {
                    "metrics_url": "http://127.0.0.1:4318/v1/metrics",
                    "stream_metrics_enabled": true,
                    "stream_timing_enabled": true,
                    "custom_resource_attributes": {},
                    "gen_ai_provider": null
                },
                "mlflow": {
                    "tracking_uri": null,
                    "experiment": "aiperf",
                    "run_name": null,
                    "tags": null,
                    "parent_run_id": null,
                    "artifact_globs": null
                }
            }
        }
    });
    let bytes = serde_json::to_vec(&request).unwrap();
    let binary = env!("CARGO_BIN_EXE_aiperf-runner").to_string();
    let output = tokio::task::spawn_blocking(move || {
        let mut child = Command::new(binary)
            .env("PYTHONPATH", worker_root)
            .env("FIXTURE_ARTIFACT_DIR", &artifact_dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        child.stdin.take().unwrap().write_all(&bytes).unwrap();
        child.wait_with_output().unwrap()
    })
    .await
    .unwrap();

    assert!(
        output.status.success(),
        "runner stdout: {}\nrunner stderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    let proof: serde_json::Value = serde_json::from_slice(
        &std::fs::read(root.path().join("artifacts/live-activation-proof.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(proof["activated_after_artifact_commit"], true);
    assert!(root.path().join("artifacts/native-v2.json").is_file());
}
