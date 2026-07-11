// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Real-HTTP proof of the unified dataset-to-segment-to-session-to-transport path.

use std::sync::{Arc, Mutex};

use axum::{
    Router, body::Bytes, extract::State, http::header, response::IntoResponse, routing::post,
};

const SSE: &str = concat!(
    "data: {\"choices\":[{\"delta\":{\"content\":\"live answer\"},\"finish_reason\":null}]}\n\n",
    "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":2}}\n\n",
    "data: [DONE]\n\n",
);

#[derive(Clone, Default)]
struct CapturedBodies(Arc<Mutex<Vec<serde_json::Value>>>);

async fn capture(State(captured): State<CapturedBodies>, body: Bytes) -> impl IntoResponse {
    captured
        .0
        .lock()
        .unwrap()
        .push(serde_json::from_slice(&body).unwrap());
    ([(header::CONTENT_TYPE, "text/event-stream")], SSE)
}

#[tokio::test]
async fn cli_replays_native_multiturn_dataset_and_splices_the_real_reply() {
    let captured = CapturedBodies::default();
    let app = Router::new()
        .route("/v1/chat/completions", post(capture))
        .with_state(captured.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let path = std::env::temp_dir().join(format!(
        "aiperf-native-dataset-{}-{}.json",
        std::process::id(),
        address.port()
    ));
    std::fs::write(
        &path,
        serde_json::to_vec(&serde_json::json!({
            "session_id":"proof",
            "turns":[
                {"text":"first question","timestamp":0,"output_length":2},
                {"text":"second question","delay":1,"output_length":2}
            ]
        }))
        .unwrap(),
    )
    .unwrap();

    let output = tokio::process::Command::new(env!("CARGO_BIN_EXE_aiperf"))
        .arg(format!("http://{address}"))
        .arg("fixture-model")
        .arg("--fixed-schedule")
        .arg("--input-file")
        .arg(&path)
        .arg("--input-format")
        .arg("multi_turn")
        .arg("--osl")
        .arg("2")
        .output()
        .await
        .unwrap();
    std::fs::remove_file(&path).unwrap();
    assert!(
        output.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let bodies = captured.0.lock().unwrap();
    assert_eq!(bodies.len(), 2);
    assert_eq!(bodies[0]["messages"][0]["content"], "first question");
    assert_eq!(
        bodies[1]["messages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|message| message["role"].as_str().unwrap())
            .collect::<Vec<_>>(),
        vec!["user", "assistant", "user"]
    );
    assert_eq!(bodies[1]["messages"][1]["content"], "live answer");
    assert_eq!(bodies[1]["messages"][2]["content"], "second question");
}
