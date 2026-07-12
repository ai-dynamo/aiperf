// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Opt-in proof that AgentLab/BrowserGym calls traverse the normal Rust pipeline.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::{Json, Router, extract::State, http::header, response::IntoResponse, routing::post};
use serde_json::Value;

const BROWSER_AGENTIC_LOCK_SHA256: &str =
    "2e998cbe869fa6ae21b3ce52264a2cf188316941bb2ebf8e256461a989aedb66";

#[derive(Clone, Default)]
struct Captured(Arc<Mutex<Vec<Value>>>);

async fn browser_completion(
    State(captured): State<Captured>,
    Json(body): Json<Value>,
) -> impl IntoResponse {
    let bid = click_test_bid(&body).expect("AgentLab prompt omitted MiniWoB button bid");
    let request_index = captured.0.lock().unwrap().len();
    captured.0.lock().unwrap().push(body);
    let response_id = format!("browsergym-e2e-{request_index}");
    let answer =
        format!("I'm clicking the button as requested.\n<action>\nclick('{bid}')\n</action>");
    let stream = format!(
        "data: {{\"id\":{response_id:?},\"object\":\"chat.completion.chunk\",\"choices\":[{{\"delta\":{{\"content\":{answer:?}}},\"finish_reason\":\"stop\"}}]}}\n\n\
         data: {{\"id\":{response_id:?},\"object\":\"chat.completion.chunk\",\"choices\":[],\"usage\":{{\"prompt_tokens\":101,\"completion_tokens\":19,\"prompt_tokens_details\":{{\"cached_tokens\":7}}}}}}\n\n\
         data: [DONE]\n\n"
    );
    ([(header::CONTENT_TYPE, "text/event-stream")], stream)
}

fn click_test_bid(body: &Value) -> Option<String> {
    let messages = body.get("messages")?.as_array()?;
    for message in messages {
        let Some(content) = message.get("content") else {
            continue;
        };
        let mut texts = Vec::new();
        if let Some(text) = content.as_str() {
            texts.push(text);
        } else if let Some(parts) = content.as_array() {
            texts.extend(parts.iter().filter_map(|part| part.get("text")?.as_str()));
        }
        for line in texts.into_iter().flat_map(str::lines) {
            let line = line.trim_start();
            let Some(rest) = line.strip_prefix('[') else {
                continue;
            };
            let Some((candidate, suffix)) = rest.split_once(']') else {
                continue;
            };
            if candidate
                .chars()
                .all(|character| character.is_ascii_digit())
                && suffix.to_ascii_lowercase().contains("button")
            {
                return Some(candidate.to_string());
            }
        }
    }
    None
}

#[tokio::test]
#[ignore = "requires the pinned browser worker, local MiniWoB, and Playwright Chromium"]
async fn real_browsergym_episode_uses_rust_transport_and_canonical_reward() {
    let python = std::env::var_os("AIPERF_BROWSER_AGENTIC_PYTHON")
        .expect("set AIPERF_BROWSER_AGENTIC_PYTHON to the hash-pinned worker Python");
    assert!(
        std::env::var_os("MINIWOB_URL").is_some(),
        "set MINIWOB_URL to the pinned MiniWoB++ checkout"
    );
    let timeout_seconds = std::env::var("AIPERF_BROWSER_E2E_TIMEOUT_SECONDS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(120);

    let captured = Captured::default();
    let app = Router::new()
        .route("/v1/chat/completions", post(browser_completion))
        .with_state(captured.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let root = std::env::temp_dir().join(format!(
        "aiperf_real_browsergym_{}_{}",
        std::process::id(),
        address.port()
    ));
    let artifacts = root.join("episodes");
    let report_path = root.join("report.json");
    std::fs::create_dir_all(&artifacts).unwrap();

    let output = tokio::time::timeout(
        Duration::from_secs(timeout_seconds),
        tokio::process::Command::new(env!("CARGO_BIN_EXE_aiperf"))
            .arg(format!("http://{address}"))
            .arg("browsergym-e2e-model")
            .arg("--agentic-benchmark")
            .arg("browsergym/miniwob@0.14.3")
            .arg("--agentic-tasks")
            .arg("miniwob.click-test")
            .arg("--agentic-max-episodes")
            .arg("1")
            .arg("--agentic-task-concurrency")
            .arg("1")
            .arg("--agentic-environment")
            .arg("browsergym")
            .arg("--agentic-output-dir")
            .arg(&artifacts)
            .arg("--agentic-max-tokens")
            .arg("512")
            .arg("--agentic-context-window")
            .arg("128000")
            .arg("--concurrency")
            .arg("1")
            .arg("--json")
            .arg(&report_path)
            .env("AIPERF_ACCURACY_PYTHON", python)
            .kill_on_drop(true)
            .output(),
    )
    .await
    .expect("real BrowserGym acceptance run timed out")
    .unwrap();
    server.abort();
    assert!(
        output.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let report: Value = serde_json::from_slice(&std::fs::read(&report_path).unwrap()).unwrap();
    assert_eq!(report["schema_version"], "2.0");
    assert_eq!(report["run"]["mode"], "agentic_accuracy");
    assert_eq!(report["evaluator"]["packages"]["agentlab"], "0.4.2");
    assert_eq!(report["evaluator"]["packages"]["browsergym-core"], "0.14.3");
    assert_eq!(
        report["evaluator"]["dependency_lock_sha256"],
        BROWSER_AGENTIC_LOCK_SHA256
    );
    assert_eq!(
        report["agentic"]["evaluator"]["harness"],
        "agentlab-browsergym"
    );
    assert_eq!(report["agentic"]["evaluator"]["environment"], "browsergym");
    assert_eq!(
        report["evaluator"]["dataset"]["provider"],
        "BrowserGym DEFAULT_BENCHMARKS"
    );
    assert_eq!(
        report["evaluator"]["dataset"]["benchmark"],
        "browsergym/miniwob@0.14.3"
    );
    assert!(
        report["evaluator"]["dataset"]["revision"]
            .as_str()
            .unwrap()
            .starts_with("sha256:")
    );
    assert_eq!(report["agentic"]["summary"]["episode_count"], 1);
    assert_eq!(report["agentic"]["summary"]["completed_count"], 1);
    assert_eq!(
        report["agentic"]["summary"]["infrastructure_error_count"],
        0
    );
    assert_eq!(report["agentic"]["summary"]["primary_score"], 1.0);
    let record = &report["agentic"]["records"][0];
    assert_eq!(record["outcome"], "completed");
    assert_eq!(record["rewards"]["reward"], 1.0);
    assert_eq!(record["auxiliary_model_calls"], 0);
    assert!(
        std::path::Path::new(record["artifact_path"].as_str().unwrap())
            .join("summary_info.json")
            .is_file()
    );

    let requests = captured.0.lock().unwrap();
    assert_eq!(
        record["model_calls"].as_u64().unwrap(),
        requests.len() as u64
    );
    assert_eq!(
        record["primary_model_calls"].as_u64().unwrap(),
        requests.len() as u64
    );
    assert!(requests.iter().all(|body| body["stream"] == true));
    assert!(
        requests
            .iter()
            .all(|body| body["stream_options"]["include_usage"] == true)
    );
    assert!(
        requests
            .iter()
            .all(|body| body["model"] == "browsergym-e2e-model")
    );
    assert!(
        requests
            .iter()
            .all(|body| !body["messages"].as_array().unwrap().is_empty())
    );
    drop(requests);

    eprintln!(
        "AIPERF_BROWSERGYM_E2E_PROOF={}",
        serde_json::json!({
            "dataset": report["evaluator"]["dataset"],
            "task": record["task"],
            "reward": record["rewards"]["reward"],
            "model_calls": record["model_calls"],
            "agentlab": report["evaluator"]["packages"]["agentlab"],
            "browsergym": report["evaluator"]["packages"]["browsergym-core"],
            "dependency_lock_sha256": report["evaluator"]["dependency_lock_sha256"],
        })
    );

    std::fs::remove_dir_all(root).unwrap();
}
