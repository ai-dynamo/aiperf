// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Adaptive CLI acceptance test: flags -> live controller -> schema-v2 artifacts.

use axum::{Router, http::header, response::IntoResponse, routing::post};
use serde_json::Value;

fn invalid_adaptive_command(extra: &[&str]) -> std::process::Output {
    let mut command = std::process::Command::new(env!("CARGO_BIN_EXE_aiperf"));
    command.args([
        "http://127.0.0.1:9",
        "fixture-model",
        "--duration",
        "3",
        "--adaptive-scale",
        "--adaptive-control-min",
        "1",
        "--adaptive-control-max",
        "2",
        "--adaptive-sustain-duration",
        "1",
        "--adaptive-scale-sla",
        "request_latency:p95:le:100",
    ]);
    command.args(extra).output().unwrap()
}

async fn chat_handler() -> impl IntoResponse {
    let stream = "data: {\"choices\":[{\"delta\":{\"content\":\"x\"},\"finish_reason\":null}]}\n\n\
                  data: [DONE]\n\n";
    ([(header::CONTENT_TYPE, "text/event-stream")], stream)
}

#[tokio::test]
async fn cli_runs_adaptive_scale_and_writes_terminal_artifacts() {
    let app = Router::new().route("/v1/chat/completions", post(chat_handler));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let artifact_dir = std::env::temp_dir().join(format!(
        "aiperf-adaptive-cli-{}-{}",
        std::process::id(),
        address.port()
    ));
    let output = tokio::process::Command::new(env!("CARGO_BIN_EXE_aiperf"))
        .arg(format!("http://{address}"))
        .arg("fixture-model")
        .arg("--duration")
        .arg("3")
        .arg("--concurrency")
        .arg("2")
        .arg("--isl")
        .arg("4")
        .arg("--osl")
        .arg("1")
        .arg("--adaptive-scale")
        .arg("--adaptive-control-min")
        .arg("1")
        .arg("--adaptive-control-max")
        .arg("2")
        .arg("--adaptive-assessment-period")
        .arg("1")
        .arg("--adaptive-sustain-duration")
        .arg("1")
        .arg("--adaptive-scale-sla")
        .arg("request_latency:p95:le:0")
        .arg("--adaptive-base-step")
        .arg("1")
        .arg("--adaptive-max-step-multiplier")
        .arg("1")
        .arg("--adaptive-artifact-dir")
        .arg(&artifact_dir)
        .output()
        .await
        .unwrap();
    assert!(
        output.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let events = std::fs::read_to_string(artifact_dir.join("adaptive_scale_events.jsonl")).unwrap();
    assert!(events.contains("\"adaptive_phase_started\""));
    assert!(events.contains("\"adaptive_window\""));
    assert!(events.contains("\"adaptive_failed\""));
    let summary: Value = serde_json::from_slice(
        &std::fs::read(artifact_dir.join("adaptive_scale_summary.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(summary["schema_version"], 2);
    assert_eq!(summary["status"], "failed");
    assert_eq!(summary["control_variable"], "concurrency");
    assert_eq!(
        summary["completed_reason"],
        "no_sustainable_concurrency_found"
    );
    std::fs::remove_dir_all(artifact_dir).unwrap();
}

#[test]
fn cli_rejects_invalid_adaptive_timing_and_step_settings_before_dispatch() {
    for (extra, expected) in [
        (
            vec!["--adaptive-assessment-period", "0.5"],
            "--adaptive-assessment-period must be finite and >= 1 second",
        ),
        (
            vec!["--adaptive-base-step", "0"],
            "--adaptive-base-step must be >= 1",
        ),
        (
            vec!["--adaptive-scale-strategy-type", "binary_search"],
            "unknown --adaptive-scale-strategy-type",
        ),
        (
            vec![
                "--adaptive-step-policy",
                "fixed_percent_step",
                "--adaptive-step-percent",
                "NaN",
            ],
            "--adaptive-step-percent must be positive and finite",
        ),
        (
            vec![
                "--request-rate",
                "10",
                "--prefill-concurrency",
                "2",
                "--adaptive-control-variable",
                "prefill_concurrency",
            ],
            "adaptive prefill_concurrency requires a session --concurrency cap",
        ),
    ] {
        let output = invalid_adaptive_command(&extra);
        assert!(!output.status.success());
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains(expected),
            "expected {expected:?} in stderr:\n{stderr}"
        );
    }
}
