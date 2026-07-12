// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process-level proof of the Python-orchestrator/Rust-runner contract.

use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use axum::{Router, extract::State, http::header, response::IntoResponse, routing::post};

#[test]
fn capabilities_are_a_single_versioned_json_line() {
    let output = Command::new(env!("CARGO_BIN_EXE_aiperf-runner"))
        .arg("--capabilities")
        .output()
        .expect("spawn native runner capability query");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        output.stdout.iter().filter(|byte| **byte == b'\n').count(),
        1
    );
    let capabilities: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(capabilities["event"], "runner_capabilities");
    assert_eq!(capabilities["protocol_versions"], serde_json::json!([1]));
    assert_eq!(capabilities["report_schema_version"], "2.0");
    assert_eq!(
        capabilities["dataset_types"],
        serde_json::json!(["synthetic", "file", "public"])
    );
    assert!(capabilities["phase_types"].as_array().unwrap().len() >= 6);
}

async fn chat_handler() -> impl IntoResponse {
    let body = concat!(
        "data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"a\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"m\",\"choices\":[],\"usage\":{\"prompt_tokens\":8,\"completion_tokens\":1}}\n\n",
        "data: [DONE]\n\n",
    );
    ([(header::CONTENT_TYPE, "text/event-stream")], body)
}

#[derive(Default)]
struct ConcurrencyProbe {
    active: AtomicUsize,
    peak: AtomicUsize,
    total: AtomicUsize,
}

async fn delayed_chat_handler(State(probe): State<Arc<ConcurrencyProbe>>) -> impl IntoResponse {
    let active = probe.active.fetch_add(1, Ordering::SeqCst) + 1;
    probe.peak.fetch_max(active, Ordering::SeqCst);
    probe.total.fetch_add(1, Ordering::SeqCst);
    tokio::time::sleep(Duration::from_millis(50)).await;
    probe.active.fetch_sub(1, Ordering::SeqCst);
    chat_handler().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stdio_child_runs_http_and_commits_native_report() {
    let app = Router::new().route("/v1/chat/completions", post(chat_handler));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let artifacts = tempfile::tempdir().unwrap();
    let request = serde_json::json!({
        "protocol_version": 1,
        "run": {
            "benchmark_id": "stdio-e2e",
            "label": "process proof",
            "trial": 0,
            "random_seed": 7,
            "artifact_dir": artifacts.path(),
            "models": {
                "strategy": "round_robin",
                "items": [{"name": "mock-model"}]
            },
            "endpoint": {
                "urls": [format!("http://{address}/v1/chat/completions")],
                "type": "chat",
                "streaming": true,
                "use_server_token_count": true
            },
            "dataset": {
                "type": "synthetic",
                "entries": 4,
                "prompts": {
                    "isl": {"value": 8.0},
                    "osl": {"value": 1.0},
                    "batch_size": 1
                },
                "turns": {"value": 1.0},
                "turn_delay_ms": {"value": 0.0},
                "turn_delay_ratio": 1.0
            },
            "phases": [{
                "type": "concurrency",
                "name": "profiling",
                "exclude_from_results": false,
                "requests": 4,
                "concurrency": 2
            }],
            "metrics": {
                "slice_duration_seconds": 0.1,
                "slos": {"request_latency": 1000.0}
            },
            "artifacts": {
                "records_path": "profile_export.jsonl",
                "trace": true
            }
        }
    });
    let bytes = serde_json::to_vec(&request).unwrap();
    let binary = env!("CARGO_BIN_EXE_aiperf-runner").to_string();
    let output = tokio::task::spawn_blocking(move || {
        let mut child = Command::new(binary)
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
    let terminal: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(terminal["event"], "run_terminal");
    assert_eq!(terminal["benchmark_id"], "stdio-e2e");
    assert_eq!(terminal["success"], true);

    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(artifacts.path().join("native-v2.json")).unwrap())
            .unwrap();
    assert_eq!(report["schema_version"], "2.0");
    assert_eq!(
        report["metrics"]["request_count"]["series"][0]["stats"]["total"],
        4.0
    );
    assert_eq!(
        report["metrics"]["total_output_tokens"]["series"][0]["stats"]["value"],
        4.0
    );
    assert_eq!(
        report["metrics"]["good_request_count"]["series"][0]["stats"]["total"],
        4.0
    );
    assert!(
        !report["metrics"]["request_latency"]["series"][0]["timeslices"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    let records = std::fs::read_to_string(artifacts.path().join("profile_export.jsonl")).unwrap();
    let rows = records
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(rows.len(), 4);
    assert!(rows.iter().all(|row| {
        row["metadata"]["benchmark_phase"] == "profiling"
            && row["metrics"]["request_latency"]["value"].is_number()
            && row["metrics"]["time_to_first_token"]["value"].is_number()
            && row["trace_data"]["trace_type"] == "aiperf-transport"
    }));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stdio_child_adapts_live_concurrency_and_writes_controller_artifacts() {
    let probe = Arc::new(ConcurrencyProbe::default());
    let app = Router::new()
        .route("/v1/chat/completions", post(delayed_chat_handler))
        .with_state(probe.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let artifacts = tempfile::tempdir().unwrap();
    let request = serde_json::json!({
        "protocol_version": 1,
        "run": {
            "benchmark_id": "adaptive-stdio-e2e",
            "artifact_dir": artifacts.path(),
            "models": {
                "strategy": "round_robin",
                "items": [{"name": "mock-model"}]
            },
            "endpoint": {
                "urls": [format!("http://{address}/v1/chat/completions")],
                "type": "chat",
                "streaming": true,
                "use_server_token_count": true
            },
            "dataset": {
                "type": "synthetic",
                "entries": 2,
                "prompts": {
                    "isl": {"value": 8.0},
                    "osl": {"value": 1.0}
                }
            },
            "phases": [{
                "type": "concurrency",
                "name": "profiling",
                "exclude_from_results": false,
                "duration": 8.0,
                "concurrency": 2,
                "adaptive_scale": {
                    "control_variable": "concurrency",
                    "minimum": 1.0,
                    "maximum": 2.0,
                    "assessment_period_seconds": 1.0,
                    "sustain_duration_seconds": 1.0,
                    "min_completed_requests": 1,
                    "strategy_type": "ramp_until_fail",
                    "step_policy": "fixed_percent_step",
                    "base_step": 1,
                    "max_step_multiplier": 1,
                    "step_percent": 100.0,
                    "sla_filters": [{
                        "metric_tag": "request_latency",
                        "stat": "p95",
                        "op": "le",
                        "threshold": 1000.0
                    }]
                }
            }]
        }
    });
    let bytes = serde_json::to_vec(&request).unwrap();
    let binary = env!("CARGO_BIN_EXE_aiperf-runner").to_string();
    let output = tokio::task::spawn_blocking(move || {
        let mut child = Command::new(binary)
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
    let terminal: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(terminal["success"], true);
    assert!(probe.peak.load(Ordering::SeqCst) >= 2);

    let summary: serde_json::Value = serde_json::from_slice(
        &std::fs::read(artifacts.path().join("adaptive_scale_summary.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(summary["schema_version"], 2);
    assert_eq!(summary["status"], "incomplete");
    assert_eq!(summary["control_variable"], "concurrency");
    assert_eq!(summary["control_value"], 2.0);
    assert_eq!(
        summary["completed_reason"],
        "max_control_value_reached_without_saturation"
    );
    let events = std::fs::read_to_string(artifacts.path().join("adaptive_scale_events.jsonl"))
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(events[0]["event"], "adaptive_phase_started");
    assert!(events.iter().any(|event| {
        event["event"] == "adaptive_decision" && event["control_value_after"] == 2.0
    }));
    assert_eq!(events.last().unwrap()["event"], "adaptive_incomplete");

    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(artifacts.path().join("native-v2.json")).unwrap())
            .unwrap();
    assert!(
        report["metrics"]["request_count"]["series"][0]["stats"]["total"]
            .as_f64()
            .unwrap()
            > 10.0
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stdio_child_drives_every_adaptive_actuator() {
    let probe = Arc::new(ConcurrencyProbe::default());
    let app = Router::new()
        .route("/v1/chat/completions", post(delayed_chat_handler))
        .with_state(probe.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    for variable in ["prefill_concurrency", "request_rate", "users"] {
        probe.peak.store(0, Ordering::SeqCst);
        probe.total.store(0, Ordering::SeqCst);
        let artifacts = tempfile::tempdir().unwrap();
        let (minimum, maximum, turns, mut phase) = match variable {
            "prefill_concurrency" => (
                1.0,
                2.0,
                1.0,
                serde_json::json!({
                    "type": "concurrency",
                    "name": "profiling",
                    "exclude_from_results": false,
                    "duration": 8.0,
                    "concurrency": 2,
                    "prefill_concurrency": 2
                }),
            ),
            "request_rate" => (
                5.0,
                10.0,
                1.0,
                serde_json::json!({
                    "type": "constant",
                    "name": "profiling",
                    "exclude_from_results": false,
                    "duration": 8.0,
                    "rate": 10.0,
                    "concurrency": 2
                }),
            ),
            "users" => (
                1.0,
                2.0,
                2.0,
                serde_json::json!({
                    "type": "user_centric",
                    "name": "profiling",
                    "exclude_from_results": false,
                    "duration": 8.0,
                    "rate": 20.0,
                    "users": 2,
                    "concurrency": 2
                }),
            ),
            _ => unreachable!(),
        };
        phase["adaptive_scale"] = serde_json::json!({
            "control_variable": variable,
            "minimum": minimum,
            "maximum": maximum,
            "assessment_period_seconds": 1.0,
            "sustain_duration_seconds": 1.0,
            "min_completed_requests": 1,
            "strategy_type": "ramp_until_fail",
            "step_policy": "fixed_percent_step",
            "base_step": 1,
            "max_step_multiplier": 1,
            "step_percent": 100.0,
            "sla_filters": [{
                "metric_tag": "request_latency",
                "stat": "p95",
                "op": "le",
                "threshold": 1000.0
            }]
        });
        let request = serde_json::json!({
            "protocol_version": 1,
            "run": {
                "benchmark_id": format!("adaptive-{variable}-stdio-e2e"),
                "artifact_dir": artifacts.path(),
                "models": {"items": [{"name": "mock-model"}]},
                "endpoint": {
                    "urls": [format!("http://{address}/v1/chat/completions")],
                    "type": "chat",
                    "streaming": true,
                    "use_server_token_count": true
                },
                "dataset": {
                    "type": "synthetic",
                    "entries": 4,
                    "prompts": {
                        "isl": {"value": 8.0},
                        "osl": {"value": 1.0}
                    },
                    "turns": {"value": turns}
                },
                "phases": [phase]
            }
        });
        let bytes = serde_json::to_vec(&request).unwrap();
        let binary = env!("CARGO_BIN_EXE_aiperf-runner").to_string();
        let output = tokio::task::spawn_blocking(move || {
            let mut child = Command::new(binary)
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
            "{variable} stdout: {}\n{variable} stderr: {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
        let summary: serde_json::Value = serde_json::from_slice(
            &std::fs::read(artifacts.path().join("adaptive_scale_summary.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(summary["status"], "incomplete", "{variable}");
        assert_eq!(summary["control_variable"], variable, "{variable}");
        assert_eq!(summary["control_value"], maximum, "{variable}");
        assert_eq!(summary["control"]["target_value"], maximum, "{variable}");
        let events =
            std::fs::read_to_string(artifacts.path().join("adaptive_scale_events.jsonl")).unwrap();
        assert!(events.contains("adaptive_decision"), "{variable}");
        if variable == "prefill_concurrency" {
            assert!(probe.peak.load(Ordering::SeqCst) >= 2);
        }
        if variable == "request_rate" {
            assert!(probe.total.load(Ordering::SeqCst) >= 10);
        }
    }
}
