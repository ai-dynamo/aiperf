// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process-level proof of the Python-orchestrator/Rust-runner contract.

use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use axum::{
    Router, extract::State, http::header, response::IntoResponse, routing::get, routing::post,
};

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
    assert!(
        capabilities["endpoint_types"]
            .as_array()
            .unwrap()
            .contains(&serde_json::json!("chat"))
    );
    assert_eq!(
        capabilities["dataset_types"],
        serde_json::json!(["synthetic", "file", "public"])
    );
    assert!(capabilities["phase_types"].as_array().unwrap().len() >= 6);
    assert!(
        capabilities["run_features"]
            .as_array()
            .unwrap()
            .contains(&serde_json::json!("outputs_json"))
    );
    assert!(
        capabilities["run_features"]
            .as_array()
            .unwrap()
            .contains(&serde_json::json!("raw_records"))
    );
    assert!(
        capabilities["run_features"]
            .as_array()
            .unwrap()
            .contains(&serde_json::json!("http_transport_policy"))
    );
    assert!(
        capabilities["telemetry_source_types"]
            .as_array()
            .unwrap()
            .contains(&serde_json::json!("dcgm"))
    );
    assert!(
        capabilities["telemetry_source_types"]
            .as_array()
            .unwrap()
            .contains(&serde_json::json!("python"))
    );
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

struct DcgmProbe {
    inference: Arc<ConcurrencyProbe>,
    scrapes: AtomicUsize,
    inference_count_at_first_scrape: AtomicUsize,
}

async fn dcgm_handler(State(probe): State<Arc<DcgmProbe>>) -> impl IntoResponse {
    let scrape = probe.scrapes.fetch_add(1, Ordering::SeqCst) + 1;
    let inference_count = probe.inference.total.load(Ordering::SeqCst);
    let _ = probe.inference_count_at_first_scrape.compare_exchange(
        usize::MAX,
        inference_count,
        Ordering::SeqCst,
        Ordering::SeqCst,
    );
    let energy_millijoules = scrape as u64 * 1_000_000_000;
    format!(
        concat!(
            "DCGM_FI_DEV_POWER_USAGE{{gpu=\"0\",UUID=\"GPU-e2e\",modelName=\"H100\",Hostname=\"node\"}} 250\n",
            "DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION{{gpu=\"0\",UUID=\"GPU-e2e\",modelName=\"H100\",Hostname=\"node\"}} {energy_millijoules}\n",
            "DCGM_FI_DEV_GPU_UTIL{{gpu=\"0\",UUID=\"GPU-e2e\",modelName=\"H100\",Hostname=\"node\"}} 80\n",
            "DCGM_FI_DEV_SM_CLOCK{{gpu=\"0\",UUID=\"GPU-e2e\",modelName=\"H100\",Hostname=\"node\"}} 1500\n"
        ),
        energy_millijoules = energy_millijoules,
    )
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
                "use_server_token_count": true,
                "api_key": "raw-e2e-secret",
                "headers": {"X-Custom-Tracking": "trace-e2e"}
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
                "raw_path": "profile_export_raw.jsonl",
                "outputs_path": "outputs.json",
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
    let raw_records =
        std::fs::read_to_string(artifacts.path().join("profile_export_raw.jsonl")).unwrap();
    assert!(!raw_records.contains("raw-e2e-secret"));
    let raw_rows = raw_records
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(raw_rows.len(), 4);
    assert!(raw_rows.iter().all(|row| {
        row["metadata"]["benchmark_phase"] == "profiling"
            && row["payload"]["model"] == "mock-model"
            && row["request_headers"]["Authorization"] == "<redacted>"
            && row["request_headers"]["X-Custom-Tracking"] == "trace-e2e"
            && row["status"] == 200
            && row["response_headers"]["content-type"] == "text/event-stream"
            && row["responses"]
                .as_array()
                .is_some_and(|responses| responses.len() == 4)
    }));
    let outputs: serde_json::Value =
        serde_json::from_slice(&std::fs::read(artifacts.path().join("outputs.json")).unwrap())
            .unwrap();
    assert_eq!(outputs["schema_version"], "1.0");
    assert_eq!(outputs["data"].as_array().unwrap().len(), 4);
    assert!(outputs["data"].as_array().unwrap().iter().all(|row| {
        row["response_text"] == "a"
            && row["metrics"]["output_token_count"] == 1.0
            && row["metrics"]["output_sequence_length"] == 1.0
            && row["metrics"]["request_latency"].is_number()
    }));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stdio_child_bounds_dcgm_to_profiling_and_joins_native_results() {
    let inference_probe = Arc::new(ConcurrencyProbe::default());
    let inference_app = Router::new()
        .route("/v1/chat/completions", post(delayed_chat_handler))
        .with_state(inference_probe.clone());
    let inference_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let inference_address = inference_listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(inference_listener, inference_app)
            .await
            .unwrap()
    });

    let dcgm_probe = Arc::new(DcgmProbe {
        inference: inference_probe.clone(),
        scrapes: AtomicUsize::new(0),
        inference_count_at_first_scrape: AtomicUsize::new(usize::MAX),
    });
    let dcgm_app = Router::new()
        .route("/metrics", get(dcgm_handler))
        .with_state(dcgm_probe.clone());
    let dcgm_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let dcgm_address = dcgm_listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(dcgm_listener, dcgm_app).await.unwrap() });

    let artifacts = tempfile::tempdir().unwrap();
    let request = serde_json::json!({
        "protocol_version": 1,
        "run": {
            "benchmark_id": "gpu-telemetry-stdio-e2e",
            "artifact_dir": artifacts.path(),
            "models": {"items": [{"name": "mock-model"}]},
            "endpoint": {
                "urls": [format!("http://{inference_address}/v1/chat/completions")],
                "type": "chat",
                "streaming": true,
                "use_server_token_count": true
            },
            "dataset": {
                "type": "synthetic",
                "entries": 6,
                "prompts": {
                    "isl": {"value": 8.0},
                    "osl": {"value": 1.0}
                }
            },
            "phases": [
                {
                    "type": "concurrency",
                    "name": "warmup",
                    "exclude_from_results": true,
                    "requests": 2,
                    "concurrency": 1
                },
                {
                    "type": "concurrency",
                    "name": "profiling",
                    "exclude_from_results": false,
                    "requests": 4,
                    "concurrency": 2
                }
            ],
            "gpu_telemetry": {
                "collection_interval_ns": 10_000_000,
                "request_timeout_ns": 1_000_000_000,
                "records_path": "gpu_telemetry_export.jsonl",
                "sources": [{"type": "dcgm", "url": format!("http://{dcgm_address}")}]
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
    assert_eq!(
        dcgm_probe
            .inference_count_at_first_scrape
            .load(Ordering::SeqCst),
        2,
        "the first forced scrape must occur after warmup and before profiling"
    );
    assert!(dcgm_probe.scrapes.load(Ordering::SeqCst) >= 3);

    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(artifacts.path().join("native-v2.json")).unwrap())
            .unwrap();
    assert_eq!(
        report["metrics"]["request_count"]["series"][0]["stats"]["total"],
        4.0
    );
    assert_eq!(
        report["metrics"]["total_gpu_power"]["series"][0]["stats"]["value"],
        250.0
    );
    assert!(
        report["metrics"]["total_gpu_energy"]["series"][0]["stats"]["value"]
            .as_f64()
            .unwrap()
            > 0.0
    );
    assert!(
        report["metrics"]["output_tokens_per_joule"]["series"][0]["stats"]["value"]
            .as_f64()
            .unwrap()
            > 0.0
    );
    assert_eq!(
        report["metrics"]["gpu_power_usage"]["series"][0]["labels"]["gpu_uuid"],
        "GPU-e2e"
    );

    let telemetry =
        std::fs::read_to_string(artifacts.path().join("gpu_telemetry_export.jsonl")).unwrap();
    let rows = telemetry
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
        .collect::<Vec<_>>();
    assert!(rows.len() >= 3);
    assert!(rows.iter().all(|row| {
        row["gpu_uuid"] == "GPU-e2e"
            && row["dcgm_url"] == format!("http://{dcgm_address}/metrics")
            && row["telemetry_data"]["gpu_power_usage"] == 250.0
    }));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stdio_child_supervises_canonical_python_custom_dcgm_source() {
    let inference_probe = Arc::new(ConcurrencyProbe::default());
    let inference_app = Router::new()
        .route("/v1/chat/completions", post(delayed_chat_handler))
        .with_state(inference_probe);
    let inference_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let inference_address = inference_listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(inference_listener, inference_app)
            .await
            .unwrap()
    });

    let dcgm_probe = Arc::new(DcgmProbe {
        inference: Arc::new(ConcurrencyProbe::default()),
        scrapes: AtomicUsize::new(0),
        inference_count_at_first_scrape: AtomicUsize::new(usize::MAX),
    });
    let dcgm_app = Router::new()
        .route("/metrics", get(dcgm_handler))
        .with_state(dcgm_probe.clone());
    let dcgm_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let dcgm_address = dcgm_listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(dcgm_listener, dcgm_app).await.unwrap() });

    let artifacts = tempfile::tempdir().unwrap();
    let metrics_file = artifacts.path().join("dcgm-custom.csv");
    std::fs::write(
        &metrics_file,
        "DCGM_FI_DEV_SM_CLOCK,gauge,SM Clock Frequency (in MHz)\n",
    )
    .unwrap();
    let python = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.venv/bin/python");
    assert!(python.is_file());
    let request = serde_json::json!({
        "protocol_version": 1,
        "run": {
            "benchmark_id": "python-gpu-telemetry-stdio-e2e",
            "artifact_dir": artifacts.path(),
            "models": {"items": [{"name": "mock-model"}]},
            "endpoint": {
                "urls": [format!("http://{inference_address}/v1/chat/completions")],
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
                }
            },
            "phases": [{
                "type": "concurrency",
                "name": "profiling",
                "exclude_from_results": false,
                "requests": 4,
                "concurrency": 2
            }],
            "gpu_telemetry": {
                "collection_interval_ns": 10_000_000,
                "request_timeout_ns": 1_000_000_000,
                "records_path": "gpu_telemetry_export.jsonl",
                "custom_metrics": [{
                    "name": "sm_clock",
                    "header": "SM Clock Frequency",
                    "unit": "megahertz"
                }],
                "sources": [{
                    "type": "python",
                    "collector": "dcgm",
                    "url": format!("http://{dcgm_address}"),
                    "metrics_file": metrics_file,
                    "python_executable": python,
                    "worker_module": "aiperf.gpu_telemetry.worker"
                }]
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
    let scrape_count = dcgm_probe.scrapes.load(Ordering::SeqCst);
    assert!(
        scrape_count >= 2,
        "expected forced start/end scrapes, got {scrape_count}; runner stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(artifacts.path().join("native-v2.json")).unwrap())
            .unwrap();
    assert_eq!(report["metrics"]["sm_clock"]["unit"], "MHz");
    assert_eq!(
        report["metrics"]["sm_clock"]["series"][0]["stats"]["avg"],
        1500.0
    );
    assert_eq!(
        report["metrics"]["sm_clock"]["series"][0]["labels"]["gpu_uuid"],
        "GPU-e2e"
    );
    let telemetry =
        std::fs::read_to_string(artifacts.path().join("gpu_telemetry_export.jsonl")).unwrap();
    assert!(telemetry.lines().all(|line| {
        serde_json::from_str::<serde_json::Value>(line).unwrap()["telemetry_data"]["sm_clock"]
            == 1500.0
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stdio_child_uses_python_grading_but_rust_dispatches_every_accuracy_request() {
    let app = Router::new().route("/v1/chat/completions", post(chat_handler));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let worker_dir = tempfile::tempdir().unwrap();
    std::fs::write(
        worker_dir.path().join("fixture_accuracy_worker.py"),
        r#"
import json
import sys

PROBLEMS = [
    {
        "problem_id": "fixture-0",
        "task": "task-a",
        "prompt": "first fixture",
        "messages": [{"role": "user", "content": "first fixture"}],
        "generation": {"max_tokens": 1, "temperature": 0.0, "top_p": 1.0, "stop": []},
    },
    {
        "problem_id": "fixture-1",
        "task": "task-b",
        "prompt": "second fixture",
        "messages": [{"role": "user", "content": "second fixture"}],
        "generation": {"max_tokens": 1, "temperature": 0.0, "top_p": 1.0, "stop": []},
    },
]

for line in sys.stdin:
    request = json.loads(line)
    op = request["op"]
    if op == "hello":
        result = {
            "protocol": 1,
            "worker_version": "fixture-1",
            "python_version": sys.version.split()[0],
            "python_executable": sys.executable,
            "packages": {"fixture-evaluator": "1"},
            "worker_source_sha256": "a" * 64,
            "dependency_lock_sha256": "b" * 64,
            "container_digest": None,
            "capabilities": ["load", "next_problems", "grade_batch", "grader_override", "shutdown"],
        }
    elif op == "load":
        result = {
            "benchmark": request["benchmark"],
            "problem_count": len(PROBLEMS),
            "dataset": {
                "provider": "fixture",
                "benchmark": request["benchmark"],
                "repository": "fixture/repository",
                "subset": "default",
                "revision": "fixture-revision",
                "evaluation_splits": ["test"],
                "task_version": 1,
            },
            "grader": request.get("grader") or "fixture-python-grader",
        }
    elif op == "next_problems":
        start = request["offset"]
        end = min(start + request["limit"], len(PROBLEMS))
        result = {"items": PROBLEMS[start:end], "next_offset": end, "done": end == len(PROBLEMS)}
    elif op == "grade_batch":
        result = {"items": [
            {
                "problem_id": item["problem_id"],
                "task": "task-a" if item["problem_id"] == "fixture-0" else "task-b",
                "correct": item["problem_id"] == "fixture-0",
                "unparsed": False,
                "confidence": 1.0 if item["problem_id"] == "fixture-0" else 0.0,
                "reasoning": "graded in fixture Python worker",
                "extracted_answer": item["response"],
            }
            for item in request["items"]
        ]}
    elif op == "shutdown":
        result = {"shutdown": True}
    else:
        raise RuntimeError(op)
    print(json.dumps({"id": request["id"], "ok": True, "result": result}), flush=True)
    if op == "shutdown":
        break
"#,
    )
    .unwrap();
    let python = Command::new("python3")
        .args(["-c", "import sys; print(sys.executable)"])
        .output()
        .unwrap();
    assert!(python.status.success());
    let python = String::from_utf8(python.stdout).unwrap().trim().to_string();
    assert!(std::path::Path::new(&python).is_absolute());

    let artifacts = tempfile::tempdir().unwrap();
    let request = serde_json::json!({
        "protocol_version": 1,
        "run": {
            "benchmark_id": "accuracy-stdio-e2e",
            "random_seed": 17,
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
                "entries": 1,
                "prompts": {"isl": {"value": 1.0}, "osl": {"value": 1.0}}
            },
            "phases": [{
                "type": "concurrency",
                "name": "profiling",
                "exclude_from_results": false,
                "requests": 4,
                "concurrency": 2
            }],
            "accuracy": {
                "benchmark": "fixture-benchmark",
                "grader": "exact_match",
                "python_executable": python,
                "worker_module": "fixture_accuracy_worker"
            }
        }
    });
    let bytes = serde_json::to_vec(&request).unwrap();
    let binary = env!("CARGO_BIN_EXE_aiperf-runner").to_string();
    let python_path = worker_dir.path().to_path_buf();
    let output = tokio::task::spawn_blocking(move || {
        let mut child = Command::new(binary)
            .env("PYTHONPATH", python_path)
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
    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(artifacts.path().join("native-v2.json")).unwrap())
            .unwrap();
    assert_eq!(report["run"]["mode"], "accuracy");
    assert_eq!(
        report["metrics"]["request_count"]["series"][0]["stats"]["total"],
        4.0
    );
    assert_eq!(report["accuracy"]["summary"]["overall"]["n"], 4);
    assert_eq!(report["accuracy"]["summary"]["overall"]["correct_count"], 2);
    assert_eq!(report["accuracy"]["summary"]["overall"]["accuracy"], 0.5);
    assert_eq!(report["accuracy_records"].as_array().unwrap().len(), 4);
    let correlations = report["accuracy_records"]
        .as_array()
        .unwrap()
        .iter()
        .map(|record| record["correlation_id"].as_str().unwrap())
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(correlations.len(), 4);
    assert_eq!(report["evaluator"]["grader"], "exact_match");
    assert_eq!(
        report["evaluator"]["dataset"]["revision"],
        "fixture-revision"
    );
}
