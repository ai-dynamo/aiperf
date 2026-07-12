// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process proofs for the protocol-v2 Dynamo scheduled adapter.

#![cfg(feature = "dynamo-offline")]

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

use serde_json::{Value, json};

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_aiperf-runner")
}

fn run(request: &Value) -> Output {
    let mut child = Command::new(binary())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(serde_json::to_string(request).unwrap().as_bytes())
        .unwrap();
    child.wait_with_output().unwrap()
}

fn one_line(output: &Output) -> Value {
    let lines = output
        .stdout
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>();
    assert_eq!(
        lines.len(),
        1,
        "stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(lines[0]).unwrap()
}

fn distribution_id() -> String {
    let output = Command::new(binary())
        .arg("--capabilities")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "capabilities stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let capabilities = one_line(&output);
    assert!(
        capabilities["supported_pairs"]
            .as_array()
            .unwrap()
            .contains(&json!(["dynamo_offline", "scheduled"]))
    );
    capabilities["distribution_id"].as_str().unwrap().to_owned()
}

fn synthetic_dataset() -> Value {
    json!({
        "type": "synthetic",
        "entries": 4,
        "sampling": "sequential",
        "prompts": {
            "isl": {"value": 8.0},
            "osl": {"value": 2.0}
        }
    })
}

fn user_dataset() -> Value {
    let mut dataset = synthetic_dataset();
    dataset["turns"] = json!({"value": 2.0});
    dataset
}

fn fixed_dataset() -> Value {
    json!({
        "type": "file",
        "format": "mooncake_trace",
        "sampling": "sequential",
        "osl": {"value": 2.0},
        "records": [
            {
                "session_id": "fixed-a",
                "timestamp": 100,
                "input_length": 8,
                "output_length": 2,
                "hash_ids": [1]
            },
            {
                "session_id": "fixed-b",
                "timestamp": 110,
                "input_length": 8,
                "output_length": 2,
                "hash_ids": [2]
            }
        ]
    })
}

fn envelope(
    distribution_id: &str,
    benchmark_id: &str,
    artifact_target: &Path,
    dataset: Value,
    phases: Value,
) -> Value {
    json!({
        "protocol_version": 2,
        "operation": "execute",
        "expected_distribution_id": distribution_id,
        "run": {
            "identity": {
                "benchmark_id": benchmark_id,
                "random_seed": 41
            },
            "artifact_target": artifact_target,
            "backend": {
                "type": "dynamo_offline",
                "config": {
                    "sla": {"e2e_ms": 1000.0},
                    "artifacts": {
                        "report_json": "dynamo/report.json",
                        "per_request_jsonl": "dynamo/requests.jsonl"
                    }
                }
            },
            "workload": {
                "type": "scheduled",
                "config": {
                    "worker_count": 1,
                    "dataset": dataset,
                    "tokenizer": {
                        "name": "builtin",
                        "revision": "main",
                        "trust_remote_code": false,
                        "apply_chat_template": false
                    },
                    "phases": phases
                }
            },
            "resources": {
                "models": {"items": [{"name": "mock-model"}]},
                "endpoints": {"profiles": [{
                    "id": "default",
                    "type": "chat_completions",
                    "urls": ["http://127.0.0.1:9"]
                }]},
                "metrics": {
                    "slice_duration_seconds": 0.1,
                    "slos": {"request_latency": 1000.0}
                },
                "artifacts": {},
                "sidecars": {}
            }
        }
    })
}

fn target(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "aiperf-runner-offline-scheduled-{}-{name}",
        std::process::id()
    ))
}

fn assert_success(
    output: Output,
    target: &Path,
    distribution_id: &str,
    workload: &str,
    independently_accumulated_fields: u64,
) -> Value {
    let terminal = one_line(&output);
    assert!(
        output.status.success(),
        "terminal={terminal}, stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(terminal["success"], true);
    assert_eq!(terminal["provenance"]["backend"], "dynamo_offline");
    assert_eq!(terminal["provenance"]["workload"], workload);
    assert_eq!(terminal["provenance"]["parity_shared_fields"], "77");
    assert_eq!(
        PathBuf::from(terminal["report_path"].as_str().unwrap()),
        target.join("native-v2.json")
    );
    let native: Value =
        serde_json::from_slice(&std::fs::read(target.join("native-v2.json")).unwrap()).unwrap();
    assert_eq!(native["schema_version"], "2.0");
    assert_eq!(native["run"]["distribution_id"], distribution_id);
    assert_eq!(native["run"]["backend"], "dynamo_offline");
    assert_eq!(native["run"]["workload"], workload);
    assert_eq!(native["run"]["extensions"], json!([]));
    assert_eq!(
        native["run"]["endpoint_profiles"],
        json!([{"profile_id": "default", "endpoint_id": "chat"}])
    );
    assert_eq!(native["run"]["mode"], "offline:scheduled");
    assert_eq!(native["run"]["dynamo"]["clock"], "sim");
    assert_eq!(native["run"]["dynamo"]["parity"]["shared_fields"], 77);
    assert_eq!(
        native["run"]["dynamo"]["parity"]["independently_accumulated_fields"],
        independently_accumulated_fields
    );
    assert_eq!(
        native["run"]["dynamo"]["parity"]["backend_owned_fields"],
        77 - independently_accumulated_fields
    );
    assert!(native["run"]["dynamo"]["capacity"].is_object());
    assert!(target.join("dynamo/report.json").is_file());
    terminal
}

#[test]
fn warmup_and_profiling_share_one_engine_clock_and_exact_live_parity_collector() {
    let distribution_id = distribution_id();
    let target = target("warmup-profiling");
    let _ = std::fs::remove_dir_all(&target);
    let mut request = envelope(
        &distribution_id,
        "offline-scheduled-warmup",
        &target,
        synthetic_dataset(),
        json!([
            {
                "type": "concurrency",
                "name": "warmup",
                "exclude_from_results": true,
                "requests": 2,
                "concurrency": 1
            },
            {
                "type": "constant",
                "name": "profiling",
                "exclude_from_results": false,
                "seamless": true,
                "requests": 4,
                "rate": 100.0,
                "concurrency": 2,
                "prefill_concurrency": 2,
                "rate_ramp": {"duration": 0.2, "strategy": "exponential"},
                "cancellation": {"rate": 25.0, "delay": 0.001}
            }
        ]),
    );
    request["run"]["backend"]["config"]["topology"] = json!("aggregated");
    request["run"]["backend"]["config"]["workers"] = json!(2);
    request["run"]["backend"]["config"]["router_mode"] = json!("kv");
    let mut validation_request = request.clone();
    validation_request["operation"] = json!("validate");
    let validation = run(&validation_request);
    let validation_terminal = one_line(&validation);
    assert!(validation.status.success(), "{validation_terminal}");
    assert_eq!(validation_terminal["event"], "run_validation");
    assert_eq!(validation_terminal["success"], true);
    assert!(!target.exists(), "scheduled validation created artifacts");

    let terminal = assert_success(run(&request), &target, &distribution_id, "scheduled", 69);
    assert_eq!(terminal["provenance"]["phase_count"], "2");
    assert_eq!(terminal["provenance"]["topology"], "aggregated");
    assert_eq!(terminal["provenance"]["router"], "kv");
    let dynamo: Value =
        serde_json::from_slice(&std::fs::read(target.join("dynamo/report.json")).unwrap()).unwrap();
    assert_eq!(dynamo["num_requests"], 6);
    let native: Value =
        serde_json::from_slice(&std::fs::read(target.join("native-v2.json")).unwrap()).unwrap();
    assert!(native["metrics"]["goodput"].is_object());
    assert_eq!(native["run"]["dynamo"]["topology"], "aggregated");
    assert_eq!(native["run"]["dynamo"]["router"], "kv");
    assert_eq!(native["run"]["dynamo"]["workers"], 2);
    std::fs::remove_dir_all(target).unwrap();
}

#[test]
fn every_scheduled_phase_family_and_ramp_curve_executes_through_the_pair() {
    let distribution_id = distribution_id();
    let cases = [
        (
            "concurrency-linear",
            synthetic_dataset(),
            json!({
                "type": "concurrency",
                "name": "profiling",
                "exclude_from_results": false,
                "requests": 4,
                "concurrency": 2,
                "concurrency_ramp": {"duration": 0.1, "strategy": "linear"}
            }),
            4.0,
        ),
        (
            "poisson-exponential",
            synthetic_dataset(),
            json!({
                "type": "poisson",
                "name": "profiling",
                "exclude_from_results": false,
                "requests": 4,
                "rate": 100.0,
                "concurrency": 2,
                "rate_ramp": {"duration": 0.2, "strategy": "exponential"}
            }),
            4.0,
        ),
        (
            "gamma-cancellation",
            synthetic_dataset(),
            json!({
                "type": "gamma",
                "name": "profiling",
                "exclude_from_results": false,
                "requests": 4,
                "rate": 100.0,
                "smoothness": 2.0,
                "concurrency": 2,
                "cancellation": {"rate": 50.0, "delay": 0.001}
            }),
            1.0,
        ),
        (
            "constant-poisson",
            synthetic_dataset(),
            json!({
                "type": "constant",
                "name": "profiling",
                "exclude_from_results": false,
                "requests": 4,
                "rate": 100.0,
                "concurrency": 2,
                "rate_ramp": {"duration": 0.2, "strategy": "poisson"}
            }),
            4.0,
        ),
        (
            "user-centric",
            user_dataset(),
            json!({
                "type": "user_centric",
                "name": "profiling",
                "exclude_from_results": false,
                "requests": 4,
                "rate": 100.0,
                "users": 2,
                "concurrency": 2,
                "concurrency_ramp": {"duration": 0.1, "strategy": "exponential"}
            }),
            4.0,
        ),
        (
            "fixed-schedule",
            fixed_dataset(),
            json!({
                "type": "fixed_schedule",
                "name": "profiling",
                "exclude_from_results": false,
                "auto_offset": true
            }),
            2.0,
        ),
    ];

    for (name, dataset, phase, expected_request_count) in cases {
        let target = target(name);
        let _ = std::fs::remove_dir_all(&target);
        let request = envelope(
            &distribution_id,
            &format!("offline-scheduled-{name}"),
            &target,
            dataset,
            json!([phase]),
        );
        assert_success(run(&request), &target, &distribution_id, "scheduled", 0);
        let native: Value =
            serde_json::from_slice(&std::fs::read(target.join("native-v2.json")).unwrap()).unwrap();
        assert_eq!(
            native["metrics"]["request_count"]["series"][0]["stats"]["total"],
            expected_request_count,
            "phase family {name} did not report its deterministic expected request count"
        );
        std::fs::remove_dir_all(target).unwrap();
    }
}

fn adaptive(control_variable: &str, minimum: f64, maximum: f64) -> Value {
    json!({
        "control_variable": control_variable,
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
    })
}

#[test]
fn every_adaptive_actuator_executes_and_commits_schema_v2_artifacts() {
    let distribution_id = distribution_id();
    let cases = [
        (
            "concurrency",
            json!({
                "type": "concurrency",
                "name": "profiling",
                "exclude_from_results": false,
                "duration": 4.0,
                "concurrency": 2,
                "adaptive_scale": adaptive("concurrency", 1.0, 2.0)
            }),
        ),
        (
            "prefill-concurrency",
            json!({
                "type": "concurrency",
                "name": "profiling",
                "exclude_from_results": false,
                "duration": 4.0,
                "concurrency": 2,
                "prefill_concurrency": 2,
                "adaptive_scale": adaptive("prefill_concurrency", 1.0, 2.0)
            }),
        ),
        (
            "request-rate",
            json!({
                "type": "constant",
                "name": "profiling",
                "exclude_from_results": false,
                "duration": 4.0,
                "rate": 20.0,
                "concurrency": 2,
                "adaptive_scale": adaptive("request_rate", 10.0, 20.0)
            }),
        ),
        (
            "users",
            json!({
                "type": "user_centric",
                "name": "profiling",
                "exclude_from_results": false,
                "duration": 4.0,
                "rate": 20.0,
                "users": 1,
                "concurrency": 2,
                "adaptive_scale": adaptive("users", 1.0, 2.0)
            }),
        ),
    ];

    for (name, phase) in cases {
        let target = target(&format!("adaptive-{name}"));
        let _ = std::fs::remove_dir_all(&target);
        let request = envelope(
            &distribution_id,
            &format!("offline-adaptive-{name}"),
            &target,
            if name == "users" {
                user_dataset()
            } else {
                synthetic_dataset()
            },
            json!([phase]),
        );
        assert_success(run(&request), &target, &distribution_id, "scheduled", 0);
        let summary: Value = serde_json::from_slice(
            &std::fs::read(target.join("adaptive_scale_summary.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(summary["schema_version"], 2);
        assert!(target.join("adaptive_scale_events.jsonl").is_file());
        std::fs::remove_dir_all(target).unwrap();
    }
}
