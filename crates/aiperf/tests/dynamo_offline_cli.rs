// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "dynamo-offline")]

use std::process::Command;

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_aiperf")
}

#[test]
fn feature_build_exposes_offline_cli_options() {
    let output = Command::new(binary())
        .args([
            "--offline",
            "--offline-topology",
            "disaggregated",
            "--offline-prefill-workers",
            "1",
            "--offline-decode-workers",
            "1",
            "--offline-router",
            "kv",
            "--requests",
            "2",
            "--concurrency",
            "1",
            "--isl",
            "8",
            "--osl",
            "2",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn separate_disaggregated_profiles_and_full_router_json_are_consumed() {
    let output = Command::new(binary())
        .args([
            "--offline",
            "--offline-topology",
            "disaggregated",
            "--offline-router",
            "kv",
            "--prefill-engine-args",
            r#"{"worker_type":"prefill","block_size":32,"num_gpu_blocks":128}"#,
            "--decode-engine-args",
            r#"{"worker_type":"decode","block_size":32,"num_gpu_blocks":128}"#,
            "--router-config",
            r#"{"overlap_score_credit":0.25,"prefill_load_scale":0.75,"router_queue_threshold":0.5,"router_ttl_secs":42.0}"#,
            "--requests",
            "3",
            "--concurrency",
            "2",
            "--isl",
            "32",
            "--osl",
            "2",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("3 completed"), "stdout: {stdout}");
}

#[test]
fn every_dynamo_eviction_backend_runs_through_the_aiperf_frontend() {
    for backend in ["lineage", "lru", "multi_lru"] {
        let engine_args =
            format!(r#"{{"block_size":4,"num_gpu_blocks":6,"eviction_backend":"{backend}"}}"#);
        let output = Command::new(binary())
            .args([
                "--offline",
                "--extra-engine-args",
                &engine_args,
                "--requests",
                "2",
                "--concurrency",
                "1",
                "--isl",
                "8",
                "--osl",
                "2",
            ])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "backend={backend}, stderr: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains("2 completed"),
            "backend={backend}, stdout: {stdout}"
        );
    }
}

#[test]
fn g2_g3_g4_offload_is_feature_gated_and_runs_in_every_topology() {
    let engine_args = r#"{"num_gpu_blocks":4,"block_size":4,"max_num_batched_tokens":16,"max_num_seqs":2,"num_g2_blocks":8,"num_g3_blocks":8,"enable_g4_storage":true,"kv_bytes_per_token":1}"#;
    for topology in ["single", "aggregated", "disaggregated"] {
        let output = Command::new(binary())
            .args([
                "--offline",
                "--offline-topology",
                topology,
                "--extra-engine-args",
                engine_args,
                "--requests",
                "2",
                "--concurrency",
                "1",
                "--isl",
                "8",
                "--osl",
                "2",
            ])
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&output.stderr);
        #[cfg(feature = "dynamo-kvbm-offload")]
        {
            assert!(output.status.success(), "topology={topology}: {stderr}");
            assert!(
                !stderr.contains("offload offline init failed"),
                "requested offload must never degrade silently: {stderr}"
            );
        }
        #[cfg(not(feature = "dynamo-kvbm-offload"))]
        {
            assert!(!output.status.success(), "topology={topology}");
            assert!(
                stderr.contains("requires the AIPerf `dynamo-kvbm-offload` feature"),
                "stderr: {stderr}"
            );
        }
    }
}

fn write_jsonl(path: &std::path::Path, rows: &[serde_json::Value]) {
    let payload = rows
        .iter()
        .map(serde_json::Value::to_string)
        .collect::<Vec<_>>()
        .join("\n")
        + "\n";
    std::fs::write(path, payload).unwrap();
}

#[test]
fn every_canonical_trace_format_runs_natively_through_aiperf() {
    let unique = std::process::id();
    let cases = [
        (
            "mooncake",
            vec![
                serde_json::json!({"session_id":"s","timestamp":0.0,"input_length":128,"output_length":2,"hash_ids":(1..=32).collect::<Vec<_>>()}),
                serde_json::json!({"session_id":"s","delay":1.0,"input_length":128,"output_length":2,"hash_ids":(1..=32).collect::<Vec<_>>()}),
            ],
            2,
            false,
        ),
        (
            "mooncake-delta",
            vec![
                serde_json::json!({"session_id":"s","timestamp":0.0,"input_length":4,"output_length":2,"hash_ids":[1]}),
                serde_json::json!({"session_id":"s","delay":1.0,"input_length":4,"output_length":2,"hash_ids":[2]}),
            ],
            2,
            false,
        ),
        (
            "agentic_mooncake",
            vec![
                serde_json::json!({"request_id":"r1","session_id":"root","timestamp":0.0,"input_length":4,"output_length":1,"hash_ids":[1]}),
                serde_json::json!({"request_id":"r2","session_id":"root","wait_for":["r1"],"delay":1.0,"input_length":4,"output_length":1,"hash_ids":[1]}),
            ],
            2,
            false,
        ),
        (
            "applied_compute_agentic",
            vec![serde_json::json!({
                "num_turns":1,
                "input_prompt_length":4,
                "assistant_response_length":[1],
                "tool_call_output_length":[1],
                "tool_call_latency":[0.001],
                "final_assistant_response_length":1
            })],
            2,
            true,
        ),
        (
            "dynamo",
            vec![serde_json::json!({
                "schema":"dynamo.request.trace.v1",
                "event_type":"request_end",
                "event_time_unix_ms":1100,
                "request":{
                    "request_id":"native-1",
                    "request_received_ms":1000,
                    "output_tokens":2,
                    "replay":{
                        "trace_block_size":4,
                        "input_length":4,
                        "input_sequence_hashes":[11]
                    }
                }
            })],
            1,
            false,
        ),
    ];

    for (format, rows, expected, needs_concurrency) in cases {
        let trace_path =
            std::env::temp_dir().join(format!("aiperf-native-trace-{unique}-{format}.jsonl"));
        let report_path =
            std::env::temp_dir().join(format!("aiperf-native-trace-report-{unique}-{format}.json"));
        let artifacts_path = std::env::temp_dir().join(format!(
            "aiperf-native-trace-artifacts-{unique}-{format}.json"
        ));
        let pass_start_artifacts_path = std::env::temp_dir().join(format!(
            "aiperf-native-trace-artifacts-pass-start-{unique}-{format}.json"
        ));
        write_jsonl(&trace_path, &rows);
        let mut command = Command::new(binary());
        command
            .arg("--offline")
            .arg("--trace-file")
            .arg(&trace_path)
            .args(["--trace-format", format, "--trace-block-size", "4"])
            .arg("--report-json")
            .arg(&report_path);
        if needs_concurrency {
            command.args(["--replay-concurrency", "1"]);
        }
        if format == "mooncake" {
            command
                .arg("--worker-artifacts-json")
                .arg(&artifacts_path)
                .args(["--kv-event-visibility", "pass-end"]);
        }
        let output = command.output().unwrap();
        assert!(
            output.status.success(),
            "format={format}, stderr: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let report: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&report_path).unwrap()).unwrap();
        assert_eq!(
            report["completed_requests"], expected,
            "format={format}, report={report}"
        );
        if format == "mooncake" {
            let artifacts: serde_json::Value =
                serde_json::from_slice(&std::fs::read(&artifacts_path).unwrap()).unwrap();
            assert_eq!(artifacts["timed_requests"].as_array().unwrap().len(), 2);
            assert!(
                !artifacts["timed_output_signals"]
                    .as_array()
                    .unwrap()
                    .is_empty()
            );
            assert!(!artifacts["timed_kv_events"].as_array().unwrap().is_empty());

            let pass_start = Command::new(binary())
                .arg("--offline")
                .arg("--trace-file")
                .arg(&trace_path)
                .args([
                    "--trace-format",
                    format,
                    "--trace-block-size",
                    "4",
                    "--kv-event-visibility",
                    "pass-start",
                    "--worker-artifacts-json",
                ])
                .arg(&pass_start_artifacts_path)
                .output()
                .unwrap();
            assert!(
                pass_start.status.success(),
                "pass-start stderr: {}",
                String::from_utf8_lossy(&pass_start.stderr)
            );
            let pass_start_artifacts: serde_json::Value =
                serde_json::from_slice(&std::fs::read(&pass_start_artifacts_path).unwrap())
                    .unwrap();
            assert!(
                !pass_start_artifacts["timed_kv_events"]
                    .as_array()
                    .unwrap()
                    .is_empty()
            );
            std::fs::remove_file(&artifacts_path).unwrap();
            std::fs::remove_file(&pass_start_artifacts_path).unwrap();
        }
        std::fs::remove_file(trace_path).unwrap();
        std::fs::remove_file(report_path).unwrap();
    }
}

#[test]
fn canonical_trace_runs_through_every_topology_and_router_mode() {
    let unique = std::process::id();
    let trace_path =
        std::env::temp_dir().join(format!("aiperf-native-trace-topology-{unique}.jsonl"));
    write_jsonl(
        &trace_path,
        &[
            serde_json::json!({"session_id":"a","timestamp":0.0,"input_length":8,"output_length":2,"hash_ids":[1,2]}),
            serde_json::json!({"session_id":"b","timestamp":0.0,"input_length":8,"output_length":2,"hash_ids":[3,4]}),
        ],
    );
    let cases = [
        ("single", "round-robin"),
        ("aggregated", "round-robin"),
        ("aggregated", "kv"),
        ("disaggregated", "round-robin"),
        ("disaggregated", "kv"),
    ];
    for (topology, router) in cases {
        let report_path = std::env::temp_dir().join(format!(
            "aiperf-native-trace-topology-{unique}-{topology}-{router}.json"
        ));
        let output = Command::new(binary())
            .arg("--offline")
            .args(["--offline-topology", topology, "--offline-router", router])
            .args([
                "--offline-workers",
                "2",
                "--offline-prefill-workers",
                "2",
                "--offline-decode-workers",
                "2",
            ])
            .arg("--trace-file")
            .arg(&trace_path)
            .args([
                "--trace-format",
                "mooncake",
                "--trace-block-size",
                "4",
                "--report-json",
            ])
            .arg(&report_path)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "topology={topology}, router={router}, stderr: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let report: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&report_path).unwrap()).unwrap();
        assert_eq!(report["completed_requests"], 2);
        std::fs::remove_file(report_path).unwrap();
    }
    std::fs::remove_file(trace_path).unwrap();
}

#[test]
fn native_trace_max_sim_time_stops_before_future_arrivals() {
    let unique = std::process::id();
    let trace_path = std::env::temp_dir().join(format!("aiperf-native-trace-cap-{unique}.jsonl"));
    let report_path =
        std::env::temp_dir().join(format!("aiperf-native-trace-cap-report-{unique}.json"));
    write_jsonl(
        &trace_path,
        &[
            serde_json::json!({"timestamp":0.0,"input_length":4,"output_length":1,"hash_ids":[1]}),
            serde_json::json!({"timestamp":1000.0,"input_length":4,"output_length":1,"hash_ids":[2]}),
        ],
    );
    let output = Command::new(binary())
        .arg("--offline")
        .arg("--trace-file")
        .arg(&trace_path)
        .args([
            "--trace-format",
            "mooncake",
            "--trace-block-size",
            "4",
            "--max-sim-time-seconds",
            "0.5",
            "--report-json",
        ])
        .arg(&report_path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&report_path).unwrap()).unwrap();
    assert_eq!(report["num_requests"], 1);
    assert_eq!(report["completed_requests"], 1);
    assert!(report["duration_ms"].as_f64().unwrap() <= 500.0);
    std::fs::remove_file(trace_path).unwrap();
    std::fs::remove_file(report_path).unwrap();
}

#[test]
fn native_trace_max_sim_time_leaves_inflight_requests_incomplete() {
    let unique = std::process::id();
    let trace_path =
        std::env::temp_dir().join(format!("aiperf-native-trace-inflight-cap-{unique}.jsonl"));
    let report_path = std::env::temp_dir().join(format!(
        "aiperf-native-trace-inflight-cap-report-{unique}.json"
    ));
    let records_path = std::env::temp_dir().join(format!(
        "aiperf-native-trace-inflight-cap-records-{unique}.jsonl"
    ));
    write_jsonl(
        &trace_path,
        &[serde_json::json!({
            "timestamp":0.0,
            "input_length":128,
            "output_length":128,
            "hash_ids":[1,2]
        })],
    );
    let output = Command::new(binary())
        .arg("--offline")
        .arg("--trace-file")
        .arg(&trace_path)
        .args([
            "--trace-format",
            "mooncake",
            "--trace-block-size",
            "64",
            "--max-sim-time-seconds",
            "0.000000001",
            "--report-json",
        ])
        .arg(&report_path)
        .arg("--report-jsonl")
        .arg(&records_path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&report_path).unwrap()).unwrap();
    assert_eq!(report["num_requests"], 1);
    assert_eq!(report["completed_requests"], 0);
    assert!(std::fs::read_to_string(&records_path).unwrap().is_empty());
    std::fs::remove_file(trace_path).unwrap();
    std::fs::remove_file(report_path).unwrap();
    std::fs::remove_file(records_path).unwrap();
}

#[test]
fn sla_goodput_and_per_request_jsonl_cover_every_offline_topology() {
    let unique = std::process::id();
    for topology in ["single", "aggregated", "disaggregated"] {
        let report_path =
            std::env::temp_dir().join(format!("aiperf-dynamo-goodput-{unique}-{topology}.json"));
        let records_path =
            std::env::temp_dir().join(format!("aiperf-dynamo-records-{unique}-{topology}.jsonl"));
        let native_path = std::env::temp_dir().join(format!(
            "aiperf-dynamo-native-goodput-{unique}-{topology}.json"
        ));
        let output = Command::new(binary())
            .args([
                "--offline",
                "--offline-topology",
                topology,
                "--requests",
                "3",
                "--concurrency",
                "2",
                "--isl",
                "8",
                "--osl",
                "3",
                "--sla-e2e-ms",
                "1000000000",
                "--report-json",
            ])
            .arg(&report_path)
            .arg("--report-jsonl")
            .arg(&records_path)
            .arg("--json")
            .arg(&native_path)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "topology={topology}, stderr: {}",
            String::from_utf8_lossy(&output.stderr)
        );

        let aggregate: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&report_path).unwrap()).unwrap();
        assert_eq!(aggregate["goodput_completed_requests"], 3);
        assert!(
            aggregate["goodput_request_throughput_rps"]
                .as_f64()
                .unwrap()
                > 0.0
        );

        let records = std::fs::read_to_string(&records_path).unwrap();
        let decoded = records
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(decoded.len(), 3, "topology={topology}, records={records}");
        assert!(
            decoded
                .iter()
                .all(|record| record["terminal_status"] == "completed")
        );
        let native: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&native_path).unwrap()).unwrap();
        assert!(native["metrics"]["goodput"].is_object());
        assert!(native["metrics"]["good_request_count"].is_object());

        std::fs::remove_file(report_path).unwrap();
        std::fs::remove_file(records_path).unwrap();
        std::fs::remove_file(native_path).unwrap();
    }
}

#[test]
fn offline_sla_rejects_non_finite_or_negative_thresholds() {
    for value in ["-1", "NaN", "inf"] {
        let output = Command::new(binary())
            .args(["--offline", "--requests", "1", "--sla-e2e-ms", value])
            .output()
            .unwrap();
        assert!(!output.status.success(), "value={value}");
    }
}

#[test]
fn disaggregated_profile_pair_is_transactional() {
    let output = Command::new(binary())
        .args([
            "--offline",
            "--offline-topology",
            "disaggregated",
            "--prefill-engine-args",
            r#"{"worker_type":"prefill"}"#,
            "--requests",
            "1",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .contains("requires both --prefill-engine-args and --decode-engine-args")
    );
}

#[test]
fn router_policy_path_overrides_the_inline_router_json() {
    let path = std::env::temp_dir().join(format!(
        "aiperf-dynamo-router-policy-{}.yaml",
        std::process::id()
    ));
    std::fs::write(
        &path,
        r#"
default_policy_family: root
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: root
    policy_family: root
    cache_bucket: all
    quantum: 1
"#,
    )
    .unwrap();
    let output = Command::new(binary())
        .args([
            "--offline",
            "--offline-topology",
            "aggregated",
            "--offline-router",
            "kv",
            "--router-config",
            r#"{"router_policy_config":"/definitely/missing.yaml"}"#,
            "--router-policy-config",
        ])
        .arg(&path)
        .args(["--requests", "2", "--concurrency", "1"])
        .output()
        .unwrap();
    std::fs::remove_file(path).unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn aic_cli_is_complete_and_feature_gated() {
    let missing = Command::new(binary())
        .args(["--offline", "--aic-backend", "vllm", "--requests", "1"])
        .output()
        .unwrap();
    assert!(!missing.status.success());
    assert!(
        String::from_utf8_lossy(&missing.stderr)
            .contains("AIC replay modeling requires --aic-system")
    );

    let gated = Command::new(binary())
        .args([
            "--offline",
            "--offline-topology",
            "aggregated",
            "--offline-router",
            "kv",
            "--router-config",
            r#"{"router_prefill_load_model":"aic"}"#,
            "--aic-backend",
            "vllm",
            "--aic-system",
            "h200_sxm",
            "--aic-model-path",
            "Qwen/Qwen3-0.6B",
            "--requests",
            "1",
        ])
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&gated.stderr);
    #[cfg(not(feature = "dynamo-aic-forward-pass"))]
    {
        assert!(!gated.status.success());
        assert!(
            stderr.contains("requires the AIPerf `dynamo-aic-forward-pass` feature"),
            "stderr: {stderr}"
        );
    }
    #[cfg(feature = "dynamo-aic-forward-pass")]
    {
        assert!(
            !stderr.contains("requires the AIPerf `dynamo-aic-forward-pass` feature"),
            "the enabled feature must enter the AIC runtime: {stderr}"
        );
        if !gated.status.success() {
            assert!(
                stderr.contains("AIC") || stderr.contains("aiconfigurator"),
                "an unavailable external AIC data/package must fail actionably: {stderr}"
            );
        }
    }
}

#[test]
fn closed_loop_cli_runs_without_an_http_server() {
    let output = Command::new(binary())
        .args([
            "--offline",
            "--requests",
            "6",
            "--concurrency",
            "3",
            "--isl",
            "16",
            "--osl",
            "4",
        ])
        .output()
        .expect("run feature-gated offline CLI");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stdout.contains("6 completed"), "stdout: {stdout}");
    assert!(
        stderr.contains("verified byte-exact AIPerf/Dynamo shared metrics"),
        "stderr: {stderr}"
    );
}

#[test]
fn graph_cli_runs_multi_turn_cosimulation_without_a_url() {
    let output = Command::new(binary())
        .args([
            "--offline",
            "--mode",
            "graph",
            "--offline-topology",
            "aggregated",
            "--offline-workers",
            "2",
            "--offline-router",
            "kv",
            "--turns",
            "2",
            "--instances",
            "4",
            "--workers",
            "1",
            "--concurrency",
            "2",
            "--osl",
            "3",
        ])
        .output()
        .expect("run feature-gated offline graph CLI");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("backend=dynamo-offline"),
        "stdout: {stdout}"
    );
    assert!(stdout.contains("completed=8"), "stdout: {stdout}");
}

#[test]
fn fixed_schedule_cli_interleaves_authored_time_and_engine_events() {
    let path = std::env::temp_dir().join(format!(
        "aiperf-dynamo-offline-fixed-{}.json",
        std::process::id()
    ));
    std::fs::write(
        &path,
        serde_json::to_vec(&serde_json::json!({
            "session_id": "fixed-proof",
            "turns": [
                {"text": "first", "timestamp": 0, "output_length": 2},
                {"text": "second", "delay": 1, "output_length": 3}
            ]
        }))
        .unwrap(),
    )
    .unwrap();

    let output = Command::new(binary())
        .arg("--offline")
        .args(["--offline-topology", "disaggregated"])
        .arg("--fixed-schedule")
        .arg("--input-file")
        .arg(&path)
        .arg("--input-format")
        .arg("multi_turn")
        .arg("--osl")
        .arg("2")
        .output()
        .expect("run feature-gated offline fixed schedule");
    std::fs::remove_file(&path).unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("2 completed"), "stdout: {stdout}");
}

#[test]
fn request_rate_cli_runs_multi_turn_continuations_offline() {
    let output = Command::new(binary())
        .args([
            "--offline",
            "--request-rate",
            "1000",
            "--arrival",
            "constant",
            "--turns",
            "2",
            "--requests",
            "8",
            "--isl",
            "8",
            "--osl",
            "2",
        ])
        .output()
        .expect("run feature-gated offline request-rate workload");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("8 completed"), "stdout: {stdout}");
}

#[test]
fn user_centric_cli_runs_virtual_history_and_continuations_offline() {
    let output = Command::new(binary())
        .args([
            "--offline",
            "--user-centric-rate",
            "100",
            "--num-users",
            "2",
            "--turns",
            "2",
            "--requests",
            "6",
            "--isl",
            "8",
            "--osl",
            "2",
        ])
        .output()
        .expect("run feature-gated offline user-centric workload");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("6 completed"), "stdout: {stdout}");
}

#[test]
fn request_rate_cli_runs_all_clock_native_ramps_offline() {
    let output = Command::new(binary())
        .args([
            "--offline",
            "--request-rate",
            "1000",
            "--arrival",
            "constant",
            "--concurrency",
            "2",
            "--prefill-concurrency",
            "2",
            "--concurrency-ramp-duration",
            "0.01",
            "--prefill-concurrency-ramp-duration",
            "0.01",
            "--request-rate-ramp-duration",
            "0.01",
            "--turns",
            "2",
            "--requests",
            "8",
            "--isl",
            "8",
            "--osl",
            "2",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("8 completed"));
}

#[test]
fn user_centric_cli_runs_session_concurrency_ramp_offline() {
    let output = Command::new(binary())
        .args([
            "--offline",
            "--user-centric-rate",
            "100",
            "--num-users",
            "2",
            "--concurrency",
            "2",
            "--concurrency-ramp-duration",
            "0.01",
            "--turns",
            "2",
            "--requests",
            "6",
            "--isl",
            "8",
            "--osl",
            "2",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("6 completed"));
}

#[test]
fn adaptive_concurrency_consumes_incremental_engine_completions() {
    let artifact_dir = std::env::temp_dir().join(format!(
        "aiperf-dynamo-offline-adaptive-{}",
        std::process::id()
    ));
    let output = Command::new(binary())
        .arg("--offline")
        .args([
            "--duration",
            "3",
            "--concurrency",
            "2",
            "--isl",
            "8",
            "--osl",
            "2",
            "--adaptive-scale",
            "--adaptive-control-min",
            "1",
            "--adaptive-control-max",
            "2",
            "--adaptive-assessment-period",
            "1",
            "--adaptive-sustain-duration",
            "1",
            "--adaptive-scale-sla",
            "request_latency:p95:le:0",
            "--adaptive-base-step",
            "1",
            "--adaptive-max-step-multiplier",
            "1",
            "--adaptive-artifact-dir",
        ])
        .arg(&artifact_dir)
        .output()
        .expect("run adaptive control against incremental offline events");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let events = std::fs::read_to_string(artifact_dir.join("adaptive_scale_events.jsonl")).unwrap();
    assert!(events.contains("\"adaptive_window\""), "events: {events}");
    assert!(events.contains("\"adaptive_failed\""), "events: {events}");
    std::fs::remove_dir_all(artifact_dir).unwrap();
}

#[test]
fn adaptive_request_rate_runs_on_the_shared_offline_control_path() {
    let artifact_dir = std::env::temp_dir().join(format!(
        "aiperf-dynamo-offline-adaptive-rate-{}",
        std::process::id()
    ));
    let output = Command::new(binary())
        .args([
            "--offline",
            "--request-rate",
            "10",
            "--arrival",
            "constant",
            "--turns",
            "2",
            "--duration",
            "3",
            "--concurrency",
            "2",
            "--isl",
            "8",
            "--osl",
            "2",
            "--adaptive-scale",
            "--adaptive-control-variable",
            "request_rate",
            "--adaptive-control-min",
            "1",
            "--adaptive-control-max",
            "2",
            "--adaptive-assessment-period",
            "1",
            "--adaptive-sustain-duration",
            "1",
            "--adaptive-scale-sla",
            "request_latency:p95:le:0",
            "--adaptive-base-step",
            "1",
            "--adaptive-max-step-multiplier",
            "1",
            "--adaptive-artifact-dir",
        ])
        .arg(&artifact_dir)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let events = std::fs::read_to_string(artifact_dir.join("adaptive_scale_events.jsonl")).unwrap();
    assert!(events.contains("\"adaptive_window\""), "events: {events}");
    assert!(events.contains("\"adaptive_failed\""), "events: {events}");
    std::fs::remove_dir_all(artifact_dir).unwrap();
}

#[test]
fn adaptive_users_runs_on_the_shared_offline_control_path() {
    let artifact_dir = std::env::temp_dir().join(format!(
        "aiperf-dynamo-offline-adaptive-users-{}",
        std::process::id()
    ));
    let output = Command::new(binary())
        .args([
            "--offline",
            "--user-centric-rate",
            "4",
            "--num-users",
            "2",
            "--turns",
            "2",
            "--duration",
            "3",
            "--isl",
            "8",
            "--osl",
            "2",
            "--adaptive-scale",
            "--adaptive-control-variable",
            "users",
            "--adaptive-control-min",
            "1",
            "--adaptive-control-max",
            "2",
            "--adaptive-assessment-period",
            "1",
            "--adaptive-sustain-duration",
            "1",
            "--adaptive-scale-sla",
            "request_latency:p95:le:0",
            "--adaptive-base-step",
            "1",
            "--adaptive-max-step-multiplier",
            "1",
            "--adaptive-artifact-dir",
        ])
        .arg(&artifact_dir)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let events = std::fs::read_to_string(artifact_dir.join("adaptive_scale_events.jsonl")).unwrap();
    assert!(events.contains("\"adaptive_window\""), "events: {events}");
    assert!(events.contains("\"adaptive_failed\""), "events: {events}");
    std::fs::remove_dir_all(artifact_dir).unwrap();
}

#[test]
fn offline_cancellation_runs_through_the_disaggregated_kv_engine() {
    let output = Command::new(binary())
        .args([
            "--offline",
            "--offline-topology",
            "disaggregated",
            "--offline-router",
            "kv",
            "--requests",
            "4",
            "--concurrency",
            "2",
            "--request-cancellation-rate",
            "100",
            "--request-cancellation-delay",
            "0.000000001",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stdout.contains("0 completed"), "stdout: {stdout}");
    assert!(
        stderr.contains("verified byte-exact AIPerf/Dynamo shared metrics"),
        "stderr: {stderr}"
    );
}

#[test]
fn unsupported_accuracy_fails_instead_of_falling_back_online() {
    let expected =
        "--accuracy-benchmark requires model-generated text and is unavailable with --offline";
    let output = Command::new(binary())
        .args(["--offline", "--accuracy-benchmark", "mmlu-pro"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains(expected),
        "expected {expected:?} in {stderr}"
    );
}

#[test]
fn engine_profile_is_loaded_and_native_json_is_reproducible() {
    let unique = std::process::id();
    let profile = std::env::temp_dir().join(format!("aiperf-dynamo-profile-{unique}.json"));
    let first_json = std::env::temp_dir().join(format!("aiperf-dynamo-first-{unique}.json"));
    let second_json = std::env::temp_dir().join(format!("aiperf-dynamo-second-{unique}.json"));
    std::fs::write(&profile, b"{}").unwrap();

    for output_path in [&first_json, &second_json] {
        let output = Command::new(binary())
            .arg("--offline")
            .arg("--engine-profile")
            .arg(&profile)
            .args([
                "--requests",
                "6",
                "--concurrency",
                "3",
                "--isl",
                "16",
                "--osl",
                "4",
                "--json",
            ])
            .arg(output_path)
            .output()
            .expect("run profiled offline CLI");
        assert!(
            output.status.success(),
            "stderr: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let first = std::fs::read(&first_json).unwrap();
    let second = std::fs::read(&second_json).unwrap();
    let report: serde_json::Value = serde_json::from_slice(&first).unwrap();
    std::fs::remove_file(profile).unwrap();
    std::fs::remove_file(first_json).unwrap();
    std::fs::remove_file(second_json).unwrap();
    assert_eq!(first, second, "offline native-v2 JSON must be byte-stable");
    assert_eq!(report["schema_version"], "2.0");
    assert_eq!(report["run"]["mode"], "offline");
    assert!(
        report.get("dynamo").is_none(),
        "normal report schema changed"
    );
}
