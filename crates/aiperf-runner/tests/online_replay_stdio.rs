// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process proofs for the wall-clock in-process (`replay_mode=online`) Dynamo
//! adapter: the same protocol-v2 scheduled and graph pairs driven through
//! AIPerf's own flow against the passive Dynamo engine under a real clock — the
//! equivalent of Dynamo's `--replay-mode online`, with no sockets, HTTP, or
//! frontend, and without the mocker's own trace driver.
//!
//! These prove the runner-visible wiring: the authored `online` clock axis
//! selects the wall-clock driver, request/token counts stay exact against the
//! engine's own report, and the terminal/native provenance reflects the real
//! clock. The apples-to-apples gate additionally drives a hash-block trace
//! through the real runner product path and compares its native-v2 report to
//! Dynamo's own wall-clock online driver within 3%.

#![cfg(feature = "dynamo-offline")]

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

use aiperf::dynamo_offline::{NativeBaselineRequest, native_live_concurrency_baseline};
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
    let output = Command::new(binary()).arg("--capabilities").output().unwrap();
    assert!(output.status.success());
    let capabilities = one_line(&output);
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

fn scheduled_envelope(
    distribution_id: &str,
    benchmark_id: &str,
    artifact_target: &Path,
    replay_mode: &str,
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
                    "replay_mode": replay_mode,
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
                    "dataset": synthetic_dataset(),
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
        "aiperf-runner-online-replay-{}-{name}",
        std::process::id()
    ))
}

/// The full wired online scheduled path executes end to end under the real
/// clock, reports `online:scheduled` / `clock=real` provenance, and returns the
/// exact deterministic request count the engine served — proving the wall-clock
/// driver reproduces the native engine's served-request accounting.
#[test]
fn online_scheduled_runs_in_process_under_real_clock_with_exact_counts() {
    let distribution_id = distribution_id();
    let target = target("scheduled");
    let _ = std::fs::remove_dir_all(&target);
    let request = scheduled_envelope(
        &distribution_id,
        "online-scheduled",
        &target,
        "online",
        json!([{
            "type": "constant",
            "name": "profiling",
            "exclude_from_results": false,
            "requests": 4,
            "rate": 200.0,
            "concurrency": 4
        }]),
    );

    let output = run(&request);
    let terminal = one_line(&output);
    assert!(
        output.status.success(),
        "terminal={terminal}, stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(terminal["success"], true);
    assert_eq!(terminal["provenance"]["backend"], "dynamo_offline");
    assert_eq!(terminal["provenance"]["replay_mode"], "online");
    assert_eq!(terminal["provenance"]["clock"], "real");

    let native: Value =
        serde_json::from_slice(&std::fs::read(target.join("native-v2.json")).unwrap()).unwrap();
    assert_eq!(native["run"]["mode"], "online:scheduled");
    assert_eq!(native["run"]["dynamo"]["clock"], "real");
    assert_eq!(
        native["metrics"]["request_count"]["series"][0]["stats"]["total"],
        4.0,
        "online scheduled run did not serve the exact authored request count"
    );

    // The engine's own canonical report (the "native" side) served the same
    // request count that AIPerf accumulated from its observer stream.
    let dynamo: Value =
        serde_json::from_slice(&std::fs::read(target.join("dynamo/report.json")).unwrap()).unwrap();
    assert_eq!(dynamo["num_requests"], 4);
    std::fs::remove_dir_all(target).unwrap();
}

/// The authored `online` clock axis is honored per-run: the same envelope run as
/// `offline` reports the virtual clock and byte-exact provenance, while `online`
/// reports the real clock. Both serve the identical request count.
#[test]
fn offline_and_online_agree_on_served_counts_and_differ_only_on_clock() {
    let distribution_id = distribution_id();
    let phases = json!([{
        "type": "concurrency",
        "name": "profiling",
        "exclude_from_results": false,
        "requests": 4,
        "concurrency": 2
    }]);

    let mut counts = Vec::new();
    for mode in ["offline", "online"] {
        let target = target(mode);
        let _ = std::fs::remove_dir_all(&target);
        let request = scheduled_envelope(
            &distribution_id,
            &format!("agree-{mode}"),
            &target,
            mode,
            phases.clone(),
        );
        let output = run(&request);
        let terminal = one_line(&output);
        assert!(
            output.status.success(),
            "mode={mode} terminal={terminal}, stderr={}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(terminal["provenance"]["replay_mode"], mode);
        let native: Value =
            serde_json::from_slice(&std::fs::read(target.join("native-v2.json")).unwrap()).unwrap();
        assert_eq!(native["run"]["mode"], format!("{mode}:scheduled"));
        assert_eq!(
            native["run"]["dynamo"]["clock"],
            if mode == "online" { "real" } else { "sim" }
        );
        let dynamo: Value =
            serde_json::from_slice(&std::fs::read(target.join("dynamo/report.json")).unwrap())
                .unwrap();
        counts.push(dynamo["num_requests"].as_u64().unwrap());
        std::fs::remove_dir_all(target).unwrap();
    }
    assert_eq!(
        counts[0], counts[1],
        "offline and online replay served different request counts"
    );
}

/// Apples-to-apples gate through the **real runner product path**: a hash-block
/// trace is replayed by the runner's `dynamo_offline` + `replay_mode=online`
/// scheduled pair (AIPerf's own flow, native-format `synthesize_tokens` tokens
/// via the trace-hash encoder — no injected test encoder), and its native-v2
/// report is compared to Dynamo's own wall-clock online driver
/// (`simulate_concurrency_live_requests`) over the same hash ids and engine
/// args. Both measure real wall-clock latency, so the comparison is exact:
/// counts identical, every latency stat within 3%, throughput >= native.
#[test]
fn online_product_path_matches_native_live_replay_within_3pct() {
    const REQUESTS: usize = 16;
    const BLOCK_SIZE: usize = 16;
    const BLOCKS: usize = 16;
    const ISL: usize = BLOCK_SIZE * BLOCKS;
    const OSL: usize = 8;

    // Unique hash-block ids per request (no cross-request cache reuse: every
    // request does a full prefill, so TTFT is engine-compute-dominated).
    let hash_ids: Vec<Vec<i64>> = (0..REQUESTS)
        .map(|i| (0..BLOCKS as i64).map(|b| (i as i64) * 1000 + b).collect())
        .collect();

    let records: Vec<Value> = hash_ids
        .iter()
        .enumerate()
        .map(|(i, ids)| {
            json!({
                "session_id": format!("r{i}"),
                "timestamp": 0,
                "input_length": ISL,
                "output_length": OSL,
                "hash_ids": ids
            })
        })
        .collect();

    let distribution_id = distribution_id();
    let target = target("product-apples");
    let _ = std::fs::remove_dir_all(&target);
    let mut request = json!({
        "protocol_version": 2,
        "operation": "execute",
        "expected_distribution_id": distribution_id,
        "run": {
            "identity": {"benchmark_id": "online-product-apples", "random_seed": 41},
            "artifact_target": target,
            "backend": {
                "type": "dynamo_offline",
                "config": {
                    "replay_mode": "online",
                    "engine": {"block_size": BLOCK_SIZE},
                    "artifacts": {"report_json": "dynamo/report.json"}
                }
            },
            "workload": {
                "type": "scheduled",
                "config": {
                    "worker_count": 1,
                    "dataset": {
                        "type": "file",
                        "format": "mooncake_trace",
                        "sampling": "sequential",
                        "osl": {"value": OSL as f64},
                        "records": records
                    },
                    "tokenizer": {
                        "name": "builtin",
                        "revision": "main",
                        "trust_remote_code": false,
                        "apply_chat_template": false
                    },
                    "phases": [{
                        "type": "concurrency",
                        "name": "profiling",
                        "exclude_from_results": false,
                        "requests": REQUESTS,
                        "concurrency": REQUESTS
                    }]
                }
            },
            "resources": {
                "models": {"items": [{"name": "mock-model"}]},
                "endpoints": {"profiles": [{
                    "id": "default",
                    "type": "chat_completions",
                    "urls": ["http://127.0.0.1:9"]
                }]},
                "metrics": {"slice_duration_seconds": 0.5, "slos": {}},
                "artifacts": {},
                "sidecars": {}
            }
        }
    });
    request["run"]["backend"]["config"]["engine"]["block_size"] = json!(BLOCK_SIZE);

    let output = run(&request);
    let terminal = one_line(&output);
    assert!(
        output.status.success(),
        "terminal={terminal}, stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(terminal["provenance"]["replay_mode"], "online");
    let native_v2: Value =
        serde_json::from_slice(&std::fs::read(target.join("native-v2.json")).unwrap()).unwrap();

    // Native baseline: Dynamo's own real-clock online driver over the same hash
    // ids and engine args, byte-identical native-format tokens.
    let baseline_reqs: Vec<NativeBaselineRequest> = hash_ids
        .iter()
        .map(|ids| NativeBaselineRequest {
            hash_ids: ids.clone(),
            max_output_tokens: OSL,
        })
        .collect();
    let native = native_live_concurrency_baseline(
        &format!(r#"{{"block_size":{BLOCK_SIZE}}}"#),
        &baseline_reqs,
        ISL,
        REQUESTS,
        1,
    )
    .unwrap();

    let avg = |tag: &str| -> f64 {
        native_v2["metrics"][tag]["series"][0]["stats"]["avg"]
            .as_f64()
            .unwrap_or_else(|| panic!("missing metric {tag} avg in native-v2 report"))
    };
    let total = |tag: &str| -> f64 {
        native_v2["metrics"][tag]["series"][0]["stats"]["total"]
            .as_f64()
            .or_else(|| native_v2["metrics"][tag]["series"][0]["stats"]["value"].as_f64())
            .unwrap_or_else(|| panic!("missing metric {tag} total/value in native-v2 report"))
    };

    // Counts: exact (the sim results are 100% comparable).
    assert_eq!(total("request_count") as usize, native.completed_requests);
    assert_eq!(
        total("total_usage_prompt_tokens") as usize,
        native.total_input_tokens
    );
    assert_eq!(
        total("total_usage_completion_tokens") as usize,
        native.total_output_tokens
    );

    // Latency: every stat within 3% (1ms absolute floor for sub-ms noise).
    let within_3pct = |name: &str, a: f64, n: f64| {
        let delta = (a - n).abs();
        let tol = (n.abs() * 0.03).max(1.0);
        assert!(
            delta <= tol,
            "{name}: runner={a:.4} native={n:.4} delta={delta:.4} exceeds 3% (tol={tol:.4})"
        );
    };
    within_3pct("ttft", avg("time_to_first_token"), native.ttft_mean_ms);
    within_3pct("e2e", avg("request_latency"), native.request_latency_mean_ms);
    within_3pct(
        "itl",
        avg("inter_token_latency"),
        native.inter_token_latency_mean_ms,
    );

    // Throughput: the runner product path is at least as fast as native
    // (3% slack for real-timer jitter).
    let runner_rps = total("request_throughput");
    assert!(
        runner_rps >= native.request_throughput_rps * 0.97,
        "runner rps {runner_rps:.3} < native rps {:.3}",
        native.request_throughput_rps
    );

    std::fs::remove_dir_all(target).unwrap();
}
