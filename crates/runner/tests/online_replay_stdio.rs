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

#![cfg(feature = "dynosim")]

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

use aiperf::dynosim::{NativeBaselineRequest, native_live_concurrency_baseline};
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
    transport_type: &str,
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
            "transport": {
                "type": transport_type,
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
        "dynosim_online",
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
    assert_eq!(terminal["provenance"]["transport"], "dynosim_online");
    assert_eq!(terminal["provenance"]["clock"], "real");

    let native: Value =
        serde_json::from_slice(&std::fs::read(target.join("native-v2.json")).unwrap()).unwrap();
    assert_eq!(native["run"]["mode"], "online:scheduled");
    assert_eq!(native["run"]["dynamo"]["clock"], "real");
    assert_eq!(
        native["metrics"]["request_count"]["series"][0]["stats"]["total"], 4.0,
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
            &format!("dynosim_{mode}"),
            phases.clone(),
        );
        let output = run(&request);
        let terminal = one_line(&output);
        assert!(
            output.status.success(),
            "mode={mode} terminal={terminal}, stderr={}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(
            terminal["provenance"]["transport"],
            format!("dynosim_{mode}")
        );
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
/// trace is replayed by the runner's `dynosim` + `replay_mode=online`
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
            "transport": {
                "type": "dynosim_online",
                "config": {
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
    request["run"]["transport"]["config"]["engine"]["block_size"] = json!(BLOCK_SIZE);

    let output = run(&request);
    let terminal = one_line(&output);
    assert!(
        output.status.success(),
        "terminal={terminal}, stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(terminal["provenance"]["transport"], "dynosim_online");
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
    within_3pct(
        "e2e",
        avg("request_latency"),
        native.request_latency_mean_ms,
    );
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

/// Locate the sibling `dynamo-aiperf-native` checkout that provides the native
/// `python -m dynamo.replay` CLI (its compiled bindings + components). Honors
/// `AIPERF_DYNAMO_NATIVE_DIR`, else falls back to the conventional sibling path.
fn dynamo_native_dir() -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(dir) = std::env::var("AIPERF_DYNAMO_NATIVE_DIR") {
        candidates.push(PathBuf::from(dir));
    }
    if let Ok(home) = std::env::var("HOME") {
        candidates.push(PathBuf::from(home).join("nvidia/projects/dynamo-aiperf-native"));
    }
    candidates
        .into_iter()
        .find(|dir| dir.join("components/src/dynamo/replay").is_dir())
}

fn python_bin() -> Option<&'static str> {
    for bin in ["python3", "python"] {
        if Command::new(bin)
            .arg("--version")
            .output()
            .is_ok_and(|o| o.status.success())
        {
            return Some(bin);
        }
    }
    None
}

/// End-to-end **subprocess vs subprocess** apples-to-apples gate: the SAME
/// mooncake hash-block trace file is replayed under the real clock by
///   (1) the AIPerf product path — the `aiperf-runner` binary with
///       `dynosim` + `replay_mode=online` (AIPerf's own flow), and
///   (2) Dynamo's own native CLI — `python -m dynamo.replay --replay-mode online`.
/// Both drive the same passive engine and measure real wall-clock latency; the
/// gate asserts input/output tokens exact, every latency mean within 3%, and the
/// AIPerf product throughput >= native. Skips when the native checkout/python is
/// unavailable (cross-repo integration gate).
#[test]
fn online_product_path_matches_python_dynamo_replay_subprocess_within_3pct() {
    let Some(native_dir) = dynamo_native_dir() else {
        eprintln!("SKIP: no dynamo-aiperf-native checkout (set AIPERF_DYNAMO_NATIVE_DIR)");
        return;
    };
    let Some(python) = python_bin() else {
        eprintln!("SKIP: no python interpreter");
        return;
    };
    let pythonpath = format!(
        "{}:{}",
        native_dir.join("components/src").display(),
        native_dir.join("lib/bindings/python/src").display()
    );
    // Verify the native replay CLI actually imports before committing the run.
    let probe = Command::new(python)
        .env("PYTHONPATH", &pythonpath)
        .args(["-m", "dynamo.replay", "--help"])
        .output();
    if !probe.is_ok_and(|o| o.status.success()) {
        eprintln!("SKIP: `python -m dynamo.replay` not runnable with PYTHONPATH={pythonpath}");
        return;
    }

    const REQUESTS: usize = 16;
    const BLOCK_SIZE: usize = 16;
    const BLOCKS: usize = 16;
    const ISL: usize = BLOCK_SIZE * BLOCKS;
    const OSL: usize = 8;

    // One mooncake trace file, unique hash-block ids per request (no cross-request
    // reuse → full prefill each → engine-compute-dominated TTFT). Consumed byte
    // for byte by BOTH subprocesses.
    let base = std::env::temp_dir().join(format!("aiperf-apples-{}", std::process::id()));
    std::fs::create_dir_all(&base).unwrap();
    let trace_path = base.join("trace.jsonl");
    {
        let mut trace = String::new();
        for i in 0..REQUESTS {
            let hash_ids: Vec<i64> = (0..BLOCKS as i64).map(|b| (i as i64) * 1000 + b).collect();
            trace.push_str(
                &serde_json::to_string(&json!({
                    "timestamp": 0,
                    "input_length": ISL,
                    "output_length": OSL,
                    "hash_ids": hash_ids
                }))
                .unwrap(),
            );
            trace.push('\n');
        }
        std::fs::write(&trace_path, trace).unwrap();
    }

    // (1) AIPerf product path: aiperf-runner subprocess, replay_mode=online,
    //     reading the SAME trace file.
    let distribution_id = distribution_id();
    let target = base.join("aiperf");
    let request = json!({
        "protocol_version": 2,
        "operation": "execute",
        "expected_distribution_id": distribution_id,
        "run": {
            "identity": {"benchmark_id": "online-apples-subproc", "random_seed": 41},
            "artifact_target": target,
            "transport": {
                "type": "dynosim_online",
                "config": {
                    "engine": {"block_size": BLOCK_SIZE},
                    "artifacts": {}
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
                        "path": trace_path,
                        "osl": {"value": OSL as f64}
                    },
                    "tokenizer": {
                        "name": "builtin", "revision": "main",
                        "trust_remote_code": false, "apply_chat_template": false
                    },
                    "phases": [{
                        "type": "concurrency", "name": "profiling",
                        "exclude_from_results": false,
                        "requests": REQUESTS, "concurrency": REQUESTS
                    }]
                }
            },
            "resources": {
                "models": {"items": [{"name": "mock-model"}]},
                "endpoints": {"profiles": [{
                    "id": "default", "type": "chat_completions",
                    "urls": ["http://127.0.0.1:9"]
                }]},
                "metrics": {"slice_duration_seconds": 0.5, "slos": {}},
                "artifacts": {}, "sidecars": {}
            }
        }
    });
    let output = run(&request);
    let terminal = one_line(&output);
    assert!(
        output.status.success(),
        "aiperf terminal={terminal}, stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(terminal["provenance"]["transport"], "dynosim_online");
    let aiperf: Value =
        serde_json::from_slice(&std::fs::read(target.join("native-v2.json")).unwrap()).unwrap();
    let a_avg = |tag: &str| -> f64 {
        aiperf["metrics"][tag]["series"][0]["stats"]["avg"]
            .as_f64()
            .unwrap()
    };
    let a_total = |tag: &str| -> f64 {
        aiperf["metrics"][tag]["series"][0]["stats"]["total"]
            .as_f64()
            .or_else(|| aiperf["metrics"][tag]["series"][0]["stats"]["value"].as_f64())
            .unwrap()
    };

    // (2) Native path: python -m dynamo.replay subprocess on the SAME trace.
    let native_report = base.join("native.json");
    let native_out = Command::new(python)
        .env("PYTHONPATH", &pythonpath)
        .args([
            "-m",
            "dynamo.replay",
            trace_path.to_str().unwrap(),
            "--replay-mode",
            "online",
            "--replay-concurrency",
            &REQUESTS.to_string(),
            "--num-workers",
            "1",
            "--extra-engine-args",
            &format!(r#"{{"block_size":{BLOCK_SIZE}}}"#),
            "--trace-format",
            "mooncake",
            "--report-json",
            native_report.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        native_out.status.success(),
        "python dynamo.replay failed: {}",
        String::from_utf8_lossy(&native_out.stderr)
    );
    let native: Value = serde_json::from_slice(&std::fs::read(&native_report).unwrap()).unwrap();
    let n = |key: &str| -> f64 { native[key].as_f64().unwrap() };

    // Counts: exact.
    assert_eq!(
        a_total("total_usage_prompt_tokens"),
        n("total_input_tokens")
    );
    assert_eq!(
        a_total("total_usage_completion_tokens"),
        n("total_output_tokens")
    );
    assert_eq!(a_total("request_count"), n("completed_requests"));

    // Latency means: within 3% (1ms floor).
    let within_3pct = |name: &str, a: f64, nv: f64| {
        let tol = (nv.abs() * 0.03).max(1.0);
        assert!(
            (a - nv).abs() <= tol,
            "{name}: aiperf={a:.4} python-dynamo={nv:.4} delta={:.4} exceeds 3% (tol={tol:.4})",
            (a - nv).abs()
        );
    };
    within_3pct("ttft", a_avg("time_to_first_token"), n("mean_ttft_ms"));
    within_3pct("e2e", a_avg("request_latency"), n("mean_e2e_latency_ms"));
    within_3pct("itl", a_avg("inter_token_latency"), n("mean_itl_ms"));

    // Throughput: AIPerf product path >= native (3% slack for real-timer jitter).
    let a_rps = a_total("request_throughput");
    assert!(
        a_rps >= n("request_throughput_rps") * 0.97,
        "aiperf rps {a_rps:.3} < python-dynamo rps {:.3}",
        n("request_throughput_rps")
    );

    std::fs::remove_dir_all(&base).unwrap();
}

/// End-to-end **byte-exact** offline gate: the runner's `dynosim` offline
/// concurrency replay (auto-driven by Dynamo's `execute_pass` single engine for
/// this single-worker, single-turn, no-ancillary-timing case) versus
/// `python -m dynamo.replay --replay-mode offline` on the SAME mooncake trace.
/// Both are deterministic virtual-clock runs, so counts and `wall_time_ms` are
/// byte-identical and every latency mean matches native's own 6-decimal report
/// precision — the parity that failed (tens of percent) before the single-engine
/// fix under prefill saturation. Skips when the native checkout/python is absent.
#[test]
fn offline_product_path_is_byte_exact_with_python_dynamo_replay() {
    let Some(native_dir) = dynamo_native_dir() else {
        eprintln!("SKIP: no dynamo-aiperf-native checkout");
        return;
    };
    let Some(python) = python_bin() else {
        eprintln!("SKIP: no python");
        return;
    };
    let pythonpath = format!(
        "{}:{}",
        native_dir.join("components/src").display(),
        native_dir.join("lib/bindings/python/src").display()
    );
    if !Command::new(python)
        .env("PYTHONPATH", &pythonpath)
        .args(["-m", "dynamo.replay", "--help"])
        .output()
        .is_ok_and(|o| o.status.success())
    {
        eprintln!("SKIP: dynamo.replay not runnable");
        return;
    }

    // Saturated: 32 requests, ISL 256 (16 blocks), conc 16 → concurrent prefill
    // demand (16*256) exceeds the default 8192 batch budget.
    const REQUESTS: usize = 32;
    const BLOCKS: usize = 16;
    const ISL: usize = 16 * BLOCKS;
    const OSL: usize = 8;
    const CONCURRENCY: usize = 16;

    let base = std::env::temp_dir().join(format!("aiperf-offline-be-{}", std::process::id()));
    std::fs::create_dir_all(&base).unwrap();
    let trace_path = base.join("trace.jsonl");
    {
        let mut trace = String::new();
        for i in 0..REQUESTS {
            let hash_ids: Vec<i64> = (0..BLOCKS as i64).map(|b| (i as i64) * 1000 + b).collect();
            trace.push_str(
                &serde_json::to_string(&json!({
                    "timestamp": 0, "input_length": ISL, "output_length": OSL, "hash_ids": hash_ids
                }))
                .unwrap(),
            );
            trace.push('\n');
        }
        std::fs::write(&trace_path, trace).unwrap();
    }

    let target = base.join("aiperf");
    let request = json!({
        "protocol_version": 2, "operation": "execute",
        "expected_distribution_id": distribution_id(),
        "run": {
            "identity": {"benchmark_id": "offline-be", "random_seed": 41},
            "artifact_target": target,
            "transport": {"type": "dynosim_offline", "config": {
                "engine": {"block_size": 16},
                "artifacts": {"report_json": "dynamo/report.json"}
            }},
            "workload": {"type": "scheduled", "config": {
                "worker_count": 1,
                "dataset": {"type": "file", "format": "mooncake_trace", "sampling": "sequential",
                            "path": trace_path, "osl": {"value": OSL as f64}},
                "tokenizer": {"name": "builtin", "revision": "main",
                              "trust_remote_code": false, "apply_chat_template": false},
                "phases": [{"type": "concurrency", "name": "profiling",
                            "exclude_from_results": false, "requests": REQUESTS, "concurrency": CONCURRENCY}]
            }},
            "resources": {
                "models": {"items": [{"name": "mock-model"}]},
                "endpoints": {"profiles": [{"id": "default", "type": "chat_completions", "urls": ["http://127.0.0.1:9"]}]},
                "metrics": {"slos": {}}, "artifacts": {}, "sidecars": {}
            }
        }
    });
    let output = run(&request);
    let terminal = one_line(&output);
    assert!(
        output.status.success(),
        "aiperf terminal={terminal} stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    // clock=sim: this is a deterministic offline run.
    assert_eq!(terminal["provenance"]["clock"], "sim");
    let aiperf: Value =
        serde_json::from_slice(&std::fs::read(target.join("dynamo/report.json")).unwrap()).unwrap();

    let native_report = base.join("native.json");
    let native_out = Command::new(python)
        .env("PYTHONPATH", &pythonpath)
        .args([
            "-m",
            "dynamo.replay",
            trace_path.to_str().unwrap(),
            "--replay-mode",
            "offline",
            "--replay-concurrency",
            &CONCURRENCY.to_string(),
            "--num-workers",
            "1",
            "--extra-engine-args",
            r#"{"block_size":16}"#,
            "--trace-format",
            "mooncake",
            "--report-json",
            native_report.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        native_out.status.success(),
        "python dynamo.replay failed: {}",
        String::from_utf8_lossy(&native_out.stderr)
    );
    let native: Value = serde_json::from_slice(&std::fs::read(&native_report).unwrap()).unwrap();

    let f = |v: &Value, k: &str| v[k].as_f64().unwrap();
    // Counts + wall time: byte-identical (deterministic offline).
    assert_eq!(
        f(&aiperf, "completed_requests"),
        f(&native, "completed_requests")
    );
    assert_eq!(
        f(&aiperf, "total_input_tokens"),
        f(&native, "total_input_tokens")
    );
    assert_eq!(
        f(&aiperf, "total_output_tokens"),
        f(&native, "total_output_tokens")
    );
    assert_eq!(
        f(&aiperf, "wall_time_ms"),
        f(&native, "wall_time_ms"),
        "wall_time not byte-exact"
    );
    // Latency means: equal to native's 6-decimal report precision (the property
    // that diverged by tens of percent before the single-engine fix).
    let to_native_precision = |name: &str| {
        let a = f(&aiperf, name);
        let n = f(&native, name);
        assert!(
            (a - n).abs() <= n.abs() * 1e-6 + 1e-6,
            "{name}: aiperf={a} native={n}"
        );
    };
    to_native_precision("mean_ttft_ms");
    to_native_precision("mean_e2e_latency_ms");
    to_native_precision("mean_itl_ms");

    std::fs::remove_dir_all(&base).unwrap();
}
