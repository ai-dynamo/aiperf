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
//! clock. Byte-exact latency parity is deliberately not asserted here — real
//! timers cannot reproduce the engine's internal completion times; the library
//! within-tolerance parity is proven in `aiperf`'s own tests.

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
