// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! End-to-end tests for the native `aiperf` binary.
//!
//! The full run test is `#[ignore]` because it needs the sibling
//! `aiperf-runner` and `aiperf-mock-server` binaries built into the same target
//! directory. Run it with:
//!   cargo build -p aiperf-runner --no-default-features -p aiperf-mock-server
//!   cargo test -p aiperf-cli --test e2e_mock -- --ignored

use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::time::{Duration, Instant};

/// Path to the built `aiperf` binary under test.
fn aiperf_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_aiperf"))
}

/// A sibling binary in the same target dir as `aiperf` (runner / mock server).
fn sibling(name: &str) -> PathBuf {
    aiperf_bin().parent().expect("target dir").join(name)
}

#[test]
fn unknown_subcommand_delegates_to_python() {
    // A subcommand the native binary does not own re-execs `python -m aiperf`.
    // With or without a python env, the native binary must not treat it as its
    // own command (it either delegates and returns python's code, or reports a
    // delegation failure) — never a clap "unknown subcommand" from our parser.
    let out = Command::new(aiperf_bin())
        .args(["definitely-not-native", "--help"])
        .env("AIPERF_PYTHON", "this-python-does-not-exist")
        .output()
        .expect("spawn aiperf");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("delegate") || stderr.contains("aiperf"),
        "expected a delegation attempt, got: {stderr}"
    );
}

/// Reserve an ephemeral localhost port by binding and immediately dropping it.
fn free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral");
    listener.local_addr().expect("addr").port()
}

/// Wait until `child` has bound `port` (or fail after a timeout).
fn wait_for_port(port: u16, child: &mut Child) {
    let deadline = Instant::now() + Duration::from_secs(15);
    while Instant::now() < deadline {
        if TcpListener::bind(("127.0.0.1", port)).is_err() {
            // Port is taken → the mock is listening.
            return;
        }
        if let Ok(Some(status)) = child.try_wait() {
            panic!("mock server exited early with {status}");
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    panic!("mock server did not bind port {port} in time");
}

#[test]
#[ignore = "needs sibling aiperf-runner + aiperf-mock-server built in the target dir"]
fn profile_single_run_against_mock_writes_native_report() {
    let runner = sibling("aiperf-runner");
    let mock = sibling("aiperf-mock-server");
    assert!(runner.exists(), "build aiperf-runner first: {runner:?}");
    assert!(mock.exists(), "build aiperf-mock-server first: {mock:?}");

    let out_dir = tempfile::tempdir().expect("tempdir");
    let port = free_port();

    let mut mock_child = Command::new(&mock)
        .args(["--fast", "--host", "127.0.0.1", "--port", &port.to_string()])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("spawn mock");
    wait_for_port(port, &mut mock_child);

    let status = Command::new(aiperf_bin())
        .args([
            "profile",
            "--model",
            "Qwen/Qwen3-0.6B",
            "--url",
            &format!("127.0.0.1:{port}"),
            "--endpoint-type",
            "chat",
            "--streaming",
            "--concurrency",
            "1",
            "--request-count",
            "5",
            "--artifact-dir",
            out_dir.path().to_str().unwrap(),
        ])
        .env("AIPERF_RUNNER_BIN", &runner)
        .status()
        .expect("spawn aiperf profile");

    let _ = mock_child.kill();

    assert!(status.success(), "native profile run failed: {status}");
    let report = out_dir.path().join("native-v2.json");
    assert!(report.exists(), "native-v2.json not written at {report:?}");
    assert_report_has_records(&report);
}

#[test]
#[ignore = "needs sibling aiperf-runner + aiperf-mock-server built in the target dir"]
fn profile_yaml_config_against_mock_writes_native_report() {
    // Exercises the full YAML `--config` path end-to-end (not just the flag path).
    let runner = sibling("aiperf-runner");
    let mock = sibling("aiperf-mock-server");
    assert!(runner.exists(), "build aiperf-runner first: {runner:?}");
    assert!(mock.exists(), "build aiperf-mock-server first: {mock:?}");

    let out_dir = tempfile::tempdir().expect("tempdir");
    let port = free_port();

    let mut mock_child = Command::new(&mock)
        .args(["--fast", "--host", "127.0.0.1", "--port", &port.to_string()])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("spawn mock");
    wait_for_port(port, &mut mock_child);

    // A config that exercises several YAML-only sections (warmup/profiling blocks,
    // multi-turn, tokenizer, telemetry-off) resolved purely from YAML.
    let config = out_dir.path().join("bench.yaml");
    std::fs::write(
        &config,
        format!(
            "schemaVersion: \"2.0\"\n\
             randomSeed: 7\n\
             benchmark:\n\
             \x20 model: Qwen/Qwen3-0.6B\n\
             \x20 endpoint:\n\
             \x20   type: chat\n\
             \x20   url: 127.0.0.1:{port}\n\
             \x20   streaming: true\n\
             \x20 dataset:\n\
             \x20   type: synthetic\n\
             \x20   prompts: {{isl: 128, osl: 32}}\n\
             \x20 gpu_telemetry: {{enabled: false}}\n\
             \x20 server_metrics: {{enabled: false}}\n\
             \x20 warmup:\n\
             \x20   type: concurrency\n\
             \x20   requests: 2\n\
             \x20   concurrency: 1\n\
             \x20 profiling:\n\
             \x20   type: concurrency\n\
             \x20   requests: 5\n\
             \x20   concurrency: 2\n\
             \x20 artifacts:\n\
             \x20   dir: {}\n",
            out_dir.path().to_str().unwrap(),
        ),
    )
    .expect("write config");

    let status = Command::new(aiperf_bin())
        .args(["profile", "--config", config.to_str().unwrap()])
        .env("AIPERF_RUNNER_BIN", &runner)
        .status()
        .expect("spawn aiperf profile --config");

    let _ = mock_child.kill();

    assert!(status.success(), "native YAML profile run failed: {status}");
    let report = out_dir.path().join("native-v2.json");
    assert!(report.exists(), "native-v2.json not written at {report:?}");
    assert_report_has_records(&report);
}

#[test]
#[ignore = "needs sibling aiperf-runner + aiperf-mock-server built in the target dir"]
fn profile_sweep_against_mock_writes_per_cell_reports() {
    // A 2-cell concurrency sweep runs both cells and writes an aggregate.
    let runner = sibling("aiperf-runner");
    let mock = sibling("aiperf-mock-server");
    assert!(runner.exists(), "build aiperf-runner first: {runner:?}");
    assert!(mock.exists(), "build aiperf-mock-server first: {mock:?}");

    let out_dir = tempfile::tempdir().expect("tempdir");
    let port = free_port();
    let mut mock_child = Command::new(&mock)
        .args(["--fast", "--host", "127.0.0.1", "--port", &port.to_string()])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("spawn mock");
    wait_for_port(port, &mut mock_child);

    let status = Command::new(aiperf_bin())
        .args([
            "profile",
            "--model",
            "Qwen/Qwen3-0.6B",
            "--url",
            &format!("127.0.0.1:{port}"),
            "--endpoint-type",
            "chat",
            "--streaming",
            "--concurrency",
            "2,4",
            "--request-count",
            "6",
            "--artifact-dir",
            out_dir.path().to_str().unwrap(),
        ])
        .env("AIPERF_RUNNER_BIN", &runner)
        .status()
        .expect("spawn aiperf profile sweep");
    let _ = mock_child.kill();
    assert!(status.success(), "native sweep run failed: {status}");

    // Both cells produced a report, and the aggregate exists.
    for dir in ["concurrency_2", "concurrency_4"] {
        let report = out_dir.path().join(dir).join("native-v2.json");
        assert!(report.exists(), "missing per-cell report {report:?}");
        assert_report_has_records(&report);
    }
    let agg = out_dir
        .path()
        .join("sweep_aggregate")
        .join("profile_export_aiperf_sweep.json");
    assert!(agg.exists(), "missing sweep aggregate {agg:?}");
    let doc: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&agg).unwrap()).expect("aggregate json");
    assert_eq!(doc["cells"].as_array().unwrap().len(), 2, "2 sweep cells");
}

/// Assert the native report is a schema-2.0 object with a metrics section.
fn assert_report_has_records(report: &Path) {
    let bytes = std::fs::read(report).expect("read report");
    let value: serde_json::Value = serde_json::from_slice(&bytes).expect("report is JSON");
    assert_eq!(value["schema_version"], serde_json::json!("2.0"));
    assert!(value.get("metrics").is_some(), "report missing metrics");
    assert!(value.get("summary").is_some(), "report missing summary");
}
