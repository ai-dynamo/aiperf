// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod common;

use std::io::Read;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use common::{DEFAULT_MODEL, MockServer, exec_binary};

fn run_native_profile(
    mock_url: &str,
    artifact_dir: &std::path::Path,
    extra_args: &[&str],
    extra_env: &[(&str, &str)],
) -> i32 {
    let mut cmd = Command::new(exec_binary());
    cmd.arg("profile")
        .args(["--model", DEFAULT_MODEL])
        .args(["--tokenizer", DEFAULT_MODEL])
        .args(["--url", mock_url])
        .args(["--endpoint-type", "chat"])
        .arg("--streaming")
        .args(extra_args)
        .arg("--artifact-dir")
        .arg(artifact_dir)
        .env("HF_HUB_OFFLINE", "1")
        .env("TRANSFORMERS_OFFLINE", "1")
        .env("PYTHONUNBUFFERED", "1")
        .env("MALLOC_ARENA_MAX", "2")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    for (key, value) in extra_env {
        cmd.env(key, value);
    }

    let mut child = cmd.spawn().expect("failed to spawn native aiperf");
    // Drain both pipes concurrently to prevent child blockage on full buffers.
    let mut out = child.stdout.take().expect("child stdout");
    let mut err = child.stderr.take().expect("child stderr");
    let out_thread = std::thread::spawn(move || {
        let mut s = String::new();
        let _ = out.read_to_string(&mut s);
        s
    });
    let err_thread = std::thread::spawn(move || {
        let mut s = String::new();
        let _ = err.read_to_string(&mut s);
        s
    });

    let deadline = Instant::now() + Duration::from_secs(180);
    let status = loop {
        match child.try_wait().expect("try_wait") {
            Some(status) => break status,
            None if Instant::now() >= deadline => {
                let _ = child.kill();
                break child.wait().expect("wait after kill");
            }
            None => std::thread::sleep(Duration::from_millis(50)),
        }
    };
    let _ = out_thread.join();
    let _ = err_thread.join();
    status.code().unwrap_or(-1)
}

#[tokio::test]
async fn native_front_door_writes_full_log_narrative() {
    let mock = MockServer::start();
    let dir = tempfile::tempdir().expect("tempdir");

    let exit = run_native_profile(
        &mock.url,
        dir.path(),
        &[
            "--request-count",
            "8",
            "--warmup-request-count",
            "2",
            "--concurrency",
            "2",
        ],
        &[],
    );
    assert_eq!(exit, 0, "native aiperf profile run should exit 0");

    let log_path = dir.path().join("logs").join("aiperf.log");
    let log = std::fs::read_to_string(&log_path)
        .unwrap_or_else(|e| panic!("logs/aiperf.log missing at {}: {e}", log_path.display()));

    // Existing log consumers depend on these exact milestones and their order.
    let ordered_milestones = [
        "Starting native AIPerf run",
        "Initialized 2 phase(s)",
        "Phase warmup started",
        "Phase warmup complete",
        "Phase profiling started",
        "Phase profiling complete",
        "All credits completed",
        "Processing records results...",
        "Report written to:",
        "Exporting all records",
        "Native AIPerf run completed",
    ];

    let mut search_from = 0usize;
    for milestone in ordered_milestones {
        let found = log[search_from..].find(milestone).unwrap_or_else(|| {
            panic!(
                "log/aiperf.log missing milestone {milestone:?} (after prior milestones).\n\
                 --- log ---\n{log}"
            )
        });
        search_from += found + milestone.len();
    }

    assert!(
        log.contains(" INFO "),
        "log should contain INFO-level lines"
    );
    assert!(
        !log.contains(" DEBUG ") && !log.contains(" TRACE "),
        "default run must not emit DEBUG/TRACE lines"
    );
}

#[tokio::test]
async fn realtime_metrics_block_is_logged() {
    let mock = MockServer::start();
    let dir = tempfile::tempdir().expect("tempdir");

    // Pacing keeps profiling active long enough to emit interval metrics.
    let exit = run_native_profile(
        &mock.url,
        dir.path(),
        &[
            "--request-rate",
            "5",
            "--request-count",
            "20",
            "--warmup-request-count",
            "1",
        ],
        &[("AIPERF_STATS_INTERVAL", "1")],
    );
    assert_eq!(exit, 0, "native aiperf profile run should exit 0");

    let log_path = dir.path().join("logs").join("aiperf.log");
    let log = std::fs::read_to_string(&log_path)
        .unwrap_or_else(|e| panic!("logs/aiperf.log missing at {}: {e}", log_path.display()));

    let headers = log
        .lines()
        .filter(|line| line.contains("[realtime ") && line.contains(" profiling]"))
        .count();
    assert!(
        headers >= 1,
        "expected at least one [realtime …] block header in the log; got none.\n\
         --- log ---\n{log}"
    );
    assert!(
        log.contains("done=") && log.contains("ok=") && log.contains("err="),
        "realtime block must include the done/ok/err counter row.\n--- log ---\n{log}"
    );
    assert!(
        log.contains("  ttft ") && log.contains("  itl ") && log.contains("  e2e "),
        "realtime block must include the TTFT/ITL/e2e latency rows.\n--- log ---\n{log}"
    );
}

#[tokio::test]
async fn realtime_heartbeat_disabled_by_zero_interval() {
    let mock = MockServer::start();
    let dir = tempfile::tempdir().expect("tempdir");

    let exit = run_native_profile(
        &mock.url,
        dir.path(),
        &[
            "--request-rate",
            "5",
            "--request-count",
            "20",
            "--warmup-request-count",
            "1",
        ],
        &[("AIPERF_STATS_INTERVAL", "0")],
    );
    assert_eq!(exit, 0, "native aiperf profile run should exit 0");

    let log_path = dir.path().join("logs").join("aiperf.log");
    let log = std::fs::read_to_string(&log_path).unwrap_or_default();
    assert!(
        !log.contains("[realtime "),
        "AIPERF_STATS_INTERVAL=0 must suppress the realtime heartbeat, but the log has one.\n\
         --- log ---\n{log}"
    );
}
