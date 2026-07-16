// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! End-to-end verification of the native `aiperf` front door's logging parity
//! with the Python frontend (`src/aiperf/common/logging.py`).
//!
//! Unlike the rest of the e2e suite (which drives `python -m aiperf`), this test
//! runs the native `aiperf` binary DIRECTLY as the front door, so it exercises
//! the Rust logging stack end to end:
//!
//! - the default level is INFO (no verbosity flag given),
//! - the front-door lifecycle lines (`Starting native AIPerf run` … `Native
//!   AIPerf run completed`) are emitted,
//! - the `--execute` child's runtime narrative (dataset / phase lifecycle /
//!   export) is forwarded as `aiperf-runner:` lines, and
//! - every line lands in `<artifact_dir>/logs/aiperf.log` (Python's FileHandler).

mod common;

use std::io::Read;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use common::{DEFAULT_MODEL, MockServer, exec_binary};

/// Drive the native `aiperf` binary directly (bypassing `python -m aiperf`) and
/// return `(exit_code, artifact_dir)`. Bounded run, with a hard timeout guard.
fn run_native_profile(mock_url: &str, artifact_dir: &std::path::Path) -> i32 {
    let mut cmd = Command::new(exec_binary());
    cmd.arg("profile")
        .args(["--model", DEFAULT_MODEL])
        .args(["--tokenizer", DEFAULT_MODEL])
        .args(["--url", mock_url])
        .args(["--endpoint-type", "chat"])
        .arg("--streaming")
        .args(["--request-count", "8"])
        .args(["--warmup-request-count", "2"])
        .args(["--concurrency", "2"])
        .arg("--artifact-dir")
        .arg(artifact_dir)
        .env("HF_HUB_OFFLINE", "1")
        .env("TRANSFORMERS_OFFLINE", "1")
        .env("PYTHONUNBUFFERED", "1")
        .env("MALLOC_ARENA_MAX", "2")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = cmd.spawn().expect("failed to spawn native aiperf");
    // Drain both pipes so a full OS pipe buffer can never deadlock the child.
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

    let exit = run_native_profile(&mock.url, dir.path());
    assert_eq!(exit, 0, "native aiperf profile run should exit 0");

    // Python's `setup_rich_logging` writes to `<artifact_dir>/logs/aiperf.log`;
    // the native front door must produce the same file.
    let log_path = dir.path().join("logs").join("aiperf.log");
    let log = std::fs::read_to_string(&log_path)
        .unwrap_or_else(|e| panic!("logs/aiperf.log missing at {}: {e}", log_path.display()));

    // The run's milestone narrative, in the order it must appear. Front-door
    // lines come from `aiperf_cli`; the runtime lines are forwarded from the
    // `--execute` child as `aiperf-runner:` lines. Wording mirrors the Python
    // frontend so existing log-scraping keeps matching on the native path.
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

    // Default level is INFO: the file must carry INFO lines and no DEBUG/TRACE
    // (no verbosity flag was passed).
    assert!(log.contains(" INFO "), "log should contain INFO-level lines");
    assert!(
        !log.contains(" DEBUG ") && !log.contains(" TRACE "),
        "default run must not emit DEBUG/TRACE lines"
    );
}
