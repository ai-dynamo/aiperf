// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod common;
use common::*;

// Ctrl+C stops admission, drains in-flight requests, and writes cancelled results.

use std::io::Read;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Spawn `aiperf profile <args> --artifact-dir <dir> --tokenizer <model>` — the
/// unified native binary (the product entry point) — wait for the
/// "AIPerf System is PROFILING" log line, delay `sigint_delay`, send SIGINT, and
/// wait for the process to flush partial artifacts and exit. SIGINT is handled by
/// the native entry point's signal handler, which forwards graceful cancellation
/// to its `--execute` child.
///
/// Returns a `RunResult` reading the emitted artifact tree.
fn run_with_sigint(
    h: &AIPerfHarness,
    profile_args: &str,
    sigint_delay: Duration,
    wait_for_profiling: bool,
) -> RunResult {
    let bin = exec_binary();

    let mut full_args = vec!["profile".to_string()];
    for a in profile_args.split_whitespace() {
        full_args.push(a.to_string());
    }
    full_args.push("--artifact-dir".to_string());
    full_args.push(h.artifact_dir.path().display().to_string());
    full_args.push("--tokenizer".to_string());
    full_args.push(DEFAULT_MODEL.to_string());

    let mut cmd = Command::new(&bin);
    cmd.args(&full_args);
    cmd.env("HF_HUB_OFFLINE", "1")
        .env("TRANSFORMERS_OFFLINE", "1")
        .env("PYTHONUNBUFFERED", "1")
        .env("MALLOC_ARENA_MAX", "2")
        .env("AIPERF_EXEC_BIN", exec_binary())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = cmd
        .spawn()
        .unwrap_or_else(|e| panic!("failed to spawn `{bin}`: {e}"));

    // Incremental capture permits waiting for profiling without blocking either pipe.
    let out_buf = Arc::new(Mutex::new(String::new()));
    let err_buf = Arc::new(Mutex::new(String::new()));

    let mut out_pipe = child.stdout.take().expect("child stdout");
    let mut err_pipe = child.stderr.take().expect("child stderr");

    let out_buf_t = Arc::clone(&out_buf);
    let out_thread = std::thread::spawn(move || {
        let mut chunk = [0u8; 4096];
        loop {
            match out_pipe.read(&mut chunk) {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    let s = String::from_utf8_lossy(&chunk[..n]);
                    out_buf_t.lock().unwrap().push_str(&s);
                }
            }
        }
    });
    let err_buf_t = Arc::clone(&err_buf);
    let err_thread = std::thread::spawn(move || {
        let mut chunk = [0u8; 4096];
        loop {
            match err_pipe.read(&mut chunk) {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    let s = String::from_utf8_lossy(&chunk[..n]);
                    err_buf_t.lock().unwrap().push_str(&s);
                }
            }
        }
    });

    // The hard deadline prevents a hung child from wedging the test.
    let hard_deadline = Instant::now() + Duration::from_secs(120);

    if wait_for_profiling {
        loop {
            let seen = {
                let o = out_buf.lock().unwrap();
                let e = err_buf.lock().unwrap();
                o.contains("AIPerf System is PROFILING") || e.contains("AIPerf System is PROFILING")
            };
            if seen {
                break;
            }
            if child.try_wait().expect("try_wait").is_some() {
                break;
            }
            if Instant::now() >= hard_deadline {
                break;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    std::thread::sleep(sigint_delay);

    send_sigint(&child);

    let status = loop {
        match child.try_wait().expect("try_wait after SIGINT") {
            Some(s) => break s,
            None => {
                if Instant::now() >= hard_deadline {
                    let _ = child.kill();
                    break child.wait().expect("wait after SIGKILL");
                }
                std::thread::sleep(Duration::from_millis(50));
            }
        }
    };

    out_thread.join().ok();
    err_thread.join().ok();

    let stdout = Arc::try_unwrap(out_buf)
        .map(|m| m.into_inner().unwrap())
        .unwrap_or_default();
    let stderr = Arc::try_unwrap(err_buf)
        .map(|m| m.into_inner().unwrap())
        .unwrap_or_default();
    let exit_code = status.code().unwrap_or(-1);

    RunResult {
        exit_code,
        stdout,
        stderr,
        artifacts: ArtifactReader {
            dir: h.artifact_dir.path().to_path_buf(),
        },
    }
}

/// Send SIGINT to the child so aiperf's handler can flush partial artifacts.
#[cfg(unix)]
fn send_sigint(child: &std::process::Child) {
    use nix::sys::signal::{Signal, kill};
    use nix::unistd::Pid;
    let pid = Pid::from_raw(child.id() as i32);
    let _ = kill(pid, Signal::SIGINT);
}

/// Windows lacks POSIX SIGINT; the graceful-cancel test early-returns there, so
/// this stub is never invoked at runtime — it exists only so the crate compiles.
#[cfg(not(unix))]
fn send_sigint(_child: &std::process::Child) {}

/// Ctrl+C triggers graceful cancellation, writes all output files, and sets
/// was_cancelled=True.
#[tokio::test]
async fn test_ctrl_c_graceful_cancel_writes_results() {
    // Windows uses CTRL_C_EVENT/CTRL_BREAK_EVENT instead of POSIX SIGINT; graceful
    // Ctrl+C cancellation is not supported there.
    if cfg!(target_os = "windows") {
        return;
    }

    let h = AIPerfHarness::new().await;
    let r = run_with_sigint(
        &h,
        &format!(
            "--model {DEFAULT_MODEL} --url {} --endpoint-type chat --streaming \
             --benchmark-duration 20 --concurrency 5 --random-seed 42 --osl 100 \
             --workers-max 1 --ui simple",
            h.mock.url
        ),
        Duration::from_secs_f64(2.0),
        true,
    );

    assert!(!r.artifacts.json().is_null(), "JSON export should exist");
    assert!(!r.artifacts.csv().is_empty(), "CSV export should exist");
    let jsonl = r.artifacts.jsonl();
    assert!(!r.artifacts.inputs().is_null(), "Inputs file should exist");

    assert!(
        r.artifacts.was_cancelled(),
        "was_cancelled flag should be True after Ctrl+C"
    );

    assert!(!jsonl.is_empty(), "Should have some completed requests");

    for record in &jsonl {
        assert!(
            record.get("metrics").is_some(),
            "Record should have metrics"
        );
    }
}
