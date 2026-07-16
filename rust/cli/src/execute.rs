// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Drive one execution child (`aiperf --execute`) over stdio for a single run.
//!
//! There is no separate runner binary: the entry point re-execs itself in the
//! internal `--execute` mode ([`crate::execute_mode`]). The protocol is unchanged:
//! write the request JSON to the child's stdin and close it; the child streams
//! human-readable lifecycle/readiness lines to stderr and writes exactly one
//! terminal JSON line to stdout. This module forwards the child's stderr to our
//! stderr live (so readiness/progress is visible while the run is in flight) and
//! parses the single terminal line.
//!
//! Ported from `src/aiperf/orchestrator/rust_executor.py::_parse_terminal` and
//! `runner_installation._communicate_forwarding_signals`. Graceful Ctrl+C
//! cancellation is handled by [`crate::signals`]: `run_once` publishes the child
//! PID so the forwarder delivers one SIGINT to the child (which drains + writes a
//! partial `was_cancelled=true` report) instead of an abrupt kill.

use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;
use std::process::{Command, Stdio};
use std::thread;

use serde::Deserialize;

/// The runner's single terminal response line, deserialized with typed field
/// access (no `serde_json::Value` poking). Mirrors the `run_terminal` envelope
/// the runner writes to stdout; unknown fields are ignored so the CLI reads only
/// what it acts on.
#[derive(Debug, Deserialize)]
struct TerminalResponse {
    /// Wire protocol discriminator; must equal `2`.
    protocol_version: u32,
    /// Envelope discriminator; must equal `"run_terminal"`.
    event: String,
    /// Whether the run committed a report successfully.
    success: bool,
    /// Absolute path to the committed `native-v2.json`, present on success.
    #[serde(default)]
    report_path: Option<String>,
    /// Human-readable failure detail, present on failure.
    #[serde(default)]
    error: Option<String>,
}

/// The runner's terminal outcome, reduced to the fields the CLI acts on plus the
/// observed process exit code.
#[derive(Debug)]
pub struct Terminal {
    /// Whether the run committed a report successfully.
    pub success: bool,
    /// The child's process exit code (`-1` if terminated by signal).
    pub returncode: i32,
    /// Absolute path to the committed `native-v2.json`, present on success.
    pub report_path: Option<String>,
    /// Human-readable failure detail, present on failure.
    pub error: Option<String>,
}

/// Spawn the runner once, send `request_json`, forward stderr live, and parse the
/// terminal line.
///
/// Returns an error only for I/O/protocol faults (spawn failed, no single JSON
/// line, malformed JSON, wrong envelope fields). A run that fails *cleanly*
/// returns `Ok(Terminal { success: false, .. })` so the caller can render the
/// runner's own error.
pub fn run_once(
    exec_bin: &Path,
    request_json: &[u8],
    child_pid: &crate::signals::ChildPid,
) -> anyhow::Result<Terminal> {
    let mut child = Command::new(exec_bin)
        .arg(crate::execute_mode::EXECUTE_FLAG)
        // Hand the resolved log-level directive to the child: its argv is just
        // `--execute`, so the parent's level is the only way it inherits one
        // (mirrors Python, where the parent owns the log config).
        .env(crate::logging::LOG_ENV, crate::logging::current_directive())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| {
            anyhow::anyhow!(
                "failed to spawn aiperf --execute ({}): {e}",
                exec_bin.display()
            )
        })?;
    // Publish the PID so the signal forwarder can deliver a graceful SIGINT.
    child_pid.set(child.id());

    // Write the request on a dedicated thread and close stdin so the runner can
    // begin. Writing before we drain stdout is safe: the runner reads its whole
    // request before emitting anything.
    let mut stdin = child.stdin.take().expect("piped stdin");
    let payload = request_json.to_vec();
    let writer = thread::spawn(move || {
        // A BrokenPipe here means the child died early; the stdout/exit-code
        // path reports the real cause, so we swallow it.
        let _ = stdin.write_all(&payload);
        let _ = stdin.flush();
        drop(stdin);
    });

    // Forward stderr line-by-line to our stderr, live.
    let stderr = child.stderr.take().expect("piped stderr");
    let stderr_reader = thread::spawn(move || {
        let reader = BufReader::new(stderr);
        for line in reader.lines().map_while(Result::ok) {
            if !line.trim().is_empty() {
                // Forward each child (execution-engine) stderr line through our
                // own tracing so it reaches the console AND `logs/aiperf.log`,
                // mirroring Python's `_forward_runner_stderr_line`
                // (`logger.info("aiperf-runner: %s", line)`).
                tracing::info!(target: "aiperf_runner", "aiperf-runner: {line}");
            }
        }
    });

    // Drain stdout on the main thread.
    let mut stdout = child.stdout.take().expect("piped stdout");
    let mut out = Vec::new();
    stdout
        .read_to_end(&mut out)
        .map_err(|e| anyhow::anyhow!("failed to read execution child stdout: {e}"))?;

    let status = child
        .wait()
        .map_err(|e| anyhow::anyhow!("failed to wait for execution child: {e}"))?;
    child_pid.clear();
    let _ = writer.join();
    let _ = stderr_reader.join();

    let returncode = status.code().unwrap_or(-1);
    parse_terminal(&out, returncode)
}

/// Parse the runner's stdout into a [`Terminal`], enforcing the "exactly one
/// terminal JSON line" contract via typed deserialization (no `Value` poking).
/// Mirrors `rust_executor._parse_terminal`.
fn parse_terminal(stdout: &[u8], returncode: i32) -> anyhow::Result<Terminal> {
    let text = String::from_utf8_lossy(stdout);
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    anyhow::ensure!(
        lines.len() == 1,
        "execution child must write exactly one terminal JSON line to stdout; received {} non-empty lines (exit {returncode})",
        lines.len()
    );

    let response: TerminalResponse = serde_json::from_str(lines[0])
        .map_err(|e| anyhow::anyhow!("execution child returned invalid terminal JSON: {e}"))?;

    anyhow::ensure!(
        response.protocol_version == 2,
        "execution terminal protocol_version {} != 2",
        response.protocol_version
    );
    anyhow::ensure!(
        response.event == "run_terminal",
        "execution terminal event {:?} != run_terminal",
        response.event
    );

    Ok(Terminal {
        success: response.success,
        returncode,
        report_path: response.report_path,
        error: response.error,
    })
}

#[cfg(test)]
#[cfg(unix)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    /// A tiny shell script standing in for `aiperf-runner`: discards stdin and
    /// prints `script_body` verbatim (the caller supplies the emit commands).
    fn fake_runner_raw(script_body: &str) -> tempfile::TempPath {
        let mut f = tempfile::Builder::new().suffix(".sh").tempfile().unwrap();
        writeln!(f, "#!/bin/sh\ncat >/dev/null\n{script_body}").unwrap();
        f.flush().unwrap();
        let path = f.into_temp_path();
        let mut perms = std::fs::metadata(&path).unwrap().permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&path, perms).unwrap();
        path
    }

    /// A fake runner that emits `line` as its single stdout line.
    fn fake_runner(line: &str) -> tempfile::TempPath {
        fake_runner_raw(&format!("printf '%s\\n' '{line}'"))
    }

    #[test]
    fn parses_successful_terminal_line() {
        let runner = fake_runner(
            r#"{"protocol_version":2,"event":"run_terminal","benchmark_id":"b1","success":true,"report_path":"/tmp/x/native-v2.json"}"#,
        );
        let terminal =
            run_once(runner.as_ref(), b"{}", &crate::signals::ChildPid::default()).unwrap();
        assert!(terminal.success);
        assert_eq!(
            terminal.report_path.as_deref(),
            Some("/tmp/x/native-v2.json")
        );
    }

    #[test]
    fn rejects_multiple_stdout_lines() {
        // Two genuine stdout lines must violate the one-terminal-line contract.
        let runner = fake_runner_raw("printf 'first\\nsecond\\n'");
        let err =
            run_once(runner.as_ref(), b"{}", &crate::signals::ChildPid::default()).unwrap_err();
        assert!(err.to_string().contains("exactly one terminal JSON line"));
    }
}
