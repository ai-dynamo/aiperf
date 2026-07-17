// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Drive one execution child (`aiperf --execute`) over stdio for a single run.
//!
//! The request is written to stdin, lifecycle diagnostics are forwarded from
//! stderr, and exactly one terminal JSON line is read from stdout. [`crate::signals`]
//! forwards SIGINT to the published child PID so cancellation can drain and
//! produce a partial `was_cancelled=true` report.

use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;
use std::process::{Command, Stdio};
use std::thread;

use serde::Deserialize;

/// Fields consumed from the execution child's `run_terminal` envelope.
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
    /// Scalar failure detail accepted for forward compatibility.
    #[serde(default)]
    error: Option<String>,
    /// Typed failure diagnostics emitted in `RunTerminalV2.errors`.
    #[serde(default)]
    errors: Vec<TerminalDiagnostic>,
}

#[derive(Debug, Deserialize)]
struct TerminalDiagnostic {
    #[serde(default)]
    message: String,
}

/// Execution outcome consumed by the CLI.
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

/// Spawn one execution child and drive the stdio protocol to completion.
///
/// I/O and protocol faults return an error. A valid failure envelope returns
/// `Ok(Terminal { success: false, .. })`.
pub fn run_once(
    exec_bin: &Path,
    request_json: &[u8],
    child_pid: &crate::signals::ChildPid,
) -> anyhow::Result<Terminal> {
    let mut command = Command::new(exec_bin);
    command
        .arg(crate::execute_mode::EXECUTE_FLAG)
        // The child has no verbosity flags, so inherit the resolved directive.
        .env(crate::logging::LOG_ENV, crate::logging::current_directive())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    // The front door blocks SIGINT/SIGTERM in its main-thread mask so the
    // forwarder can `sigwait` them; that blocked mask is inherited across
    // fork+exec, so unblock it in the child or its `tokio::signal` graceful-
    // cancel listener never observes the forwarded SIGINT.
    unblock_signals_in_child(&mut command);
    let mut child = command.spawn().map_err(|e| {
        anyhow::anyhow!(
            "failed to spawn aiperf --execute ({}): {e}",
            exec_bin.display()
        )
    })?;
    child_pid.set(child.id());

    // The child consumes the complete request before writing stdout, avoiding a
    // pipe deadlock while the writer thread closes stdin.
    let mut stdin = child.stdin.take().expect("piped stdin");
    let payload = request_json.to_vec();
    let writer = thread::spawn(move || {
        // A BrokenPipe here means the child died early; the stdout/exit-code
        // path reports the real cause, so we swallow it.
        let _ = stdin.write_all(&payload);
        let _ = stdin.flush();
        drop(stdin);
    });

    let stderr = child.stderr.take().expect("piped stderr");
    let stderr_reader = thread::spawn(move || {
        let reader = BufReader::new(stderr);
        for line in reader.lines().map_while(Result::ok) {
            if !line.trim().is_empty() {
                // Route child diagnostics through both configured logging sinks.
                tracing::info!(target: "aiperf", "aiperf: {line}");
            }
        }
    });

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

/// Reset the child's signal mask for its graceful-cancellation listener.
///
/// The front-door process BLOCKS SIGINT/SIGTERM in its main-thread mask so the
/// [`crate::signals`] forwarder can `sigwait` them (see [`crate::signals::install`]).
/// `Command::spawn` fork+execs from that thread, so the child inherits the
/// blocked mask. Without this reset, forwarded signals remain pending and the
/// phase orchestrator cannot produce a partial cancelled report.
#[cfg(unix)]
fn unblock_signals_in_child(command: &mut Command) {
    use std::os::unix::process::CommandExt;
    // SAFETY: only async-signal-safe work runs in the post-fork/pre-exec child.
    // `SigSet::empty` touches only stack memory and `thread_set_mask` calls
    // `pthread_sigmask`, which POSIX lists as async-signal-safe.
    unsafe {
        command.pre_exec(|| {
            nix::sys::signal::SigSet::empty()
                .thread_set_mask()
                .map_err(|errno| std::io::Error::from_raw_os_error(errno as i32))
        });
    }
}

#[cfg(not(unix))]
fn unblock_signals_in_child(_command: &mut Command) {}

/// Parse stdout while enforcing the single-terminal-line contract.
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

    // Preserve every typed diagnostic, with the scalar field as a compatibility
    // fallback.
    let error = if !response.errors.is_empty() {
        let joined = response
            .errors
            .iter()
            .map(|d| d.message.as_str())
            .filter(|m| !m.is_empty())
            .collect::<Vec<_>>()
            .join("; ");
        (!joined.is_empty()).then_some(joined).or(response.error)
    } else {
        response.error
    };

    Ok(Terminal {
        success: response.success,
        returncode,
        report_path: response.report_path,
        error,
    })
}

#[cfg(test)]
#[cfg(unix)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

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
        let runner = fake_runner_raw("printf 'first\\nsecond\\n'");
        let err =
            run_once(runner.as_ref(), b"{}", &crate::signals::ChildPid::default()).unwrap_err();
        assert!(err.to_string().contains("exactly one terminal JSON line"));
    }
}
