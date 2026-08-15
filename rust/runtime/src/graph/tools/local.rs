// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker-local persistent shell sandbox backed by Tokio child processes.

use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;

use async_trait::async_trait;
use bytes::{Buf, Bytes, BytesMut};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::Mutex;

use crate::clock::Clock;

use super::{ToolCommandResult, ToolSandbox, ToolSandboxError, WorkspaceSpec};

const TERMINAL_PREFIX: &[u8] = b"\0aiperf-terminal:";
const REAP_GRACE_NS: i64 = 1_000_000_000;

/// Process-launch settings for one persistent local shell session.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LocalProcessRequest {
    /// Host working directory inherited by every command in this session.
    pub workdir: PathBuf,
}

/// One worker-local spawned shell session.
#[async_trait(?Send)]
pub trait ProcessSession {
    /// Write protocol bytes to the persistent shell's standard input.
    async fn write_all(&mut self, command: &[u8]) -> Result<(), ToolSandboxError>;
    /// Append the next combined output bytes and return their count.
    async fn read(&mut self, output: &mut BytesMut) -> Result<usize, ToolSandboxError>;
    /// Request graceful termination of the session and its command descendants.
    async fn terminate(&mut self) -> Result<(), ToolSandboxError>;
    /// Force termination after graceful cleanup has exceeded its grace interval.
    async fn kill(&mut self) -> Result<(), ToolSandboxError>;
    /// Reap the session process without blocking the worker reactor.
    async fn wait(&mut self) -> Result<(), ToolSandboxError>;
}

/// Injectable creator of one persistent local shell process.
#[async_trait(?Send)]
pub trait ProcessSpawner {
    /// Start the session described by `request`.
    async fn spawn(
        &self,
        request: LocalProcessRequest,
    ) -> Result<Box<dyn ProcessSession>, ToolSandboxError>;
}

/// Tokio-process implementation used by the local sandbox in production.
#[derive(Clone, Copy, Debug, Default)]
pub struct TokioProcessSpawner;

#[async_trait(?Send)]
impl ProcessSpawner for TokioProcessSpawner {
    async fn spawn(
        &self,
        request: LocalProcessRequest,
    ) -> Result<Box<dyn ProcessSession>, ToolSandboxError> {
        let mut command = Command::new("bash");
        command
            .arg("-c")
            // Route all persistent-shell diagnostics through stdout before the
            // command loop starts so the protocol sees one arrival-ordered stream.
            .arg("exec 2>&1; exec bash --noprofile --norc")
            .current_dir(request.workdir)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());
        set_session_process_group(&mut command);
        let mut child = command
            .spawn()
            .map_err(|error| ToolSandboxError::new(format!("cannot start local shell: {error}")))?;
        let process_group_id = child.id().ok_or_else(|| {
            ToolSandboxError::new("local shell has no process identifier after spawn")
        })? as i32;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| ToolSandboxError::new("local shell did not expose standard input"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| ToolSandboxError::new("local shell did not expose standard output"))?;
        Ok(Box::new(TokioProcessSession {
            child,
            process_group_id,
            stdin,
            stdout,
        }))
    }
}

/// Tokio child-process session with a single combined output pipe.
struct TokioProcessSession {
    child: Child,
    process_group_id: i32,
    stdin: ChildStdin,
    stdout: ChildStdout,
}

#[async_trait(?Send)]
impl ProcessSession for TokioProcessSession {
    async fn write_all(&mut self, command: &[u8]) -> Result<(), ToolSandboxError> {
        self.stdin.write_all(command).await.map_err(|error| {
            ToolSandboxError::new(format!("cannot write local shell command: {error}"))
        })?;
        self.stdin.flush().await.map_err(|error| {
            ToolSandboxError::new(format!("cannot flush local shell command: {error}"))
        })
    }

    async fn read(&mut self, output: &mut BytesMut) -> Result<usize, ToolSandboxError> {
        self.stdout.read_buf(output).await.map_err(|error| {
            ToolSandboxError::new(format!("cannot read local shell output: {error}"))
        })
    }

    async fn terminate(&mut self) -> Result<(), ToolSandboxError> {
        signal_process_group(self.process_group_id, libc::SIGTERM)
    }

    async fn kill(&mut self) -> Result<(), ToolSandboxError> {
        signal_process_group(self.process_group_id, libc::SIGKILL)
    }

    async fn wait(&mut self) -> Result<(), ToolSandboxError> {
        self.child
            .wait()
            .await
            .map(|_| ())
            .map_err(|error| ToolSandboxError::new(format!("cannot reap local shell: {error}")))
    }
}

#[cfg(unix)]
fn set_session_process_group(command: &mut Command) {
    // SAFETY: `setsid` is async-signal-safe and runs in the forked child before
    // exec, giving timeout cleanup ownership over the shell and descendants.
    unsafe {
        command.pre_exec(|| {
            if libc::setsid() < 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
}

#[cfg(not(unix))]
fn set_session_process_group(_command: &mut Command) {}

#[cfg(unix)]
fn signal_process_group(
    process_group_id: i32,
    signal: libc::c_int,
) -> Result<(), ToolSandboxError> {
    // A negative process identifier targets the session/process group created by
    // `setsid`, including command descendants that outlive their direct parent.
    let result = unsafe { libc::kill(-process_group_id, signal) };
    if result == 0 || std::io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH) {
        Ok(())
    } else {
        Err(ToolSandboxError::new(format!(
            "cannot signal local shell process group: {}",
            std::io::Error::last_os_error()
        )))
    }
}

#[cfg(not(unix))]
fn signal_process_group(
    _process_group_id: i32,
    _signal: libc::c_int,
) -> Result<(), ToolSandboxError> {
    Err(ToolSandboxError::new(
        "local shell process-group cleanup requires a Unix host",
    ))
}

/// Persistent local sandbox for one trace-owned workspace.
pub struct LocalSessionSandbox {
    workspace: WorkspaceSpec,
    clock: Rc<dyn Clock>,
    spawner: Rc<dyn ProcessSpawner>,
    output_limit: usize,
    reap_grace_ns: i64,
    session: RefCell<Option<Box<dyn ProcessSession>>>,
    command_gate: Mutex<()>,
}

impl LocalSessionSandbox {
    /// Build a sandbox with a caller-selected process spawner and output bound.
    pub fn new(
        workspace: WorkspaceSpec,
        clock: Rc<dyn Clock>,
        spawner: Rc<dyn ProcessSpawner>,
        output_limit: usize,
    ) -> Self {
        Self {
            workspace,
            clock,
            spawner,
            output_limit,
            reap_grace_ns: REAP_GRACE_NS,
            session: RefCell::new(None),
            command_gate: Mutex::new(()),
        }
    }

    /// Build a production sandbox backed by Tokio process primitives.
    pub fn with_tokio_processes(
        workspace: WorkspaceSpec,
        clock: Rc<dyn Clock>,
        output_limit: usize,
    ) -> Self {
        Self::new(workspace, clock, Rc::new(TokioProcessSpawner), output_limit)
    }

    /// Build a sandbox with a testable grace interval for process reaping.
    pub fn with_reap_grace(
        workspace: WorkspaceSpec,
        clock: Rc<dyn Clock>,
        spawner: Rc<dyn ProcessSpawner>,
        output_limit: usize,
        reap_grace_ns: i64,
    ) -> Self {
        let mut sandbox = Self::new(workspace, clock, spawner, output_limit);
        sandbox.reap_grace_ns = reap_grace_ns.max(1);
        sandbox
    }

    async fn open_unlocked(&self) -> Result<(), ToolSandboxError> {
        if self.session.borrow().is_some() {
            return Ok(());
        }
        let session = self
            .spawner
            .spawn(LocalProcessRequest {
                workdir: PathBuf::from(&self.workspace.workdir),
            })
            .await?;
        self.session.replace(Some(session));
        Ok(())
    }

    async fn recycle_unlocked(&self) -> Result<(), ToolSandboxError> {
        let previous = self.session.borrow_mut().take();
        if let Some(mut session) = previous {
            self.reap(&mut *session).await?;
        }
        self.open_unlocked().await
    }

    async fn reap(&self, session: &mut dyn ProcessSession) -> Result<(), ToolSandboxError> {
        let terminate = session.terminate().await;
        let grace = self.clock.clone().sleep(self.reap_grace_ns);
        tokio::pin!(grace);
        tokio::select! {
            result = session.wait() => {
                let kill = session.kill().await;
                combine_cleanup_results(terminate, combine_cleanup_results(kill, result))
            }
            () = &mut grace => {
                let kill = session.kill().await;
                let final_grace = self.clock.clone().sleep(self.reap_grace_ns);
                tokio::pin!(final_grace);
                tokio::select! {
                    result = session.wait() => combine_cleanup_results(terminate, combine_cleanup_results(kill, result)),
                    () = &mut final_grace => Err(merge_cleanup_error(
                        combine_cleanup_results(terminate, kill).err(),
                        ToolSandboxError::new("local shell did not exit after SIGKILL"),
                    )),
                }
            }
        }
    }

    async fn discard_session(
        &self,
        mut session: Box<dyn ProcessSession>,
    ) -> Result<(), ToolSandboxError> {
        self.reap(&mut *session).await
    }

    async fn terminal_result(
        &self,
        session: &mut dyn ProcessSession,
        frame_prefix: &[u8],
        timeout_ns: Option<u64>,
    ) -> Result<CommandEnd, ToolSandboxError> {
        let timeout_ns = timeout_ns.filter(|timeout| *timeout > 0);
        let timer = timeout_ns.map(|timeout| {
            self.clock
                .clone()
                .sleep(timeout.min(i64::MAX as u64) as i64)
        });
        tokio::pin!(timer);
        let mut wire = BytesMut::new();
        let mut output = Vec::new();
        let mut is_output_truncated = false;
        loop {
            let mut chunk = BytesMut::with_capacity(4096);
            let read = async {
                let count = session.read(&mut chunk).await?;
                Ok::<_, ToolSandboxError>((count, chunk))
            };
            let (count, chunk) = match timer.as_mut().as_pin_mut() {
                Some(timer) => tokio::select! {
                    result = read => result?,
                    () = timer => {
                        capture_output(&wire, &mut output, &mut is_output_truncated, self.output_limit);
                        return Ok(CommandEnd::TimedOut { output, is_output_truncated });
                    }
                },
                None => read.await?,
            };
            if count == 0 {
                return Err(ToolSandboxError::new(
                    "local shell reached EOF before its terminal frame",
                ));
            }
            wire.extend_from_slice(&chunk);
            if let Some(result) = consume_terminal_frame(
                &mut wire,
                frame_prefix,
                &mut output,
                &mut is_output_truncated,
                self.output_limit,
            )? {
                return Ok(CommandEnd::Completed {
                    output,
                    is_output_truncated,
                    exit_code: result,
                });
            }
        }
    }
}

enum CommandEnd {
    Completed {
        output: Vec<u8>,
        is_output_truncated: bool,
        exit_code: i32,
    },
    TimedOut {
        output: Vec<u8>,
        is_output_truncated: bool,
    },
}

#[async_trait(?Send)]
impl ToolSandbox for LocalSessionSandbox {
    async fn open(&self) -> Result<(), ToolSandboxError> {
        let _command_turn = self.command_gate.lock().await;
        self.open_unlocked().await
    }

    async fn run(
        &self,
        command: &str,
        timeout_ns: Option<u64>,
    ) -> Result<ToolCommandResult, ToolSandboxError> {
        let _command_turn = self.command_gate.lock().await;
        self.open_unlocked().await?;
        let sentinel = uuid::Uuid::new_v4().simple().to_string();
        let frame_prefix = [TERMINAL_PREFIX, sentinel.as_bytes(), b":"].concat();
        let command_wire = command_wire(&self.workspace.interpreter, command, &sentinel)?;
        let mut session = self.session.borrow_mut().take().ok_or_else(|| {
            ToolSandboxError::new("local shell disappeared while starting a command")
        })?;
        let started_ns = self.clock.now_ns();
        if let Err(error) = session.write_all(&command_wire).await {
            self.discard_session(session).await?;
            return Err(error);
        }
        let timeout_ns = timeout_ns.or(Some(self.workspace.command_timeout_ns));
        match self
            .terminal_result(&mut *session, &frame_prefix, timeout_ns)
            .await
        {
            Ok(CommandEnd::Completed {
                output,
                is_output_truncated,
                exit_code,
            }) => {
                let duration_ns = elapsed_ns(started_ns, self.clock.now_ns());
                // End this command's outer session after its terminal frame.
                // The next command opens a fresh session, so descendants that
                // inherited this pipe cannot contaminate its output.
                self.discard_session(session).await?;
                Ok(ToolCommandResult {
                    output: Bytes::from(output),
                    exit_code,
                    duration_ns,
                    is_timed_out: false,
                    is_output_truncated,
                })
            }
            Ok(CommandEnd::TimedOut {
                output,
                is_output_truncated,
            }) => {
                // The measurement boundary ends before descendant termination,
                // grace waiting, and session recreation perturb wall-clock timing.
                let duration_ns = elapsed_ns(started_ns, self.clock.now_ns());
                self.discard_session(session).await?;
                self.open_unlocked().await?;
                Ok(ToolCommandResult {
                    output: Bytes::from(output),
                    exit_code: 124,
                    duration_ns,
                    is_timed_out: true,
                    is_output_truncated,
                })
            }
            Err(error) => {
                self.discard_session(session).await?;
                Err(error)
            }
        }
    }

    fn recovers_timed_out_commands(&self) -> bool {
        true
    }

    async fn recycle(&self) -> Result<(), ToolSandboxError> {
        let _command_turn = self.command_gate.lock().await;
        self.recycle_unlocked().await
    }

    async fn close(&self) -> Result<(), ToolSandboxError> {
        let _command_turn = self.command_gate.lock().await;
        let session = self.session.borrow_mut().take();
        if let Some(mut session) = session {
            self.reap(&mut *session).await?;
        }
        Ok(())
    }
}

fn command_wire(
    interpreter: &[String],
    command: &str,
    sentinel: &str,
) -> Result<Vec<u8>, ToolSandboxError> {
    if interpreter.is_empty() {
        return Err(ToolSandboxError::new(
            "local sandbox recipe has no command interpreter",
        ));
    }
    let interpreter = interpreter
        .iter()
        .map(|argument| shell_quote(argument))
        .collect::<Vec<_>>()
        .join(" ");
    Ok(format!(
        "{{ {interpreter} {}; status=$?; printf '\\0aiperf-terminal:{sentinel}:%d\\0' \"$status\"; }} 2>&1\n",
        shell_quote(command),
    )
    .into_bytes())
}

fn combine_cleanup_results(
    first: Result<(), ToolSandboxError>,
    second: Result<(), ToolSandboxError>,
) -> Result<(), ToolSandboxError> {
    match (first, second) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(error), Ok(())) | (Ok(()), Err(error)) => Err(error),
        (Err(first), Err(second)) => Err(merge_cleanup_error(Some(first), second)),
    }
}

fn merge_cleanup_error(
    first: Option<ToolSandboxError>,
    second: ToolSandboxError,
) -> ToolSandboxError {
    match first {
        Some(first) => ToolSandboxError::new(format!("{first}; {second}")),
        None => second,
    }
}

fn shell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}

fn elapsed_ns(started_ns: i64, ended_ns: i64) -> u64 {
    ended_ns.saturating_sub(started_ns) as u64
}

fn consume_terminal_frame(
    wire: &mut BytesMut,
    frame_prefix: &[u8],
    output: &mut Vec<u8>,
    is_output_truncated: &mut bool,
    output_limit: usize,
) -> Result<Option<i32>, ToolSandboxError> {
    if let Some(index) = find_subsequence(wire, frame_prefix) {
        capture_output(&wire[..index], output, is_output_truncated, output_limit);
        wire.advance(index);
        let frame = &wire[frame_prefix.len()..];
        let Some(status_end) = frame.iter().position(|byte| *byte == b'\0') else {
            return Ok(None);
        };
        let exit_code = std::str::from_utf8(&frame[..status_end])
            .ok()
            .and_then(|status| status.parse::<i32>().ok())
            .ok_or_else(|| {
                ToolSandboxError::new("local shell emitted a malformed terminal frame")
            })?;
        if frame.len() != status_end + 1 {
            return Err(ToolSandboxError::new(
                "local shell emitted bytes after its terminal frame",
            ));
        }
        return Ok(Some(exit_code));
    }
    let retained = frame_prefix.len().saturating_sub(1);
    let capture_end = wire.len().saturating_sub(retained);
    if capture_end > 0 {
        capture_output(
            &wire[..capture_end],
            output,
            is_output_truncated,
            output_limit,
        );
        wire.advance(capture_end);
    }
    Ok(None)
}

fn capture_output(
    bytes: &[u8],
    output: &mut Vec<u8>,
    is_output_truncated: &mut bool,
    output_limit: usize,
) {
    let remaining = output_limit.saturating_sub(output.len());
    let captured = bytes.len().min(remaining);
    output.extend_from_slice(&bytes[..captured]);
    *is_output_truncated |= captured != bytes.len();
}

fn find_subsequence(bytes: &[u8], needle: &[u8]) -> Option<usize> {
    bytes
        .windows(needle.len())
        .position(|window| window == needle)
}
