// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Contract coverage for the worker-local persistent tool sandbox.

use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::future::{Future, pending};
use std::pin::Pin;
use std::rc::Rc;

use async_trait::async_trait;
use bytes::{Bytes, BytesMut};

use aiperf_runtime::clock::{Clock, RealClock};
use aiperf_runtime::graph::tools::{
    LocalProcessRequest, LocalSessionSandbox, ProcessSession, ProcessSpawner, ToolSandbox,
    ToolSandboxError, WorkspaceSpec,
};

enum ReadStep {
    Bytes(Vec<u8>),
    Pending,
}

struct FakeProcess {
    output_prefix: Vec<u8>,
    reads: VecDeque<ReadStep>,
    termination_count: Rc<Cell<u8>>,
}

impl FakeProcess {
    fn new(output_prefix: impl Into<Vec<u8>>, reads: impl IntoIterator<Item = ReadStep>) -> Self {
        Self {
            output_prefix: output_prefix.into(),
            reads: reads.into_iter().collect(),
            termination_count: Rc::new(Cell::new(0)),
        }
    }

    fn with_termination_count(
        output_prefix: impl Into<Vec<u8>>,
        reads: impl IntoIterator<Item = ReadStep>,
        termination_count: Rc<Cell<u8>>,
    ) -> Self {
        Self {
            output_prefix: output_prefix.into(),
            reads: reads.into_iter().collect(),
            termination_count,
        }
    }
}

#[async_trait(?Send)]
impl ProcessSession for FakeProcess {
    async fn write_all(&mut self, command: &[u8]) -> Result<(), ToolSandboxError> {
        let command = std::str::from_utf8(command).map_err(|error| {
            ToolSandboxError::new(format!("fake command is not UTF-8: {error}"))
        })?;
        let marker = command
            .rsplit("\\0aiperf-terminal:")
            .next()
            .and_then(|tail| tail.split(":%d\\0").next())
            .ok_or_else(|| ToolSandboxError::new("fake cannot find terminal marker"))?;
        let mut frame = self.output_prefix.clone();
        frame.extend_from_slice(b"\0aiperf-terminal:");
        frame.extend_from_slice(marker.as_bytes());
        frame.extend_from_slice(b":0\0");
        self.reads.push_back(ReadStep::Bytes(frame));
        Ok(())
    }

    async fn read(&mut self, output: &mut BytesMut) -> Result<usize, ToolSandboxError> {
        match self.reads.pop_front().unwrap_or(ReadStep::Pending) {
            ReadStep::Bytes(bytes) => {
                let count = bytes.len();
                output.extend_from_slice(&bytes);
                Ok(count)
            }
            ReadStep::Pending => pending().await,
        }
    }

    async fn terminate(&mut self) -> Result<(), ToolSandboxError> {
        self.termination_count
            .set(self.termination_count.get().saturating_add(1));
        Ok(())
    }

    async fn kill(&mut self) -> Result<(), ToolSandboxError> {
        Ok(())
    }

    async fn wait(&mut self) -> Result<(), ToolSandboxError> {
        Ok(())
    }
}

struct FakeSpawner {
    processes: RefCell<VecDeque<Box<dyn ProcessSession>>>,
    requests: RefCell<Vec<LocalProcessRequest>>,
}

struct FailingCleanupProcess;

#[async_trait(?Send)]
impl ProcessSession for FailingCleanupProcess {
    async fn write_all(&mut self, _command: &[u8]) -> Result<(), ToolSandboxError> {
        Ok(())
    }

    async fn read(&mut self, _output: &mut BytesMut) -> Result<usize, ToolSandboxError> {
        pending().await
    }

    async fn terminate(&mut self) -> Result<(), ToolSandboxError> {
        Err(ToolSandboxError::new("terminate failed"))
    }

    async fn kill(&mut self) -> Result<(), ToolSandboxError> {
        Ok(())
    }

    async fn wait(&mut self) -> Result<(), ToolSandboxError> {
        Ok(())
    }
}

struct NeverReapsProcess {
    kill_count: Rc<Cell<u8>>,
}

#[async_trait(?Send)]
impl ProcessSession for NeverReapsProcess {
    async fn write_all(&mut self, _command: &[u8]) -> Result<(), ToolSandboxError> {
        Ok(())
    }

    async fn read(&mut self, _output: &mut BytesMut) -> Result<usize, ToolSandboxError> {
        pending().await
    }

    async fn terminate(&mut self) -> Result<(), ToolSandboxError> {
        Ok(())
    }

    async fn kill(&mut self) -> Result<(), ToolSandboxError> {
        self.kill_count.set(self.kill_count.get().saturating_add(1));
        Ok(())
    }

    async fn wait(&mut self) -> Result<(), ToolSandboxError> {
        pending().await
    }
}

struct RecordingClock {
    sleeps: Rc<RefCell<Vec<i64>>>,
}

impl Clock for RecordingClock {
    fn now_ns(&self) -> i64 {
        0
    }

    fn sleep(self: Rc<Self>, duration_ns: i64) -> Pin<Box<dyn Future<Output = ()>>> {
        self.sleeps.borrow_mut().push(duration_ns);
        Box::pin(pending())
    }
}

impl FakeSpawner {
    fn new(processes: impl IntoIterator<Item = Box<dyn ProcessSession>>) -> Self {
        Self {
            processes: RefCell::new(processes.into_iter().collect()),
            requests: RefCell::new(Vec::new()),
        }
    }
}

#[async_trait(?Send)]
impl ProcessSpawner for FakeSpawner {
    async fn spawn(
        &self,
        request: LocalProcessRequest,
    ) -> Result<Box<dyn ProcessSession>, ToolSandboxError> {
        self.requests.borrow_mut().push(request);
        self.processes
            .borrow_mut()
            .pop_front()
            .ok_or_else(|| ToolSandboxError::new("fake has no process left"))
    }
}

fn workspace() -> WorkspaceSpec {
    WorkspaceSpec {
        files: Vec::new(),
        workdir: ".".into(),
        interpreter: vec!["bash".into(), "-c".into()],
        mount_workspace: true,
        command_timeout_ns: 1_000_000,
    }
}

fn real_workspace(workdir: &str) -> WorkspaceSpec {
    WorkspaceSpec {
        workdir: workdir.into(),
        ..workspace()
    }
}

fn sandbox(spawner: Rc<dyn ProcessSpawner>) -> LocalSessionSandbox {
    let clock: Rc<dyn Clock> = RealClock::new();
    LocalSessionSandbox::new(workspace(), clock, spawner, 1024)
}

#[tokio::test(flavor = "current_thread")]
async fn timeout_kills_descendants_recycles_and_next_command_has_no_stale_output() {
    // This catches a timeout path that leaves the old session readable, causing
    // a following command to observe bytes emitted by its terminated predecessor.
    let termination_count = Rc::new(Cell::new(0));
    let spawner = Rc::new(FakeSpawner::new([
        Box::new(FakeProcess::with_termination_count(
            b"",
            [
                ReadStep::Bytes(b"first command output".to_vec()),
                ReadStep::Pending,
            ],
            termination_count.clone(),
        )) as Box<dyn ProcessSession>,
        Box::new(FakeProcess::new(b"second command output", [])),
    ]));
    let sandbox = sandbox(spawner.clone());

    sandbox.open().await.expect("first session opens");
    let first = sandbox
        .run("sleep forever", Some(10_000_000))
        .await
        .expect("timeout recovers its session");
    let second = sandbox
        .run("printf second", None)
        .await
        .expect("recycled session accepts the next command");

    assert!(first.is_timed_out);
    assert_eq!(first.output, Bytes::from_static(b"first command output"));
    assert_eq!(second.output, Bytes::from_static(b"second command output"));
    assert_eq!(termination_count.get(), 1);
    assert_eq!(spawner.requests.borrow().len(), 2);
}

#[tokio::test(flavor = "current_thread")]
async fn timeout_refuses_to_report_a_terminal_outcome_when_cleanup_fails() {
    // This catches cleanup being treated as best-effort, which would report a
    // timeout and spawn a replacement while the old descendant group survives.
    let spawner = Rc::new(FakeSpawner::new([
        Box::new(FailingCleanupProcess) as Box<dyn ProcessSession>,
        Box::new(FakeProcess::new(b"replacement", [])) as Box<dyn ProcessSession>,
    ]));
    let sandbox = sandbox(spawner.clone());

    let error = sandbox
        .run("sleep forever", Some(10_000_000))
        .await
        .expect_err("failed cleanup prevents a timeout result");

    assert_eq!(error.to_string(), "terminate failed");
    assert_eq!(spawner.requests.borrow().len(), 1);

    let later = sandbox
        .run("printf replacement", None)
        .await
        .expect_err("cleanup failure poisons the sandbox");
    assert_eq!(
        later.to_string(),
        "local sandbox is poisoned after cleanup failure: terminate failed"
    );
    assert_eq!(spawner.requests.borrow().len(), 1);
}

#[tokio::test(flavor = "current_thread")]
async fn detaching_command_is_rejected_without_an_escaped_side_effect() {
    // This catches accepting `setsid`, which leaves a descendant outside the
    // shell process group and lets it write after the sandbox has recycled.
    let temporary = tempfile::tempdir().expect("temporary local workspace");
    let clock: Rc<dyn Clock> = RealClock::new();
    let sandbox = LocalSessionSandbox::with_tokio_processes(
        real_workspace(temporary.path().to_string_lossy().as_ref()),
        clock.clone(),
        1024,
    );

    let error = sandbox
        .run(
            "setsid bash -c 'sleep 0.05; printf escaped > escaped-state' &",
            Some(1_000_000_000),
        )
        .await
        .expect_err("detaching command is rejected before execution");
    clock.clone().sleep(100_000_000).await;
    sandbox.close().await.expect("opened session closes safely");

    assert_eq!(
        error.to_string(),
        "recorded-agent replay blocked a detaching command to preserve sandbox containment"
    );
    assert!(!temporary.path().join("escaped-state").exists());
}

#[tokio::test(flavor = "current_thread")]
async fn nested_detaching_constructs_are_rejected_before_execution() {
    // This catches a direct sandbox check that only inspects top-level words,
    // allowing a nested shell payload or command substitution to call `setsid`.
    let temporary = tempfile::tempdir().expect("temporary local workspace");
    let clock: Rc<dyn Clock> = RealClock::new();
    let sandbox = LocalSessionSandbox::with_tokio_processes(
        real_workspace(temporary.path().to_string_lossy().as_ref()),
        clock,
        1024,
    );

    for command in [
        "bash -c 'setsid true &'",
        "bash --norc -c 'setsid true'",
        "bash -c -- 'setsid true'",
        "bash -c -x 'setsid true'",
        "echo $(setsid true)",
        "true & setsid true",
        "printf ok | (setsid true)",
        "printf ok | { setsid true; }",
    ] {
        let error = sandbox
            .run(command, Some(1_000_000_000))
            .await
            .expect_err("nested detachment is rejected before execution");
        assert_eq!(
            error.to_string(),
            "recorded-agent replay blocked a detaching command to preserve sandbox containment",
            "{command}"
        );
    }
    sandbox.close().await.expect("opened session closes safely");
}

#[tokio::test(flavor = "current_thread")]
async fn completed_command_cannot_leave_background_output_for_the_next_command() {
    // This catches a command frame emitted before its background descendants
    // are terminated, which attributes their later output to the next command.
    let temporary = tempfile::tempdir().expect("temporary local workspace");
    let clock: Rc<dyn Clock> = RealClock::new();
    let sandbox = LocalSessionSandbox::with_tokio_processes(
        real_workspace(temporary.path().to_string_lossy().as_ref()),
        clock,
        1024,
    );

    sandbox
        .run("(sleep 0.05; printf stale) &", Some(1_000_000_000))
        .await
        .expect("command completes after cleaning background descendants");
    let next = sandbox
        .run("sleep 0.1; printf next", Some(1_000_000_000))
        .await
        .expect("following command completes");
    sandbox.close().await.expect("session closes");

    assert_eq!(next.output, Bytes::from_static(b"next"));
}

#[tokio::test(flavor = "current_thread")]
async fn timeout_kills_a_background_descendant_before_it_can_write() {
    // This catches a timeout that only kills the persistent shell while an
    // inner command process group survives and performs a later side effect.
    let temporary = tempfile::tempdir().expect("temporary local workspace");
    let clock: Rc<dyn Clock> = RealClock::new();
    let sandbox = LocalSessionSandbox::with_tokio_processes(
        real_workspace(temporary.path().to_string_lossy().as_ref()),
        clock,
        1024,
    );

    let timeout = sandbox
        .run(
            "(sleep 0.05; printf escaped > escaped-state) & sleep 10",
            Some(10_000_000),
        )
        .await
        .expect("timeout cleanup and replacement succeed");
    let next = sandbox
        .run(
            "sleep 0.1; test ! -e escaped-state && printf clean",
            Some(1_000_000_000),
        )
        .await
        .expect("next session runs after timeout recovery");
    sandbox.close().await.expect("session closes");

    assert!(timeout.is_timed_out);
    assert_eq!(next.output, Bytes::from_static(b"clean"));
}

#[tokio::test(flavor = "current_thread")]
async fn final_reap_after_kill_is_clock_bounded() {
    // This catches an unbounded second `wait()` after SIGKILL that can hold the
    // worker-local command gate indefinitely when a process implementation misbehaves.
    let kill_count = Rc::new(Cell::new(0));
    let spawner = Rc::new(FakeSpawner::new([Box::new(NeverReapsProcess {
        kill_count: kill_count.clone(),
    }) as Box<dyn ProcessSession>]));
    let clock: Rc<dyn Clock> = RealClock::new();
    let sandbox =
        LocalSessionSandbox::with_reap_grace(workspace(), clock, spawner, 1024, 1_000_000);

    sandbox.open().await.expect("session opens");
    let error = sandbox
        .close()
        .await
        .expect_err("unreaped process is a bounded infrastructure error");

    assert_eq!(error.to_string(), "local shell did not exit after SIGKILL");
    assert_eq!(kill_count.get(), 1);
}

#[tokio::test(flavor = "current_thread")]
async fn oversized_timeout_saturates_at_the_clock_range() {
    // This catches a `u64 as i64` wrap that turns a very large authored timeout
    // into a negative value and silently disables the deadline.
    let sleeps = Rc::new(RefCell::new(Vec::new()));
    let clock: Rc<dyn Clock> = Rc::new(RecordingClock {
        sleeps: sleeps.clone(),
    });
    let spawner = Rc::new(FakeSpawner::new([
        Box::new(FakeProcess::new(b"ok", [])) as Box<dyn ProcessSession>
    ]));
    let sandbox = LocalSessionSandbox::new(workspace(), clock, spawner, 1024);

    sandbox
        .run("printf ok", Some(u64::MAX))
        .await
        .expect("saturated deadline still permits a completed command");

    assert_eq!(sleeps.borrow().first(), Some(&i64::MAX));
}

#[tokio::test(flavor = "current_thread")]
async fn sentinel_like_output_is_not_terminal_frame() {
    // This catches a delimiter parser that accepts any sentinel-shaped bytes
    // instead of only the unique frame generated for this command.
    let spawner = Rc::new(FakeSpawner::new([Box::new(FakeProcess::new(
        b"before\0aiperf-terminal:not-this-command:0\0after",
        [],
    )) as Box<dyn ProcessSession>]));
    let sandbox = sandbox(spawner);

    let result = sandbox
        .run("printf output", None)
        .await
        .expect("unique terminal frame completes the command");

    assert_eq!(
        result.output,
        Bytes::from_static(b"before\0aiperf-terminal:not-this-command:0\0after")
    );
}

#[tokio::test(flavor = "current_thread")]
async fn invalid_recipe_does_not_lose_an_open_session() {
    // This catches validation after taking session ownership, which drops a
    // live process instead of letting the normal close path reap it.
    let termination_count = Rc::new(Cell::new(0));
    let spawner = Rc::new(FakeSpawner::new([
        Box::new(FakeProcess::with_termination_count(
            b"",
            [],
            termination_count.clone(),
        )) as Box<dyn ProcessSession>,
    ]));
    let mut invalid_workspace = workspace();
    invalid_workspace.interpreter.clear();
    let clock: Rc<dyn Clock> = RealClock::new();
    let sandbox = LocalSessionSandbox::new(invalid_workspace, clock, spawner, 1024);

    sandbox
        .open()
        .await
        .expect("session opens before validation");
    let error = sandbox
        .run("printf ignored", None)
        .await
        .expect_err("empty interpreter is rejected");
    sandbox.close().await.expect("still-owned session closes");

    assert_eq!(
        error.to_string(),
        "local sandbox recipe has no command interpreter"
    );
    assert_eq!(termination_count.get(), 1);
}

#[tokio::test(flavor = "current_thread")]
async fn persistent_workspace_uses_fresh_interpreters_and_reports_truncation() {
    // This catches a shell loop that loses filesystem state, leaks one command's
    // shell options into the next, or silently drops bytes after its output cap.
    let temporary = tempfile::tempdir().expect("temporary local workspace");
    let clock: Rc<dyn Clock> = RealClock::new();
    let sandbox = LocalSessionSandbox::with_tokio_processes(
        real_workspace(temporary.path().to_string_lossy().as_ref()),
        clock.clone(),
        1024,
    );

    let write = sandbox
        .run(
            "printf value > shared-state; set -o pipefail; false | true",
            Some(1_000_000_000),
        )
        .await
        .expect("first fresh interpreter finishes");
    let read = sandbox
        .run("cat shared-state; false | true", Some(1_000_000_000))
        .await
        .expect("second fresh interpreter sees workspace state");
    sandbox.close().await.expect("session closes idempotently");
    sandbox.close().await.expect("second close is a no-op");
    let bounded = LocalSessionSandbox::with_tokio_processes(
        real_workspace(temporary.path().to_string_lossy().as_ref()),
        clock,
        3,
    );
    let truncated = bounded
        .run("printf 12345", Some(1_000_000_000))
        .await
        .expect("bounded command output completes");
    bounded.close().await.expect("bounded session closes");

    assert_eq!(write.exit_code, 1);
    assert_eq!(read.exit_code, 0);
    assert_eq!(read.output, Bytes::from_static(b"value"));
    assert_eq!(truncated.output, Bytes::from_static(b"123"));
    assert!(truncated.is_output_truncated);
}
